# index ((name, (start, stop, step)), (name, (start, stop, step)), (name, None))
from copy import deepcopy
from graph.types.fusions import FusionInputParameters, FusionOutputParameters
from graph.types.others import StridedSliceParameters, TransposeParameters
from graph.types import ConcatParameters


def reverse_transpose(trans):
    return [trans.index(idx) for idx in range(len(trans))]


def do_transpose(trans, shape):
    return [shape[idx] for idx in trans]


def explore_node(subg, node, state, out_slice=None, in_slice=None):
    dest_slice_sets = []
    src_slice_sets = []
    if isinstance(node, ConcatParameters):
        # out_slice needs to be reduced to the indexes range in the output of each portion
        # in_slice needs to be reduced to the elements of the portion that make up that output
        dest_start = out_slice[node.axis][1][0]
        dest_end = out_slice[node.axis][1][1]
        dest_stride = out_slice[node.axis][1][2]
        dest_next = dest_start
        in_idx = 0
        for edge in subg.indexed_in_edges(node.name):
            in_shape = node.in_dims[edge.to_idx].shape
            in_end = in_idx + in_shape[node.axis]
            seg_dest_start = dest_start + in_idx * dest_stride
            seg_dest_end = min(dest_start + in_end * dest_stride, dest_end)
            # test if this portion of the concat is present in the output
            if dest_next > seg_dest_end or dest_next > dest_end:
                dest_slice_sets.append(None)
                in_idx = in_end
                continue
            dest_slice_set = [
                [out_slice[idx][0], out_slice[idx][1] if idx != node.axis else [seg_dest_start, seg_dest_end, dest_stride]]
                for idx, _ in enumerate(in_shape)]
            dest_slice_sets.append(dest_slice_set)
            src_seg_start = dest_next - seg_dest_start
            src_seg_end = min(seg_dest_end - seg_dest_start, in_end)
            src_slice_set = [
                [in_slice[idx][0], in_slice[idx][1] if idx != node.axis else [src_seg_start, src_seg_end, dest_stride]]
                for idx, _ in enumerate(in_shape)]
            src_slice_sets.append(src_slice_set)
            dest_next += ((src_seg_end - src_seg_start) // dest_stride) * dest_stride
            in_idx = in_end
        explore_edges(src_slice_sets, dest_slice_sets, state, node, subg)
    elif isinstance(node, TransposeParameters):
        dest_slice_sets = [do_transpose(reverse_transpose(node.transpose_in[0]), out_slice)]
        src_slice_sets = [do_transpose(reverse_transpose(node.transpose_in[0]), in_slice)]
        explore_edges(src_slice_sets, dest_slice_sets, state, node, subg)
    elif isinstance(node, StridedSliceParameters):
        # in_slice needs to be expanded to the indexes before the slice
        assert len(node.act_slice) == len(out_slice)
        dest_slice_sets = [out_slice]
        src_slice_set = []
        for idx, sl in enumerate(node.act_slice):
            in_sl_name = in_slice[idx][0]
            in_sl = in_slice[idx][1]
            # express the range of the slice in terms of the input
            src_seg_start = in_sl[0] + sl[0]
            src_seg_end = src_seg_start + (in_sl[1] - in_sl[0]) * sl[2]
            src_seg_step = in_sl[2] * sl[2]
            src_slice_set.append([in_sl_name, [src_seg_start, src_seg_end, src_seg_step]])
        src_slice_sets = [src_slice_set]
        explore_edges(src_slice_sets, dest_slice_sets, state, node, subg)
    elif isinstance(node, FusionOutputParameters):
        slice_set = [[(node.name, idx), [0, dim, 1]] for idx, dim in enumerate(node.dims.shape)]
        src_slice_sets = [slice_set]
        dest_slice_sets = [slice_set]
        explore_edges(src_slice_sets, dest_slice_sets, state, node, subg)
    elif isinstance(node, FusionInputParameters):
        in_state = state.setdefault(node, [])
        in_state.append((in_slice, sorted(out_slice, key=lambda x: x[0])))
    else:
        raise ValueError(f"don't know how to handle {node.name} {node.__class__.__name__}")

def explore_edges(src_slice_sets, dest_slice_sets, state, node, subg):
    state[node] = (src_slice_sets, dest_slice_sets)

    for edge in subg.indexed_in_edges(node.name):
        explore_node(subg, edge.from_node, state,
                     out_slice=dest_slice_sets[edge.to_idx], in_slice=src_slice_sets[edge.to_idx])


def eval_copy(subg):
    state = {}
    for node in subg.outputs():
        explore_node(subg, node, state)
    return {
        'inputs': tuple((node.name, state[node]) for node in subg.inputs()),
        'outputs': tuple((node.name, state[node]) for node in subg.outputs())
    }


# group copies with compatible directions on all axis on the same inputs and outputs (i.e. they all have to be the same traversal)
# or could read inputs and outputs in both directions?
# each will be a separate user kernel
# 
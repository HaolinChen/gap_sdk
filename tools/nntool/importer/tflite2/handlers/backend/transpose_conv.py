# Copyright (C) 2020  GreenWaves Technologies, SAS

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from graph.dim import Conv2DFilterDim, PadDim, StrideDim
from graph.types import NNEdge, TransposeConv2DParameters
from importer.common.constant_mixin import ConstantMixin
from importer.common.provisional_dim import ProvisionalDim
from importer.tflite2.common.tflite_node import TFLiteNode
from importer.tflite2.tflite_schema_head.Padding import Padding
from importer.tflite2.tflite_schema_head.TransposeConvOptions import \
    TransposeConvOptions

from ..backend_handler import BackendHandler
from ..handler import tflite_op
from .filter_mixin import FilterMixin


@tflite_op("TRANSPOSE_CONV")
class TransposeConv(ConstantMixin, FilterMixin, BackendHandler):

    @classmethod
    def version_1(cls, node: TFLiteNode, **kwargs):
        node_opts = node.get_options(TransposeConvOptions)
        G = kwargs['G']
        opts = kwargs['opts']
        all_nodes = kwargs['all_nodes']

        inputs = [all_nodes[t] for t in node.input]
        x = inputs[2]
        x_shape = x[2].shape
        in_b, in_h, in_w, in_c = tuple(x_shape)
        pout_shape = [dim if x_shape[idx] is not None else None for idx,
                      dim in enumerate(cls.get_constant(inputs[0]))]
        out_b, out_h, out_w, out_c = tuple(pout_shape)

        filt = inputs[1]
        weights_node = filt[0]
        filt_shape = filt[2].shape
        # # ['in_c', 'h', 'w', 'out_c']
        filt_out_c, filt_h, filt_w, filt_in_c = tuple(filt_shape)

        filt_dim = Conv2DFilterDim(filt_h, filt_w,
                                   filt_out_c, in_c=filt_in_c)
        filt_dim = filt_dim.impose_order(cls.TF_LITE_FILTER_ORDER)

        stride_w = node_opts.StrideW()
        stride_h = node_opts.StrideH()
        # compute padding
        pad = node_opts.Padding()
        if pad == Padding.SAME:
            pad_h = ((in_h - 1) * stride_h + filt_h - out_h)
            pad_w = ((in_w - 1) * stride_w + filt_w - out_w)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad = PadDim(pad_top, pad_h - pad_top, pad_left,
                         pad_w - pad_left, same_type='balanced_right')
        else:
            pad = PadDim(0)

        params = TransposeConv2DParameters(
            node.name,
            filt=filt_dim,
            stride=StrideDim(
                stride_h, stride_w),
            padding=pad,
            in_dims_hint=[['h', 'w', 'c'], cls.TF_LITE_FILTER_ORDER.copy()],
            out_dims_hint=[['h', 'w', 'c']],
            constant_store=G.constant_store)
        G.add_edge(
            NNEdge(from_node=x[0], to_node=params, from_idx=x[1], to_idx=0))
        G.add_edge(NNEdge(from_node=weights_node, to_node=params, to_idx=1))
        pout_dims = ProvisionalDim(pout_shape)

        all_nodes[node.output[0]] = (params, 0, pout_dims)
        return params

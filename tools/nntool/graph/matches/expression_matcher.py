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

from graph.types.others import QuantizeParameters
import logging

from expressions.symbolic.symbol import Symbol, SymbolStats
from graph.types import ConstantInputParameters, ExpressionFusionParameters
from graph.types.base import CanFuseToExpression, NNEdge
from quantization.new_qrec import QRec
from quantization.unified_quantizer import UnifiedQuantizer
from utils.graph import GraphView
from utils.node_id import NodeId

from .matcher import Matcher

LOG = logging.getLogger("nntool." + __name__)


def add_edge(edge_set, edge):
    key = (edge.from_node, edge.from_idx)
    con_group = edge_set.get(key)
    if not con_group:
        con_group = set()
        edge_set[key] = con_group
    con_group.add(edge)


def group_edges(G, node_set):
    internal_edges = set()
    in_edges = {}
    out_edges = {}
    for node in node_set:
        for in_edge in G.in_edges(node.name):
            if in_edge.from_node in node_set:
                internal_edges.add(in_edge)
            else:
                add_edge(in_edges, in_edge)
        for out_edge in G.out_edges(node.name):
            if out_edge.to_node in node_set:
                internal_edges.add(out_edge)
            else:
                add_edge(out_edges, out_edge)
    return in_edges, out_edges, internal_edges


class ExpressionMatcher(Matcher):
    NAME = "expression_matcher"
    DESCRIPTION = "Groups piecewise expressions for kernel generation"
    NEEDS_VALID_DIMENSION = True

    def __init__(self, *args, **kwargs):
        super(ExpressionMatcher, self).__init__(*args, **kwargs)
        self._expr_num = 0

    def match(self, G: GraphView, set_identity: bool = True, **kwargs):
        has_modified_graph = False
        to_quantize = []
        for node_set in self.find_sets(G):
            Symbol.set_default_control(SymbolStats())
            has_modified_graph = True
            in_edges, out_edges, internal_edges = group_edges(G, node_set)
            frag = GraphView()
            for edge in internal_edges:
                frag.add_edge(edge)
            in_mapping = [[(edge.to_node, edge.to_idx) for edge in edge_group]
                          for edge_group in in_edges.values()]
            in_dims = [from_node.out_dims[from_idx]
                       for from_node, from_idx in in_edges]
            out_dims = [from_node.out_dims[from_idx]
                        for from_node, from_idx in out_edges]
            out_mapping = list(out_edges.keys())
            constant_inputs = [node_edge_idx[0]
                               for node_edge_idx in in_edges
                               if isinstance(node_edge_idx[0], ConstantInputParameters)]
            LOG.debug("inputs coming from: %s",
                      ",".join(f"{from_node.__repr__()}:{from_idx}"
                      for from_node, from_idx in in_edges))
            LOG.info("fusing nodes: %s into expr_%s",
                     ",".join(node.__repr__() for node in node_set),
                     self._expr_num)
            expr = ExpressionFusionParameters(f"expr_{self._expr_num}",
                                              subgraph=frag,
                                              qrecs=G.quantization,
                                              input_mapping=in_mapping,
                                              output_mapping=out_mapping,
                                              in_dims=in_dims,
                                              out_dims=out_dims,
                                              constant_inputs=constant_inputs)
            in_edge_mapping = list(in_edges.keys())
            out_edge_mapping = [[(edge.to_node, edge.to_idx) for edge in edge_set] for edge_set in
                                out_edges.values()]
            G.replace_fragment(frag,
                               expr,
                               frag_in_edges=list(
                                   set.union(*in_edges.values())),
                               frag_out_edges=list(
                                   set.union(*out_edges.values())),
                               edge_in_mapping=in_edge_mapping,
                               edge_out_mapping=out_edge_mapping,
                               edge_class=NNEdge
                               )
            if G.quantization:
                qrecs = G.quantization
                in_qs = [qrecs[NodeId(in_map[0][0])].in_qs[in_map[0][1]]
                         for in_map in in_mapping]
                out_qs = [qrecs[NodeId(node)].out_qs[idx]
                          for node, idx in out_mapping]
                stats = Symbol.CURRENT_CONTROL.stats
                func_col = expr.func_col
                for idx, qtype in enumerate(in_qs):
                    symbol = func_col.variables[func_col.input_names[idx]]
                    stats[symbol.name] = {
                        'min': qtype.min_val, 'max': qtype.max_val}
                for idx, qtype in enumerate(out_qs):
                    symbol = func_col.variables[func_col.output_names[idx]]
                    stats[symbol.name] = {
                        'min': qtype.min_val, 'max': qtype.max_val}
                G.quantization[NodeId(expr)] = QRec(
                    in_qs=in_qs, out_qs=out_qs, expression=stats, ktype='scaled')
                # delete any quantize parameters on outputs to allow the quantizer
                # to fuse them into the expression
                out_edges = G.out_edges(expr.name)
                for edge in out_edges:
                    if isinstance(edge.to_node, QuantizeParameters):
                        G.remove_and_reconnect(edge.to_node)
                        if NodeId(edge.to_node) in G.quantization:
                            del G.quantization[NodeId(edge.to_node)]
                to_quantize.append(expr)

            self._expr_num += 1

        if to_quantize:
            quantizer = UnifiedQuantizer.from_quantized_graph(G)
            G.quantization = quantizer.quantize(G, start_nodes=to_quantize)

        if set_identity:
            self.set_identity(G)

        return has_modified_graph

    @staticmethod
    def can_find_up(G, node, to_find):
        if node in to_find:
            return True
        in_edges = G.in_edges(node.name)
        if not in_edges:
            return False
        return any(ExpressionMatcher.can_find_up(G, edge.from_node, to_find)
                   for edge in in_edges)

    @staticmethod
    def explore(G, node, in_idx=None, curset=None):
        if curset is None:
            curset = set()
        if not isinstance(node, CanFuseToExpression):
            return curset
        if in_idx is not None:
            if any(ExpressionMatcher.can_find_up(G, edge.from_node, curset)
                   for edge in G.in_edges(node.name) if edge.to_idx != in_idx):
                return curset
        curset.add(node)
        for edge in G.out_edges(node.name):
            ExpressionMatcher.explore(
                G, edge.to_node, edge.to_idx, curset=curset)
        return curset

    @staticmethod
    def add_constants(G, node_set):
        result = node_set.copy()
        for node in node_set:
            result.add(node)
            for edge in G.in_edges(node.name):
                if isinstance(edge.from_node, ConstantInputParameters) and edge.from_node.out_dims[0].size() == 1:
                    result.add(edge.from_node)
        return result

    def find_sets(self, G):
        # find all nodes that are CanFuseToExpression
        candidate_node_set = set(
            [node for node in G.nodes() if isinstance(node, CanFuseToExpression)])
        explore_node_set = set()
        # filter out the CanFuseToExpression nodes that have immediate predecessors in the nodes found into a new set
        # ignore constant inputs since they will get fused later
        for node in candidate_node_set:
            if (not all(edge.from_node in candidate_node_set or isinstance(edge.from_node, ConstantInputParameters)
                        for edge in G.in_edges(node.name))):
                explore_node_set.add(node)
        # look down from each of these and absorb branches that are just CanFuseToExpression nodes
        node_sets = [set(self.explore(G, node)) for node in explore_node_set]
        # check if each node in each subset should fuse
        # if there is a shared node i.e. any intersection then we can fuse
        condensed_node_sets = []
        while node_sets:
            cur_set = node_sets.pop(0)
            found_sets = []
            for node_set in node_sets:
                if cur_set.intersection(node_set):
                    found_sets.append(node_set)
            for found_set in found_sets:
                cur_set |= found_set
                node_sets.remove(found_set)
            condensed_node_sets.append(cur_set)

        results = []
        for node_set in condensed_node_sets:
            while True:
                remove_node = None
                for node in node_set:
                    nid = NodeId(node)
                    qrec = G.quantization[NodeId(node)] if G.quantization and nid in G.quantization else None
                    if not node.should_fuse(node_set, qrec=qrec):
                        remove_node = node
                        break
                if remove_node:
                    node_set.remove(remove_node)
                else:
                    if node_set:
                        results.append(self.add_constants(G, node_set))
                    break
        # sort the results by sorted names in each group to guarantee consistent results
        # since sets above can yield different orders
        results.sort(key=lambda x: sorted([node.name for node in x]))
        return results

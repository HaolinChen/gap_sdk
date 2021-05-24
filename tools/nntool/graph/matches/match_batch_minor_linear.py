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

import logging
from copy import deepcopy

from graph.matches.matcher import Matcher
from graph.types import (ConvFusionParameters, FcParameters, NNEdge,
                         TransposeParameters)
from quantization.new_qrec import QRec
from utils.graph import GraphView
from utils.node_id import NodeId

LOG = logging.getLogger("nntool." + __name__)


class MatchBatchMinorLinear(Matcher):
    NAME = "match_batch_minor_linear"
    DESCRIPTION = "Extract transposes from batch minor linear layer"
    MODIFIES_DIMENSIONS = True

    def match(self, G: GraphView, set_identity: bool = True, **kwargs):
        # get a list of all the nodes that are transposable but not transposes
        # Need to do this first to avoid mutating it when doing the modifications
        batch_minor_nodes = [(node, node) for node in G.nodes(
            node_classes=FcParameters) if node.batch_size > 1 and node.batch_minor]
        for node in G.nodes(node_classes=ConvFusionParameters):
            if node.fusion_type != "linear_active":
                continue
            linear = node.contained_nodes()[0]
            if linear.batch_size <= 1 or not linear.batch_minor:
                continue
            batch_minor_nodes.append((node, linear))

        has_modified_graph = False
        for pnode, node in batch_minor_nodes:
            LOG.info(
                "Expand transpose out on batch minor linear node %s", node.name)
            has_modified_graph = True
            node.batch_minor = False
            out_edges = G.out_edges(pnode.name)
            for edge in out_edges:
                G.remove_edge(edge)
            tparams = TransposeParameters(G.unique_name(f'{node.name}_minor'), transpose=(
                1, 0), block_search_up=True, block_search_down=True)
            G.add_edge(NNEdge(from_node=pnode, to_node=tparams))
            for edge in out_edges:
                G.add_edge(NNEdge(from_node=tparams,
                                  to_node=edge.to_node, to_idx=edge.to_idx))
            if G.quantization:
                G.quantization.copy_qrec(pnode, 'out', 0, tparams)

        if set_identity:
            self.set_identity(G)
        return has_modified_graph

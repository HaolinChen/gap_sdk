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

from graph.dim import Dim
import logging
from utils.node_id import NodeId

from graph.matches.matcher import Matcher
from graph.types import NNEdge, ReshapeParameters
from utils.graph import GraphView

LOG = logging.getLogger("nntool." + __name__)


class RemoveReshapes(Matcher):
    NAME = "remove_reshapes"
    DESCRIPTION = """Remove unnecessary reshapes"""

    @staticmethod
    def validate_reshape(G, reshape):
        if reshape.transpose_out:
            return False
        out_shape = None
        candidates = []
        for edge in G.out_edges(reshape.name):
            if not isinstance(edge.to_node, ReshapeParameters):
                return False
            candidate = edge.to_node
            if candidate.transpose_out or candidate.transpose_in:
                return False
            if out_shape is not None:
                if out_shape != tuple(candidate.shape.shape):
                    return False
            else:
                out_shape = tuple(candidate.shape.shape)
            candidates.append(candidate)
        return (reshape, candidates, out_shape)

    def match(self, G: GraphView, set_identity: bool = True, **kwargs):
        modified_graph = False
        while True:
            res = None
            for reshape in G.nodes(node_classes=(ReshapeParameters,)):
                res = self.validate_reshape(G, reshape)
                if res:
                    LOG.info('unnecessary reshape found after %s', reshape.name)
                    break
            else:
                break
            modified_graph = True
            (reshape, candidates, out_shape) = res
            for candidate in candidates:
                LOG.info(
                    'removing unnecessary reshape or transpose %s', candidate.name)
                edges = G.out_edges(candidate.name)
                G.remove(candidate)
                nid = NodeId(candidate)
                if G.quantization and nid in G.quantization:
                    del G.quantization[nid]
                for edge in edges:
                    G.add_edge(NNEdge(from_node=reshape, to_node=edge.to_node, to_idx=edge.to_idx))
            reshape.shape = Dim.unnamed(out_shape)

        if set_identity:
            self.set_identity(G)

        return modified_graph

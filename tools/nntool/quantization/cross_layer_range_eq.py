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

# Implements weight equalization as per https://arxiv.org/abs/1906.04721

import logging
import math
from copy import copy

import numpy as np

from graph.types import (ActivationParameters, FilterParameters,
                         ConvFusionParameters)
from stats.ranges import Ranges
from stats.scales import Scales

LOG = logging.getLogger('nntool.'+__name__)


def process_node(G, node, last_neuron, group, groups, neurons, pnode=None):
    if not node.can_equalize:
        group = add_group(group, groups, neurons)
        return True, None, group

    if isinstance(node, FilterParameters):
        last_neuron = add_neuron(G, node.name, node, last_neuron, neurons, group, pnode=pnode)
        return True, last_neuron, group

    if isinstance(node, ActivationParameters) and\
            last_neuron is not None and node.activation == 'relu':
        assert 'activation' not in last_neuron, "weird 2 activations after conv"
        last_neuron['activation'] = node
        return True, last_neuron, group
    return False, last_neuron, group


def discover_groups(G):
    groups = []
    group = []
    neurons = []
    last_neuron = None
    filters = sorted([node for node in G.nodes() if isinstance(
        node, (FilterParameters, ConvFusionParameters))], key=lambda x: x.step_idx)
    while filters:
        node = filters.pop(0)
        while True:
            next_nodes = G.successors(node.name)
            next_node = next_nodes[0][0] if len(G.successors(node.name)) == 1 and len(
                G.successors(node.name)[0]) == 1 else None
            if not next_node:
                last_neuron = None
                group = add_group(group, groups, neurons)
                break

            should_continue, last_neuron, group = process_node(G, node, last_neuron, group,
                                                               groups, neurons)
            if should_continue:
                if next_node in filters:
                    filters.remove(next_node)
                node = next_node
                continue

            if isinstance(node, ConvFusionParameters):
                for fnode in node.contained_nodes():
                    _, last_neuron, group = process_node(G, fnode, last_neuron, group,
                                                         groups, neurons, pnode=node)
            break

    if group:
        add_group(group, groups, neurons)

    return groups, neurons


def add_group(group, groups, neurons):
    if group:
        LOG.info("Adding group with %d neuron pairs", len(group))
        groups.append(group)
        neurons.append(group[-1][1])
        group = []
    return group


def add_neuron(G, node_name, node, last_neuron, neurons, group, pnode=None):
    in_edges = G.indexed_in_edges(pnode.name) if pnode else G.indexed_in_edges(node.name)
    new_neuron = {'name': node_name, 'node': node,
                  'weights': in_edges[1].from_node.dqvalue.copy(), 'biases': in_edges[2].from_node.dqvalue.copy(),
                  'pnode': pnode}
    if last_neuron is not None:
        neurons.append(last_neuron)
        LOG.info("Discovered neuron pair %s -> %s", last_neuron['name'], new_neuron['name'])
        group.append((last_neuron, new_neuron))
    last_neuron = new_neuron
    return last_neuron


def calculate_s(range_1, range_2):
    assert len(range_1) == len(range_2)
    # note: the paper is wrong. It should be 1/range2 not 1/range1
    return [(1/range_2[i]) * math.sqrt(range_1[i] * range_2[i]) for i in range(len(range_1))]


class QuantizationError(Exception):
    pass


def calculate_precisions(G, step):
    nn_0 = step[0]
    nn_1 = step[1]
    ranges_0, max_0 = Ranges.range_output(G, nn_0['node'], weights=nn_0['weights'])
    ranges_1, max_1 = Ranges.range_input(G, nn_1['node'], weights=nn_1['weights'])
    prec_0 = ranges_0/max_0
    prec_1 = ranges_1/max_1
    return prec_0, prec_1


def process_group(G, group, threshold):
    total_precision = 0
    cycles = 0
    # Keep going until we converge
    while True:
        precisions = []
        cycles += 1
        if cycles > 50:
            raise QuantizationError("Weight scaling has failed to converge")
        for step in group:
            prec_0, prec_1 = calculate_precisions(G, step)
            precisions.append(np.sum(prec_0 * prec_1))

        new_total_precision = sum(precisions)
        # end when the precision change drops below threshold
        if abs(new_total_precision - total_precision) < threshold:
            LOG.info("group has converged under %f after %d cycles", threshold, cycles)
            break
        total_precision = new_total_precision

        # note: traversing in reverse order. Not sure that it makes any difference.
        for step in reversed(group):
            nn_0 = step[0]
            nn_1 = step[1]
            # get the ranges of the output channels of layer 0 and input channels of layer 2
            ranges_0, _ = Ranges.range_output(G, nn_0['node'], weights=nn_0['weights'])
            ranges_1, _ = Ranges.range_input(G, nn_1['node'], weights=nn_1['weights'])
            scale = calculate_s(ranges_0, ranges_1)
            # now apply the scale to the output and input channels
            nn_0['weights'], nn_0['biases'] =\
                Scales.scale_output(G, nn_0['node'], scale, nn_0['weights'], nn_0['biases'])
            nn_1['weights'] = Scales.scale_input(G, nn_1['node'], scale, nn_1['weights'])


def process_groups(G, groups, threshold=0.01):
    for group in groups:
        LOG.info("processing group")
        process_group(G, group, float(threshold))


def update_parameters(G, neurons):
    for neuron in neurons:
        params = neuron['pnode'] or neuron['node']
        in_edges = G.indexed_in_edges(params.name)
        in_edges[1].from_node.dqvalue = neuron['weights']
        if neuron['biases'] is not None:
            in_edges[2].from_node.dqvalue = neuron['biases']


def weight_equalization(G, threshold=0.01):
    LOG.info("discovering groups")
    groups, neurons = discover_groups(G)
    if groups and neurons:
        LOG.info("found %d groups and %d neurons", len(groups), len(neurons))
        process_groups(G, groups, threshold)
        update_parameters(G, neurons)
        G.graph_identity.set_equalized(threshold)
    else:
        LOG.warning("no groups to equalize found")


def adjust_biases(G, stats):
    for nid, stat in stats.items():
        node = nid.get_node(G)
        if isinstance(node, FilterParameters):
            chan_err = np.array(stat['chan_err'], dtype=np.float32)
            in_edges = G.indexed_in_edges(node.name)
            in_edges[2].from_node.value = in_edges[2].from_node.dqvalue - chan_err
            in_edges[2].from_node.qtype = None

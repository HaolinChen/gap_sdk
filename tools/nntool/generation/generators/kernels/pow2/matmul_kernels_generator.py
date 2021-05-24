
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

from generation.at_generators.cnn_convolution_pool_relu import \
    gen_activation_op
from generation.at_types.gen_ctrl import GenCtrl
from generation.generator_decorators import QREC_POW2, generation_function
from graph.types import MatMulOpParameters

from ..autotiler_kernel import NewAutoTilerKernel

LOG = logging.getLogger("nntool." + __name__)


@generation_function("kernels", (MatMulOpParameters,), qrec_types=(QREC_POW2, ))
def matmul_relu_kernels_generator(gen, node, qrec, in_eparams, out_eparams, cname):
    del in_eparams, out_eparams

    if len(node.in_dims[0].shape) == 2 and len(node.in_dims[1].shape) == 2:
        gen.kernels.append(MatMulReluKernel(
            cname, node, qrec, None, None, gen_ctrl=node.get_gen_ctrl()))
        return True
    return False


class MatMulReluKernel(NewAutoTilerKernel):
    CALL_TEMPLATE = '''
// generator for {node_name}
CNN_MatMul("{cname}", {gen_ctrl}, {at_bits(in1_q)}, {at_bits(in2_q)},
           {at_bits(bias_q)}, {at_bits(out_q)}, {in1_q.q}, {in2_q.q}, {bias_q.q}, {out_q.q},
           1, 1, 1, 1,
           {in1_shape[0]}, {in1_shape[1]}, {in2_shape[0]}, {in2_shape[1]}, 1, 1, 1, 1,
           {relu_lower}, {relu_upper}, KOP_MATMUL, {act_op});
'''

    def __init__(self, cname, params, matmul_q, act_params, act_q, gen_ctrl=None):
        if gen_ctrl is None:
            gen_ctrl = GenCtrl(None, cname=cname)
        else:
            gen_ctrl.cname = cname

        if len(params.in_dims[0]) != 2 or len(params.in_dims[1]) != 2:
            raise ValueError(f'Matmul {params.name} has inputs of rank {len(params.in_dims[0])} and {len(params.in_dims[1])}'
                             f'which are not supported by the matmul kernel')
        in1_shape = params.in_dims[0].shape
        in2_shape = params.in_dims[1].shape
        out_shape = params.out_dims[0].shape

        in1_q = matmul_q.in_qs[0]
        in2_q = matmul_q.in_qs[1]
        out_q = matmul_q.out_qs[0]
        bias_q = matmul_q.in_qs[2]

        if act_params is not None:
            act_op = gen_activation_op(act_params.activation)
            out_q = act_q.out_qs[0]
            relu_lower = 0
            if act_params.activation == "relu6" and out_q.q != 0:
                relu_upper = 6 << out_q.q
            else:
                relu_upper = 0
        else:
            relu_upper = relu_lower = 0
            act_op = "KOP_NONE"

        # attributes used to test equality - i.e. this kernel can be reused
        attrs = {
            'in1_q': in1_q,
            'in2_q': in2_q,
            'bias_q': bias_q,
            'out_q': out_q,
            'in1_shape': in1_shape,
            'in2_shape': in2_shape,
            'out_shape': out_shape,
            'relu_lower': relu_lower,
            'relu_upper': relu_upper,
            'act_op': act_op
        }

        # other attributes
        extra_attrs = {
            'cname': cname,
            'node_name': params.name
        }

        super().__init__(attrs, extra_attrs, gen_ctrl=gen_ctrl)

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

import numpy as np
from graph.types import PoolingParameters
from graph.types.pooling import MaxPoolParameters
from quantization.new_qrec import QRec
from quantization.unified_quantization_handler import (in_qs_constraint,
                                                       options,
                                                       out_qs_constraint,
                                                       params_type,
                                                       priority)

from ..mult_quantization_handler import MultQuantizionHandler

LOG = logging.getLogger('nntool.' + __name__)

AT_SW_KER_IN_ORDER = [['c', 'h', 'w']]
AT_SW_KER_OUT_ORDER = [['c', 'h', 'w']]
AT_NE16_KER_IN_ORDER = [['h', 'w', 'c']]
AT_NE16_KER_OUT_ORDER = [['h', 'w', 'c']]

@options(
    {
        'name': 'allow_asymmetric',
        'type': bool,
        'help': 'EXPERIMENTAL - allow soft kernels to use asymmetric quantization where possible',
        'default': False
    }
)
@params_type(PoolingParameters)
@in_qs_constraint({'dtype': set([np.int8])})
@out_qs_constraint({'dtype': set([np.int8])})
class PoolingMult(MultQuantizionHandler):

    @classmethod
    def _quantize(cls, params, in_qs, stats, **kwargs):
        # copy in_qs because we may modify it
        in_qs = in_qs.copy()
        opts = kwargs['opts']

        force_out_qs, out_dtype = cls.get_mult_opts(**kwargs)
        force_out_q = force_out_qs and force_out_qs[0]
        G = kwargs['G']
        in_q = in_qs[0]

        asymmetric_enabled = kwargs.get('allow_asymmetric')
        if in_q.is_asymmetric and (asymmetric_enabled and not params.padding.has_padding):
            in_qs = cls.force_symmetric(in_qs)
            if in_qs is None:
                return None
            in_q = in_qs[0]

        cls.check_valid_ranges(params, stats, idx=0, dirs='in')
        min_val = stats['range_in'][0]['min']
        max_val = stats['range_in'][0]['max']

        if force_out_q:
            o_q = force_out_q
            if o_q.dtype != in_q.dtype or o_q.zero_point != in_q.zero_point:
                if in_q.forced or not asymmetric_enabled and o_q.zero_point != 0:
                    return None
                in_q = deepcopy(o_q)
            LOG.warning('node %s output forced to range %s/%s - actual range %s/%s %s',
                        params.name, o_q.min, o_q.max, min_val, max_val,
                        "asymmetric" if o_q.is_asymmetric else "symmetric")
        else:
            o_q = deepcopy(in_q)
        cls.check_order(params, AT_SW_KER_IN_ORDER, AT_SW_KER_OUT_ORDER)
        return QRec.scaled(in_qs=[in_q],
                           out_qs=[o_q])

@params_type(MaxPoolParameters)
@in_qs_constraint({'dtype': set([np.uint8])})
@out_qs_constraint({'dtype': set([np.uint8])})
@priority(2)
class NE16PoolingMult(MultQuantizionHandler):

    @classmethod
    def _quantize(cls, params, in_qs, stats, **kwargs):
        # copy in_qs because we may modify it
        in_qs = in_qs.copy()
        opts = kwargs['opts']

        force_out_qs, out_dtype = cls.get_mult_opts(**kwargs)
        force_out_q = force_out_qs and force_out_qs[0]
        G = kwargs['G']
        in_q = in_qs[0]

        cls.check_valid_ranges(params, stats, idx=0, dirs='in')
        min_val = stats['range_in'][0]['min']
        max_val = stats['range_in'][0]['max']

        if force_out_q:
            o_q = force_out_q
            if in_q.forced and in_q.zero_point != o_q.zero_point:
                return None
            in_q = deepcopy(o_q)

            LOG.warning('node %s output forced to range %s/%s - actual range %s/%s %s',
                        params.name, o_q.min, o_q.max, min_val, max_val,
                        "asymmetric" if o_q.is_asymmetric else "symmetric")
        else:
            o_q = deepcopy(in_q)
        cls.check_order(params, AT_NE16_KER_IN_ORDER, AT_NE16_KER_OUT_ORDER)
        return QRec.scaled(in_qs=[in_q],
                           out_qs=[o_q],
                           ne16=True)

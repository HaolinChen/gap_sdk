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
from abc import ABC, abstractmethod
from copy import deepcopy

from quantization.new_qrec import QRec

LOG = logging.getLogger('nntool.' + __name__)

class ConcatMixin(ABC):
    @classmethod
    def _handle(cls, params, in_qs, _, **kwargs):
        force_out_qs = kwargs['force_out_qs']
        force_out_q = force_out_qs[0] if force_out_qs else None
        forced_in_qs = [in_q for in_q in in_qs if in_q.forced]
        # two inputs cannot be forced to different values
        if forced_in_qs and any(in_q != forced_in_qs[0] for in_q in forced_in_qs[1::]):
            LOG.info(
                'two input qtypes of concat %s are forced to different qtypes', params.name)
            return None
        # input cannot be forced to different value than output
        if force_out_q and not all(in_q == force_out_q for in_q in forced_in_qs):
            LOG.info(
                'output and input of concat %s are forced to different qtypes', params.name)
            return None

        backwards = kwargs.get('backwards')
        # if we are going backwards or are forced
        if backwards or force_out_q:
            # if output must be forced
            assert force_out_q, f'going backwards at {params.name} but output is not forced'
            in_qs = [deepcopy(force_out_q) for _ in in_qs]
            return QRec(ktype=cls.KTYPE, in_qs=in_qs, out_qs=[deepcopy(force_out_q)])

        # if all the inputs are the same qtype then we output that qtype
        if all(in_qs[0] == in_q for in_q in in_qs[1::]):
            return QRec(ktype=cls.KTYPE, in_qs=in_qs, out_qs=[deepcopy(in_qs[0])])

        # our output cannot be forced at this point
        # if an input is forced then all forced inputs must be the same here
        if forced_in_qs:
            in_qs = [deepcopy(forced_in_qs[0]) for _ in in_qs]
            return QRec(ktype=cls.KTYPE, in_qs=in_qs, out_qs=[deepcopy(forced_in_qs[0])])

        # if all the inputs are not the same then force all of them to the maximum input size with a Q that
        # fits the most int bits
        common_q = cls._get_common_q(in_qs)
        in_qs = [deepcopy(common_q) for _ in in_qs]

        return QRec(ktype=cls.KTYPE, in_qs=in_qs, out_qs=[deepcopy(common_q)])

    KTYPE=None

    @classmethod
    @abstractmethod
    def _get_common_q(cls, in_qs):
        pass

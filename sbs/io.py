#!/usr/bin/env python2
# encoding: utf-8

import numpy as np
import atexit
import os

from .logcfg import log
from . import utils


class UpdateParamsCD(object):
    """
        Utility class to synchronize update factors with custom NEST synapses
        in order to enable learning without I/O limitation even in the
        multihreaded case.
    """
    directory = "/dev/shm"

    def __init__(self, num_nodes=None, num_snapshots=None, filepath=None):
        """
            Have to specify either num_nodes/num_snapshots or filepath.
        """
        if filepath is None and (num_nodes is None or num_snapshots is None):
            raise IOError("Have to specify either num_nodes/num_snapshots or "
                          "filepath.")

        self._create(num_nodes=num_nodes, num_snapshots=num_snapshots,
                     filepath=filepath)

    def _create(self, num_nodes=None, num_snapshots=None, filepath=None):
        if filepath is None:
            create_file = True
            filepath = self.directory + "/sbs-" + utils.get_random_string(28)
        else:
            create_file = False

        self._filepath = filepath

        if create_file:
            def remove_file():
                try:
                    os.remove(filepath)
                except OSError:
                    log.warn("Shared memory file already removed.")

            atexit.register(remove_file)

        if create_file:
            mode = "w+"
            shape = (self.get_size_total(
                num_nodes=num_nodes, num_snapshots=num_snapshots),)
        else:
            mode = "r+"
            shape = None

        self._mm_dbl = np.memmap(self.get_filepath(), mode=mode,
                                 shape=shape, order="C", dtype=np.float64)
        self._mm_int = np.memmap(self.get_filepath(), mode="r+",
                                 shape=None, order="C", dtype=np.int64)

        if create_file:
            self._write_preamble(
                    num_nodes=num_nodes, num_snapshots=num_snapshots)
        self._calc_sizes()
        log.info("Shared memory file in: {}".format(self.get_filepath()))

    def _write_preamble(self, num_nodes, num_snapshots):
        self._set_num_nodes(num_nodes)
        self._set_num_snapshots(num_snapshots)
        self._set_update_num_first(1)
        self._set_update_num_latest(0)
        self._mm_int.flush()

    def prepare_next_update(self):
        """
            Advances the snapshot counter.
        """
        self._set_update_num_latest(self.get_update_num_latest()+1)

    def get_snapshots_used(self):
        return self.get_update_num_latest() - self.get_update_num_first() + 1

    def flush_files(self):
        self._mm_dbl.flush()
        self._mm_int.flush()

    def clear_updates(self):
        self._set_update_num_first(self.get_update_num_latest()+1)

    def set_eta(self, eta):
        """
            Sets learning rate for current snapshot.
        """
        self._mm_dbl[self._get_offset_snapshot() +
                     self._get_offset_eta_in_snapshot()] = eta

    def set_update_data(self, update_data):
        offset = self._get_offset_snapshot() + self._get_size_snapshot_header()
        size = self._get_size_snapshot() - self._get_size_snapshot_header()
        self._mm_dbl[offset:offset+size] = update_data.reshape(-1)

    def set_weight_conversion_factors(self, factors):
        """
            Factors should be ndarray of size num_nodes * 2.

            First factor for each node is exc, then inh.
        """
        factors = np.array(factors).reshape(-1)
        self._mm_dbl[self._get_offset_misc_nodedata():] = factors

    def get_size_total(self, num_nodes=None, num_snapshots=None):
        """
            Get size for file from either the supplied arguments or from the
            file actually open.
        """
        if num_nodes is None:
            num_nodes = self.get_num_nodes()

        if num_snapshots is None:
            num_snapshots = self.get_num_snapshots()

        return (self._get_size_header() +
                num_snapshots * self._get_size_snapshot(num_nodes=num_nodes) +
                num_nodes * self._get_size_misc_nodedata())

    def get_filepath(self):
        return self._filepath

    def _calc_sizes(self):
        self._size_snapshot = self._get_size_snapshot_header()\
                + self.get_num_nodes() * self._get_size_nodedata()

        self._offset_misc_nodedata = (self._get_size_header() +
                                      self.get_num_snapshots() *
                                      self._get_size_snapshot())

    def get_num_nodes(self):
        return self._mm_int[0]

    def _set_num_nodes(self, nn):
        self._mm_int[0] = nn

    def get_update_num_first(self):
        return self._mm_int[1]

    def _set_update_num_first(self, num):
        self._mm_int[1] = num

    def get_update_num_latest(self):
        return self._mm_int[2]

    def _set_update_num_latest(self, num):
        self._mm_int[2] = num
        self._mm_int.flush()

    def get_num_snapshots(self):
        return self._mm_int[3]

    def _set_num_snapshots(self, num_snapshots):
        self._mm_int[3] = num_snapshots

    def _get_offset_misc_nodedata(self):
        return self._offset_misc_nodedata

    def _get_offset_eta_in_snapshot(self):
        return 0

    def _get_offset_snapshot(self, update_num=None):
        """
            If update_num is None the latest update spot will be used.
        """
        if update_num is None:
            relative_update_num = self.get_update_num_latest()\
                    - self.get_update_num_first()
        else:
            relative_update_num = update_num - self.get_update_num_first()

        return (self._get_size_header() +
                relative_update_num * self._get_size_snapshot())

    def _get_size_header(self):
        return 4

    def _get_size_misc_nodedata(self):
        return 2

    def _get_size_nodedata(self):
        return 2

    def _get_size_snapshot(self, num_nodes=None):
        if num_nodes is None:
            return self._size_snapshot
        else:
            return self._get_size_snapshot_header()\
                    + num_nodes * self._get_size_nodedata()

    def _get_size_snapshot_header(self):
        return 1

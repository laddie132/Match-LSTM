# -*- coding: utf-8 -*-

import h5py
import numpy as np

_compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)


def transform_hdf5():
    f = h5py.File('../../squad_data/squad_glove.h5-600', 'a')

    f_data = f['data']
    for k1 in ['train', 'dev']:

        cur_data = f_data[k1]
        for k2 in ['context', 'question']:
            tokens = np.array(cur_data[k2])

            del f_data[k1][k2]
            cur_data.create_group(k2)
            data = cur_data[k2].create_dataset('token', tokens.shape, dtype=tokens.dtype, **_compress_option)
            data[...] = tokens

    f.close()


if __name__ == '__main__':
    transform_hdf5()
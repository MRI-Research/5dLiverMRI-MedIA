#!/usr/bin/env python3

"""
    recon_respiratory_signal.py
    - Last modified: Jan 15, 2025

    Author:
    Mungsoo Kang <kangms0511@gmail.com>
    Youngwook Kee <dr.youngwook.kee@gmail.com>
"""

import argparse
import logging
import os
import cfl
import numpy as np
from scipy import signal
import h5py


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Respiratory motion estimation.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('img_file', type=str)

    args = parser.parse_args()

    # Verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logging.info('Reading data.')
    with h5py.File('Gd_Phantom_Cones_With_Motion_1.h5', 'r') as hf:
        ksp_1   = hf["ksp"][:]
        tr    = hf["tr"][...]
    tr = float(tr)
    with h5py.File('Gd_Phantom_Cones_With_Motion_2.h5', 'r') as hf:
        ksp_2   = hf["ksp"][:]
        
    ksp = np.concatenate((ksp_1,ksp_2), axis=1)
    
    del ksp_1
    del ksp_2
    
    num_echoes, num_coils, num_tr, num_ro  = ksp.shape[-4:]  # Extract dims

    ksp_dc = np.abs(ksp[:, :, :, 0])

    logging.info('Filtering.')
    sos_lpf = signal.butter(10, 2, 'low', fs=1/tr, output='sos')
    sos_hpf = signal.butter(5, 0.1, 'high', fs=1/tr, output='sos')
    
    dc_filt = np.empty(ksp_dc.shape, dtype=np.float32)
    for e in range(num_echoes):
        for c in range(num_coils):
            dc_filt[e, c, :] = signal.sosfiltfilt(sos_lpf, ksp_dc[e, c, :] - np.mean(ksp_dc[e, c, :]))
            dc_filt[e, c, :] = signal.sosfiltfilt(sos_hpf, dc_filt[e, c, :]) 
            
    logging.info('Applying PCA along the coil dim')
    resp_e = np.empty([num_echoes, num_tr], dtype=np.float32)
    for e in range(num_echoes):
        u, s, vh = np.linalg.svd(dc_filt[e], full_matrices=False)
        resp_e[e] = s[0] * vh[0]
        
    if num_echoes > 1:
        logging.info('Applying PCA along the echo dim')
        u, s, vh = np.linalg.svd(resp_e, full_matrices=False)
        resp = np.empty([num_tr, 1], dtype=np.float32)
        resp[:,0] = s[0] * vh[0]
    
    logging.info('Normalization.')
    resp = (resp - resp.min(axis=0)) / (resp.max(axis=0) - resp.min(axis=0))

    logging.info('Writing.')
    img_file = os.path.join(args.input_dir, args.img_file)
    cfl.write_cfl(img_file, resp)   
    
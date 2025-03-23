#!/usr/bin/env python3

"""
    recon_gridding_motion_averaged.py
    - Last modified: Jan 15, 2025

    Author:
    Mungsoo Kang <kangms0511@gmail.com>
    Youngwook Kee <dr.youngwook.kee@gmail.com>
"""

import argparse
import cfl
import os
import numpy as np
import cupy as cp
import sigpy as sp
import logging
import h5py


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gridding reconstruction.')
    parser.add_argument('--rss', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('img_file', type=str)
    parser.add_argument('--device', type=int, default=0, help='GPU device.')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.info('Reading data.')
    with h5py.File(args.input_dir + '/Gd_Phantom_Cones_With_Motion_1.h5', 'r') as hf:
        ksp_1   = hf["ksp"][:]
        coord = hf["coord"][:]
        dcf  = hf["dcf"][:]
        img_shape = hf["imageDim"][:]
        voxelSize = hf["voxelSize"][:]
        te    = hf["te"][:]
        tr    = hf["tr"][...]
        cf    = hf["cf"][...]
    tr = float(tr)
    img_shape = img_shape.tolist()
    
    with h5py.File(args.input_dir + '/Gd_Phantom_Cones_With_Motion_2.h5', 'r') as hf:
        ksp_2   = hf["ksp"][:]
        
    ksp = np.concatenate((ksp_1,ksp_2), axis=1)
    
    del ksp_1
    del ksp_2
    
    num_echoes, num_coils, num_tr, num_ro  = ksp.shape[-4:]  # Extract dims

    ndim = coord.shape[-1]
    
    logging.info('Scaling coordinate.')
    coord[...,0] *= img_shape[0]/2/np.max(coord[...,0])
    coord[...,1] *= img_shape[1]/2/np.max(coord[...,1])
    coord[...,2] *= img_shape[2]/2/np.max(coord[...,2])


    num_gpus = cp.cuda.runtime.getDeviceCount()
    
    if args.rss:
        img = np.zeros([1, 1, 1, num_echoes, 1, 1] + img_shape,
                       dtype=np.complex64)
    else:
        img = np.zeros([1, 1, 1, num_echoes, 1, num_coils] + img_shape,
                       dtype=np.complex64)
        
    device = cp.cuda.Device(args.device) 
    
    with device:
        for e in range(num_echoes):
            coord = cp.array(coord)
            if args.rss:
                img_t = 0

            for c in range(num_coils):
                logging.info('Reconstructing: coil {c}/{C}, echo {e}/{E}'.format(c=c + 1, e=e + 1,  C=num_coils, E=num_echoes))
                ksp_t = cp.array(ksp[e, c, :, :])
                ksp_t *= cp.array(dcf)
                
                img_tc = sp.nufft_adjoint(ksp_t, coord, img_shape)

                if args.rss:
                    img_t += cp.abs(img_tc)**2
                else:
                    sp.copyto(img[0, 0, 0, e, 0, c], img_tc)

            if args.rss:
                    sp.copyto(img[0, 0, 0, e, 0, 0], img_t**0.5)

    logging.info('Writing.')
    img_file = os.path.join(args.input_dir, args.img_file)
    cfl.write_cfl(img_file, img)

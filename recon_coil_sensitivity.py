#!/usr/bin/env python3

"""
    recon_coil_sensitivity.py
    - Python implementation of JSENSE for cones
    - Heavily based on Frank Ong's code (extreme MRI)
    - Last modified: Jan 15, 2025

    Author:
    Mungsoo Kang <kangms0511@gmail.com>
    Youngwook Kee <dr.youngwook.kee@gmail.com>
"""

import argparse
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import logging
import cfl
import os
import h5py


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='JSENSE reconstruction')
    parser.add_argument('--mps_ker_width', type=int, default=8)
    parser.add_argument('--ksp_calib_width', type=int, default=36)
    parser.add_argument('--lamda', type=float, default=0)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--max_inner_iter', type=int, default=10)
    parser.add_argument('--show_pbar', action='store_true', help='Show progress bar.')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Toggle multi-gpu mode. Overrides device option.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('img_file', type=str)

    args = parser.parse_args()

    # Verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Choose device
    comm = sp.Communicator()
    if args.multi_gpu:
        device = sp.Device(comm.rank)
    else:
        device = sp.Device(args.device)
    
    if comm.rank == 0:
        logging.info('Number of MPI nodes: {}'.format(comm.size))
    
    # Reading data
    if comm.rank == 0:
        logging.info('Reading data.')

    with h5py.File(args.input_dir + '/Gd_Phantom_Cones_With_Motion_1.h5', 'r') as hf:
        ksp_1   = hf["ksp"][:]
        coord = hf["coord"][:]
        dcf  = hf["dcf"][:]
        img_shape = hf["imageDim"][:]

    img_shape = img_shape.tolist()
    
    with h5py.File(args.input_dir + '/Gd_Phantom_Cones_With_Motion_2.h5', 'r') as hf:
        ksp_2   = hf["ksp"][:]
        
    ksp = np.concatenate((ksp_1,ksp_2), axis=1)
    
    del ksp_1
    del ksp_2
    
    num_echoes, num_coils, _, _,  = ksp.shape[-4:]  # Extract dims

    if comm.rank == 0:
        logging.info('Scaling coordinates.')
    
    coord[...,0] *= img_shape[0]/2/np.max(coord[...,0])
    coord[...,1] *= img_shape[1]/2/np.max(coord[...,1])
    coord[...,2] *= img_shape[2]/2/np.max(coord[...,2])

    if comm.rank == 0:
        logging.info('JSENSE Recon.')
    
    if num_echoes > 1:
        ksp = ksp[0]  # first echo only

    # Split between MPI nodes
    if comm.rank == 0:
        ksp = np.array_split(ksp, comm.size)[comm.rank]

    # JSENSE
    mps = mr.app.JsenseRecon(ksp, coord=coord, weights=dcf,
                             device=device, comm=comm, show_pbar=args.show_pbar).run()

    # Gather data and stack
    mps_size = mps.shape[1:]
    mps = comm.gatherv(mps, root=0)

    # Writing.
    if comm.rank == 0:
        logging.info('Writing as CFL files.')
        mps = mps.reshape((-1, ) + mps_size)
        img = np.empty([1, 1, 1, 1, 1, num_coils] + img_shape, dtype=np.complex64)

        # TODO: num_phases
        for c in range(num_coils):
            img[0, 0, 0, 0, 0, c, :, :, :] = sp.to_device(sp.ifft(sp.resize(sp.fft(mps[c]), img_shape)))
        
        img_file = os.path.join(args.input_dir, args.img_file)
        cfl.write_cfl(img_file, img)
    

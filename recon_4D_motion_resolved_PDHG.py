#!/usr/bin/env python3

"""
    recon_4D_motion_resolved_PDHG.py
    - Motion-resolved reconstruction using PDHG
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
import sigpy as sp
from tqdm.auto import tqdm
import time
import h5py


def read_local_kspace(input_dir, comm):
    file1 = os.path.join(input_dir, 'Gd_Phantom_Cones_With_Motion_1.h5')
    file2 = os.path.join(input_dir, 'Gd_Phantom_Cones_With_Motion_2.h5')

    with h5py.File(file1, 'r') as hf:
        c1 = hf['ksp'].shape[1]
        coord = hf['coord'][:]
        dcf = hf['dcf'][:]
        img_shape = hf['imageDim'][:]
        voxelSize = hf['voxelSize'][:]
        te = hf['te'][:]
        tr = hf['tr'][...]
        cf = hf['cf'][...]

    with h5py.File(file2, 'r') as hf:
        c2 = hf['ksp'].shape[1]

    local_coils = np.array_split(np.arange(c1 + c2), comm.size)[comm.rank]
    idx1 = local_coils[local_coils < c1]
    idx2 = local_coils[local_coils >= c1] - c1

    with h5py.File(file1, 'r') as hf:
        ksp_1 = read_coils(hf['ksp'], idx1)

    with h5py.File(file2, 'r') as hf:
        ksp_2 = read_coils(hf['ksp'], idx2)

    ksp = np.concatenate((ksp_1, ksp_2), axis=1)
    return ksp, coord, dcf, img_shape.tolist(), voxelSize, te, tr, cf, c1 + c2


def read_coils(dataset, coil_idx):
    if len(coil_idx) == 0:
        return np.empty((dataset.shape[0], 0, dataset.shape[2], dataset.shape[3]), dtype=dataset.dtype)

    if np.all(np.diff(coil_idx) == 1):
        return dataset[:, slice(int(coil_idx[0]), int(coil_idx[-1]) + 1), :, :]

    return dataset[:, coil_idx, :, :]


def max_abs_in_coil_blocks(ksp, block_size=4):
    scale = 0.0
    for start in range(0, ksp.shape[1], block_size):
        stop = min(start + block_size, ksp.shape[1])
        scale = max(scale, float(np.max(np.abs(ksp[:, start:stop]))))
    return scale


def global_max(comm, value):
    if getattr(comm, 'size', 1) > 1 and hasattr(comm, 'mpi_comm'):
        from mpi4py import MPI
        return comm.mpi_comm.allreduce(value, op=MPI.MAX)
    return value


class MotionResolvedRecon(object):
    def __init__(self, ksp, coord, dcf, mps, resp, dual_q, B,
                 lambda1=1e-6, sigma=0.1, tau=0.1,max_iter=10, 
                 tol=0.01,device=sp.cpu_device, margin=2,
                 comm=None, show_pbar=True, **kwargs):
        self.B = B
        self.C = len(mps)
        self.E = ksp.shape[0]
        self.mps = mps
        self.device = sp.Device(device)
        self.xp = device.xp
        self.sigma = sigma # PDHG
        self.tau = tau # PDHG
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.tol = tol
        self.comm = comm
        self.time_file = kwargs.get('time_file')
        self.show_pbar = show_pbar

        if comm is not None:
            self.show_pbar = show_pbar and comm.rank == 0

        self.img_shape = list(mps.shape[1:])

        bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, B + 1))
        self.bksp_temp = []
        self.bksp = []
        self.bcoord = []
        self.bdcf = []
        self.dual_q_ = []
        
        for b in range(B):
            if b < self.B-1:
                idx = (resp >= bins[b]) & (resp < bins[b + 1])
                self.bksp.append(sp.to_device(ksp[:, :, idx], self.device))
                self.bcoord.append(
                    sp.to_device(coord[idx], self.device))
                self.bdcf.append(
                    sp.to_device(dcf[idx], self.device))
                self.dual_q_.append(
                    sp.to_device(dual_q[:, :, idx], self.device))
            if b == self.B -1:
                idx = (resp >= bins[b]) & (resp <= bins[b + 1])
                self.bksp.append(sp.to_device(ksp[:, :, idx], self.device))
                self.bcoord.append(
                    sp.to_device(coord[idx], self.device))
                self.bdcf.append(
                    sp.to_device(dcf[idx], self.device))
                self.dual_q_.append(
                    sp.to_device(dual_q[:, :, idx], self.device))

    # Primal-Dual Algorithm
    def pdinit(self, mrimg):

        dual_p_m = self.xp.zeros_like(mrimg)
        dual_q = self.dual_q_

        return dual_p_m, dual_q

    def pdhg(self, primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_q, it):
        
        # p: dual variable for total variation
        # q: dual variable for data term
        # u: primal variable

        primal_u_old[...] = primal_u_tmp

        ### @Dual variable p ###
        ###
        # 1) update p
        # p^k+1 = prox_P (p^k + sigma * \partial_t u^k)
        for b in range(self.B):
            if b < self.B - 1:
                dual_p_m[b] += self.sigma * (primal_u[b + 1] - primal_u[b])
        
        # proximal operator
        for b in range(self.B):
            absp = self.xp.abs(dual_p_m[b])
            dual_p_m[b] /= self.xp.maximum(1, absp / self.lambda1)

        ### @Dual Variable q ###
        ###
        # 2) update q
        # q^k+1 = prox_Q (q^k + sigma * (FSu^k - y))
        
        for b in range(self.B):
            tmp =  self.xp.zeros_like(self.dual_q_[b])
            for c in range(self.C):
                for e in range(self.E):
                    mps_c = sp.to_device(self.mps[c], self.device)
                    tmp[e, c] = sp.nufft(primal_u[b, e] * mps_c, self.bcoord[b]) - self.bksp[b][e][c]

            # proximal operator
            tmp *= self.sigma
            dual_q[b] += tmp
            dual_q[b] /= 1 + self.sigma

        ### @PRIMAL VARIABLE ###
        ##
        # update u
        # u^k+1 = u^k - tau * ( (FS)^H q^k+1 + divp )
        
        # 2) Compute divergene w/backward gradient
        divp = self.xp.zeros_like(primal_u)
        for b in range(self.B):
            if b == 0:
                divp[b] = dual_p_m[b]
            if b > 0 and b < self.B - 1:
                divp[b] = dual_p_m[b] - dual_p_m[b - 1]
            if b == self.B - 1:
                divp[b] = -dual_p_m[b - 1]

        tmp = self.xp.zeros_like(primal_u)
        for b in range(self.B):
            for c in range(self.C):
                for e in range(self.E):
                    mps_c = sp.to_device(self.mps[c], self.device)
                    tmp[b, e] += sp.nufft_adjoint(self.bdcf[b] * dual_q[b][e][c], 
                                        self.bcoord[b], oshape=primal_u.shape[2:]) * self.xp.conj(mps_c)

        if self.comm is not None:
            self.comm.allreduce(tmp)

        tmp -= divp
        tmp *= -self.tau
        primal_u_tmp += tmp

        ### @AUXILIARY UPDATE ###
        primal_u[...] = primal_u_tmp
        primal_u *= 2
        primal_u -= primal_u_old

        return primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_q

    def run(self):
        done = False
        while not done:
            with tqdm(total=self.max_iter, desc='MotionResolvedRecon\n',
                      disable=not self.show_pbar) as pbar:
                with self.device:
                    mrimg = self.xp.zeros([self.B] + [self.E] + self.img_shape,
                                         dtype=self.mps.dtype)

                    dual_p_m, dual_q = self.pdinit(mrimg)
                    primal_u_old = self.xp.zeros_like(mrimg)
                    primal_u_tmp = self.xp.zeros_like(mrimg)
                    start_time = time.monotonic()  

                    for it in range(self.max_iter):
                        mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_q = \
                            self.pdhg(mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_q, it)

                        denom = self.xp.linalg.norm(abs(primal_u_old))
                        if denom == 0:
                            denom = self.xp.linalg.norm(abs(primal_u_tmp))
                        if denom == 0:
                            denom = 1
                        _tol = self.xp.linalg.norm(abs(primal_u_tmp - primal_u_old))/denom
                        pbar.set_postfix(tol=_tol)

                        if (_tol < self.tol):
                            break
                        pbar.update()
                    done = True

        end_time = time.monotonic()
        total_time = end_time - start_time
        if self.time_file is not None:
            np.savetxt(self.time_file+'_total_time.txt',np.repeat(total_time,2),fmt='%4.4f')
        return mrimg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Motion Resolved Reconstruction (PDHG).')
    parser.add_argument('--num_bins', type=int, default=6, help='Number of phases.')
    parser.add_argument('--lambda1', type=float, default=1e-6,
                        help='Regularization for motion.')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum iteration.')
    parser.add_argument('--show_pbar', action='store_true', help='Show progress bar.')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Toggle multi-gpu. Overrides device option.')
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

    ksp, coord, dcf, img_shape, voxelSize, te, tr, cf, total_coils = read_local_kspace(args.input_dir, comm)
    num_echoes, local_coils, _, _, = ksp.shape[-4:]  # Extract dims

    if comm.rank == 0:
        logging.info('Scaling coordinates.')
        
    coord[...,0] *= img_shape[0]/2/np.max(coord[...,0])
    coord[...,1] *= img_shape[1]/2/np.max(coord[...,1])
    coord[...,2] *= img_shape[2]/2/np.max(coord[...,2])

    if comm.rank == 0:
        logging.info('Reading sensitivity map.')
    
    mps_file = os.path.join(args.input_dir, 'mps')
    mps      = np.squeeze(cfl.read_cfl(mps_file))
    
    if comm.rank == 0:
        logging.info('Reading resp signal.')
    
    resp_file = os.path.join(args.input_dir, 'resp')
    resp      = np.squeeze(cfl.read_cfl(resp_file)).real
    
    # Coil sensitivity map normalization
    if 1:
        ksp /= global_max(comm, max_abs_in_coil_blocks(ksp))
        ksp *= 1000
        mpsSOS = np.sum(abs(mps)**2, 0)**0.5
        for c in range(mps.shape[0]):
            mps[c] /= mpsSOS

    # Split between MPI nodes: make sure we split COILS of ksp and mps
    dual_q = np.zeros(ksp.shape, dtype=np.complex64)
    mps = np.array_split(mps, comm.size)[comm.rank]

    img = np.empty([args.num_bins, 1, 1, num_echoes, 1, 1] + img_shape, dtype=np.complex64)
    
    if comm.rank == 0:
        logging.info('Running motion-resolved reconstruction (PDHG): #echos={E}, #coils={C}, #bins={B}'.format(
            E=num_echoes, C=total_coils, B=args.num_bins))

    mrimg = MotionResolvedRecon(ksp, coord, dcf, mps, resp, dual_q, args.num_bins,
                            max_iter=args.max_iter, lambda1=args.lambda1, sigma=0.1, tau=0.1, tol=0.01, device=device, margin=2, comm=comm,
                            show_pbar=args.show_pbar,
                            time_file=os.path.join(args.input_dir, args.img_file)).run()
    
    with device:
        img[:, 0, 0, :, 0, 0, :, :, :] = sp.to_device(mrimg)

    # Writing.
    if comm.rank == 0:
        logging.info('Writing as CFL files.')
        
        img_file = os.path.join(args.input_dir, args.img_file)
        cfl.write_cfl(img_file, img)

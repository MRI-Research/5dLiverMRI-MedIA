#!/usr/bin/env python3

"""
    recon_5D_motion_resolved_PDHG_undersampling.py
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
import cupy as cp
import math
import pywt
from tqdm.auto import tqdm
from array import array
import time
import h5py

class MotionResolvedRecon(object):
    def __init__(self, ksp, coord, dcf, mps, resp, dual_q, B, random=True,
                 lambda1=1e-6, lambda2=1e-6, lambda3=1e-6, undersampling=2, sigma=0.1, tau=0.1,
                 max_iter=10, tol=0.01,device=sp.cpu_device, margin=2,
                 comm=None, show_pbar=True, **kwargs):
        self.B = B
        self.C = len(mps)
        self.E = ksp.shape[0]
        self.mps = mps
        self.device = sp.Device(device)
        self.xp = device.xp
        self.sigma = sigma # PDHG
        self.tau = tau # PDHG
        self.random = random
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_iter = max_iter
        self.tol = tol
        self.comm = comm
        self.undersampling = undersampling
        
        if comm is not None:
            self.show_pbar = show_pbar and comm.rank == 0

        self.img_shape = list(mps.shape[1:])

        self.bksp = []    
        self.bcoord = []
        self.bdcf = []
        self.dual_q_ = []
        self.bksp_motion = []
        self.dual_q_motion = []
        self.bcoord_motion = []
        self.bdcf_motion = []
        self.bksp_temp = []
        self.dual_q_temp = []
        self.bcoord_temp = []
        self.bdcf_temp = []
        
        idx_temp = np.zeros((self.B))
        bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, B + 1))

        for b in range(B):
            if b < self.B-1:
                idx = (resp >= bins[b]) & (resp < bins[b + 1])
                self.bksp_motion.append(ksp[:, :, idx])
                self.bcoord_motion.append(coord[idx])
                self.bdcf_motion.append(dcf[idx])
                self.dual_q_motion.append(dual_q[:, :, idx])
                ksp_1 = len(self.bksp_motion)
                ksp_2 = len(self.bksp_motion[0])
                ksp_3 = len(self.bksp_motion[0][0])
                ksp_4 = len(self.bksp_motion[0][0][0])
                ksp_5 = len(self.bksp_motion[0][0][0][0])
                if comm.rank == 0:
                    logging.info('Motion Undersampled k-space: {A},{B},{C},{D},{E}'.format(A=ksp_1,B=ksp_2,C=ksp_3,D=ksp_4,E=ksp_5))

                coord_1 = len(self.bcoord_motion)
                coord_2 = len(self.bcoord_motion[0])
                coord_3 = len(self.bcoord_motion[0][0])
                coord_4 = len(self.bcoord_motion[0][0][0])
                if comm.rank == 0:
                    logging.info('Motion Undersampled traj: {A},{B},{C},{D}'.format(A=coord_1,B=coord_2,C=coord_3,D=coord_4))

                dcf_1 = len(self.bdcf_motion)
                dcf_2 = len(self.bdcf_motion[0])
                dcf_3 = len(self.bdcf_motion[0][0])
                idx = list(idx)
                idx_temp[b] = idx.count(1)             
                if comm.rank == 0:
                    logging.info('Motion Undersampled dens: {A},{B},{C}'.format(A=dcf_1,B=dcf_2,C=dcf_3))
            if b == self.B -1:
                idx = (resp >= bins[b]) & (resp <= bins[b + 1])
                self.bksp_motion.append(ksp[:, :, idx])
                self.bcoord_motion.append(coord[idx])
                self.bdcf_motion.append(dcf[idx])
                self.dual_q_motion.append(dual_q[:, :, idx])
                ksp_1 = len(self.bksp_motion)
                ksp_2 = len(self.bksp_motion[0])
                ksp_3 = len(self.bksp_motion[0][0]) 
                ksp_4 = len(self.bksp_motion[0][0][0])
                ksp_5 = len(self.bksp_motion[0][0][0][0])
                if comm.rank == 0:
                    logging.info('Motion Undersampled k-space: {A},{B},{C},{D},{E}'.format(A=ksp_1,B=ksp_2,C=ksp_3,D=ksp_4,E=ksp_5))

                coord_1 = len(self.bcoord_motion)
                coord_2 = len(self.bcoord_motion[0])
                coord_3 = len(self.bcoord_motion[0][0])
                coord_4 = len(self.bcoord_motion[0][0][0])
                if comm.rank == 0:
                    logging.info('Motion Undersampled traj: {A},{B},{C},{D}'.format(A=coord_1,B=coord_2,C=coord_3,D=coord_4))
                
                dcf_1 = len(self.bdcf_motion)
                dcf_2 = len(self.bdcf_motion[0])
                dcf_3 = len(self.bdcf_motion[0][0])
                idx = list(idx)
                idx_temp[b] = idx.count(1)             
                if comm.rank == 0:
                    logging.info('Motion Undersampled dens: {A},{B},{C}'.format(A=dcf_1,B=dcf_2,C=dcf_3))
        del ksp
        del coord
        del dcf
        del dual_q
        
        temp_size = round(np.min(idx_temp))
       
        bksp_motion_cut = np.zeros((self.B,self.E,len(self.bksp_motion[0][0]),temp_size,len(self.bksp_motion[0][0][0][0])),dtype='complex64')
        dual_q_motion_cut = np.zeros((self.B,self.E,len(self.dual_q_motion[0][0]),temp_size,len(self.dual_q_motion[0][0][0][0])),dtype='complex64')
        bcoord_motion_cut = np.zeros((self.B,temp_size,len(self.bcoord_motion[0][0]),len(self.bcoord_motion[0][0][0])))
        bdcf_motion_cut = np.zeros((self.B,temp_size,len(self.bdcf_motion[0][0])))
        
        for b in range(B):
            for e in range(self.E):
                for c in range(self.C):
                    bksp_motion_cut[b,e,c,...] = self.bksp_motion[b][e][c][0:temp_size]
                    dual_q_motion_cut[b,e,c,...] = self.dual_q_motion[b][e][c][0:temp_size]
            bcoord_motion_cut[b,...] = self.bcoord_motion[b][0:temp_size]
            bdcf_motion_cut[b,...] = self.bdcf_motion[b][0:temp_size]
        
        del self.bksp_motion
        del self.dual_q_motion
        del self.bcoord_motion
        del self.bdcf_motion
        
        ksp_temp_size = bksp_motion_cut.shape
        if comm.rank == 0:
            logging.info('Cut ksp size: {A}'.format(A=ksp_temp_size))                         
        traj_temp_size = bcoord_motion_cut.shape
        if comm.rank == 0:
            logging.info('Cut traj size: {A}'.format(A=traj_temp_size))     
        dens_temp_size = bdcf_motion_cut.shape
        if comm.rank == 0:
            logging.info('Cut dens size: {A}'.format(A=dens_temp_size)) 
        if comm.rank == 0:
            logging.info('Undersampling Factor: {A}'.format(A=undersampling))                 
        undersampled_TR = round(np.floor(temp_size/undersampling))
        if comm.rank == 0:
            logging.info('TRs for undersampling: {A}'.format(A=undersampled_TR))
        overlap = math.ceil((undersampled_TR*self.E-temp_size)/(self.E-1))
        if comm.rank == 0:
            logging.info('Overlap: {A}'.format(A=overlap))
            
        bksp_motion_cut = np.transpose(bksp_motion_cut, [1,0,2,3,4])    
        dual_q_motion_cut = np.transpose(dual_q_motion_cut, [1,0,2,3,4])    
             
        TR_total = np.linspace(0,temp_size-1, num=temp_size).astype(int)
        

        if self.random: # Random undersampling

            if undersampling < self.E:
                TR_echo1 = np.round(np.linspace(0, temp_size-1, num=undersampled_TR)).astype(int)

                if overlap > undersampled_TR - overlap: 
                    TR_echo2_total = np.sort(np.array(list(set(TR_total).difference(TR_echo1))))
                    idx_echo2_part1 = np.round(np.linspace(0, len(TR_echo2_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo2_part1 = TR_echo2_total[idx_echo2_part1]
                    idx_echo2_part2 = np.round(np.linspace(0, len(TR_echo1)-1, num=overlap)).astype(int)
                    TR_echo2_part2 = TR_echo1[idx_echo2_part2]
                    TR_echo2 = np.zeros(len(TR_echo2_part1)+len(TR_echo2_part2))
                    TR_echo2[0:len(TR_echo2_part1)] = TR_echo2_part1
                    TR_echo2[len(TR_echo2_part1):] = TR_echo2_part2
                    TR_echo2 = np.sort(TR_echo2).astype(int)

                    overlap_new = overlap - len(TR_echo2_part1)

                    TR_echo3_total = np.sort(np.array(list(set(TR_echo2_total).difference(TR_echo2_part1))))
                    idx_echo3_part1 = np.round(np.linspace(0, len(TR_echo3_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo3_part1 = TR_echo3_total[idx_echo3_part1]
                    idx_echo3_part2 = np.round(np.linspace(0, len(TR_echo2_part2)-1, num=overlap_new)).astype(int)
                    TR_echo3_part2 = TR_echo2_part2[idx_echo3_part2]
                    TR_echo3_part3 = TR_echo2_part1
                    TR_echo3 = np.zeros(len(TR_echo3_part1)+len(TR_echo3_part2)+len(TR_echo3_part3))
                    TR_echo3[0:len(TR_echo3_part1)] = TR_echo3_part1
                    TR_echo3[len(TR_echo3_part1):len(TR_echo3_part1)+len(TR_echo3_part2)] = TR_echo3_part2
                    TR_echo3[len(TR_echo3_part1)+len(TR_echo3_part2):] = TR_echo3_part3
                    TR_echo3 = np.sort(TR_echo3).astype(int)

                    TR_echo4_total = np.sort(np.array(list(set(TR_echo3_total).difference(TR_echo3_part1))))
                    idx_echo4_part1 = np.round(np.linspace(0, len(TR_echo4_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo4_part1 = TR_echo4_total[idx_echo4_part1]
                    idx_echo4_part2 = np.round(np.linspace(0, len(TR_echo3_part2)-1, num=overlap_new)).astype(int)
                    TR_echo4_part2 = TR_echo3_part2[idx_echo4_part2]
                    TR_echo4_part3 = TR_echo3_part1
                    TR_echo4 = np.zeros(len(TR_echo4_part1)+len(TR_echo4_part2)+len(TR_echo4_part3))
                    TR_echo4[0:len(TR_echo4_part1)] = TR_echo4_part1
                    TR_echo4[len(TR_echo4_part1):len(TR_echo4_part1)+len(TR_echo4_part2)] = TR_echo4_part2
                    TR_echo4[len(TR_echo4_part1)+len(TR_echo4_part2):] = TR_echo4_part3
                    TR_echo4 = np.sort(TR_echo4).astype(int)

                    TR_echo5_total = np.sort(np.array(list(set(TR_echo4_total).difference(TR_echo4_part1))))
                    idx_echo5_part1 = np.round(np.linspace(0, len(TR_echo5_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo5_part1 = TR_echo5_total[idx_echo5_part1]
                    idx_echo5_part2 = np.round(np.linspace(0, len(TR_echo4_part2)-1, num=overlap_new)).astype(int)
                    TR_echo5_part2 = TR_echo4_part2[idx_echo5_part2]
                    TR_echo5_part3 = TR_echo4_part1
                    TR_echo5 = np.zeros(len(TR_echo5_part1)+len(TR_echo5_part2)+len(TR_echo5_part3))
                    TR_echo5[0:len(TR_echo5_part1)] = TR_echo5_part1
                    TR_echo5[len(TR_echo5_part1):len(TR_echo5_part1)+len(TR_echo5_part2)] = TR_echo5_part2
                    TR_echo5[len(TR_echo5_part1)+len(TR_echo5_part2):] = TR_echo5_part3
                    TR_echo5 = np.sort(TR_echo5).astype(int)
                    
                    TR_echo6_total = np.sort(np.array(list(set(TR_echo5_total).difference(TR_echo5_part1))))
                    idx_echo6_part1 = np.round(np.linspace(0, len(TR_echo6_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo6_part1 = TR_echo6_total[idx_echo6_part1]
                    idx_echo6_part2 = np.round(np.linspace(0, len(TR_echo5_part2)-1, num=overlap_new)).astype(int)
                    TR_echo6_part2 = TR_echo5_part2[idx_echo6_part2]
                    TR_echo6_part3 = TR_echo5_part1
                    TR_echo6 = np.zeros(len(TR_echo6_part1)+len(TR_echo6_part2)+len(TR_echo6_part3))
                    TR_echo6[0:len(TR_echo6_part1)] = TR_echo6_part1
                    TR_echo6[len(TR_echo6_part1):len(TR_echo6_part1)+len(TR_echo6_part2)] = TR_echo6_part2
                    TR_echo6[len(TR_echo6_part1)+len(TR_echo6_part2):] = TR_echo6_part3
                    TR_echo6 = np.sort(TR_echo6).astype(int)
                    
                    TR_idx = np.zeros([self.E,temp_size],dtype=bool)
                    TR_idx[0,TR_echo1]=1
                    TR_idx[1,TR_echo2]=1
                    TR_idx[2,TR_echo3]=1
                    TR_idx[3,TR_echo4]=1
                    TR_idx[4,TR_echo5]=1
                    TR_idx[5,TR_echo6]=1

                else: 

                    TR_echo2_total = np.sort(np.array(list(set(TR_total).difference(TR_echo1))))
                    idx_echo2_part1 = np.round(np.linspace(0, len(TR_echo2_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo2_part1 = TR_echo2_total[idx_echo2_part1]
                    idx_echo2_part2 = np.round(np.linspace(0, len(TR_echo1)-1, num=overlap)).astype(int)
                    TR_echo2_part2 = TR_echo1[idx_echo2_part2]
                    TR_echo2 = np.zeros(len(TR_echo2_part1)+len(TR_echo2_part2))
                    TR_echo2[0:len(TR_echo2_part1)] = TR_echo2_part1
                    TR_echo2[len(TR_echo2_part1):] = TR_echo2_part2
                    TR_echo2 = np.sort(TR_echo2).astype(int)

                    TR_echo3_total = np.sort(np.array(list(set(TR_echo2_total).difference(TR_echo2_part1))))
                    idx_echo3_part1 = np.round(np.linspace(0, len(TR_echo3_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo3_part1 = TR_echo3_total[idx_echo3_part1]
                    idx_echo3_part2 = np.round(np.linspace(0, len(TR_echo2_part1)-1, num=overlap)).astype(int)
                    TR_echo3_part2 = TR_echo2_part1[idx_echo3_part2]
                    TR_echo3 = np.zeros(len(TR_echo3_part1)+len(TR_echo3_part2))
                    TR_echo3[0:len(TR_echo3_part1)] = TR_echo3_part1
                    TR_echo3[len(TR_echo3_part1):] = TR_echo3_part2
                    TR_echo3 = np.sort(TR_echo3).astype(int)
                    
                    TR_echo4_total = np.sort(np.array(list(set(TR_echo3_total).difference(TR_echo3_part1))))
                    idx_echo4_part1 = np.round(np.linspace(0, len(TR_echo4_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo4_part1 = TR_echo4_total[idx_echo4_part1]
                    idx_echo4_part2 = np.round(np.linspace(0, len(TR_echo3_part1)-1, num=overlap)).astype(int)
                    TR_echo4_part2 = TR_echo3_part1[idx_echo4_part2]
                    TR_echo4 = np.zeros(len(TR_echo4_part1)+len(TR_echo4_part2))
                    TR_echo4[0:len(TR_echo4_part1)] = TR_echo4_part1
                    TR_echo4[len(TR_echo4_part1):] = TR_echo4_part2
                    TR_echo4 = np.sort(TR_echo4).astype(int)
                    
                    TR_echo5_total = np.sort(np.array(list(set(TR_echo4_total).difference(TR_echo4_part1))))
                    idx_echo5_part1 = np.round(np.linspace(0, len(TR_echo5_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo5_part1 = TR_echo5_total[idx_echo5_part1]
                    idx_echo5_part2 = np.round(np.linspace(0, len(TR_echo4_part1)-1, num=overlap)).astype(int)
                    TR_echo5_part2 = TR_echo4_part1[idx_echo5_part2]
                    TR_echo5 = np.zeros(len(TR_echo5_part1)+len(TR_echo5_part2))
                    TR_echo5[0:len(TR_echo5_part1)] = TR_echo5_part1
                    TR_echo5[len(TR_echo5_part1):] = TR_echo5_part2
                    TR_echo5 = np.sort(TR_echo5).astype(int)
                    
                    TR_echo6_total = np.sort(np.array(list(set(TR_echo5_total).difference(TR_echo5_part1))))
                    idx_echo6_part1 = np.round(np.linspace(0, len(TR_echo6_total)-1, num=undersampled_TR-overlap)).astype(int)
                    TR_echo6_part1 = TR_echo6_total[idx_echo6_part1]
                    idx_echo6_part2 = np.round(np.linspace(0, len(TR_echo5_part1)-1, num=overlap)).astype(int)
                    TR_echo6_part2 = TR_echo5_part1[idx_echo6_part2]
                    TR_echo6 = np.zeros(len(TR_echo6_part1)+len(TR_echo6_part2))
                    TR_echo6[0:len(TR_echo6_part1)] = TR_echo6_part1
                    TR_echo6[len(TR_echo6_part1):] = TR_echo6_part2
                    TR_echo6 = np.sort(TR_echo6).astype(int)

                    TR_idx = np.zeros([self.E,temp_size],dtype=bool)
                    TR_idx[0,TR_echo1]=1
                    TR_idx[1,TR_echo2]=1
                    TR_idx[2,TR_echo3]=1
                    TR_idx[3,TR_echo4]=1
                    TR_idx[4,TR_echo5]=1
                    TR_idx[5,TR_echo6]=1

            else:
                TR_echo1 = np.round(np.linspace(0, temp_size-1, num=undersampled_TR)).astype(int)

                TR_echo2_total = np.sort(np.array(list(set(TR_total).difference(TR_echo1))))
                idx_echo2 = np.round(np.linspace(0,len(TR_echo2_total)-1, num=undersampled_TR)).astype(int)
                TR_echo2 = TR_echo2_total[idx_echo2]

                TR_echo3_total = np.sort(np.array(list(set(TR_echo2_total).difference(TR_echo2))))
                idx_echo3 = np.round(np.linspace(0,len(TR_echo3_total)-1, num=undersampled_TR)).astype(int)
                TR_echo3 = TR_echo3_total[idx_echo3]

                TR_echo4_total = np.sort(np.array(list(set(TR_echo3_total).difference(TR_echo3))))
                idx_echo4 = np.round(np.linspace(0,len(TR_echo4_total)-1, num=undersampled_TR)).astype(int)
                TR_echo4 = TR_echo4_total[idx_echo4]

                TR_echo5_total = np.sort(np.array(list(set(TR_echo4_total).difference(TR_echo4))))
                idx_echo5 = np.round(np.linspace(0,len(TR_echo5_total)-1, num=undersampled_TR)).astype(int)
                TR_echo5 = TR_echo5_total[idx_echo5]

                TR_echo6_total = np.sort(np.array(list(set(TR_echo5_total).difference(TR_echo5))))
                idx_echo6 = np.round(np.linspace(0,len(TR_echo6_total)-1, num=undersampled_TR)).astype(int)
                TR_echo6 = TR_echo6_total[idx_echo6]

                TR_idx = np.zeros([self.E,temp_size],dtype=bool)
                TR_idx[0,TR_echo1]=1
                TR_idx[1,TR_echo2]=1
                TR_idx[2,TR_echo3]=1
                TR_idx[3,TR_echo4]=1
                TR_idx[4,TR_echo5]=1
                TR_idx[5,TR_echo6]=1

            for e in range(self.E):
                if undersampling < self.E:
                    if comm.rank == 0:
                        logging.info('TR overlap required')
                    TR_idx_e = TR_idx[e,:]

                    self.bksp_temp.append(np.transpose(bksp_motion_cut[e,:,:,TR_idx_e,:],[1,2,0,3]))
                    self.dual_q_temp.append(np.transpose(dual_q_motion_cut[e,:,:,TR_idx_e,:],[1,2,0,3]))
                    self.bcoord_temp.append(bcoord_motion_cut[:,TR_idx_e,:,:])    
                    self.bdcf_temp.append(bdcf_motion_cut[:,TR_idx_e,:])

                    ksp_1 = len(self.bksp_temp)
                    ksp_2 = len(self.bksp_temp[0])
                    ksp_3 = len(self.bksp_temp[0][0])
                    ksp_4 = len(self.bksp_temp[0][0][0])
                    ksp_5 = len(self.bksp_temp[0][0][0][0])

                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled k-space: {A},{B},{C},{D},{E}'.format(A=ksp_1,B=ksp_2,C=ksp_3,D=ksp_4,E=ksp_5))

                    coord_1 = len(self.bcoord_temp)
                    coord_2 = len(self.bcoord_temp[0])
                    coord_3 = len(self.bcoord_temp[0][0])
                    coord_4 = len(self.bcoord_temp[0][0][0])
                    coord_5 = len(self.bcoord_temp[0][0][0][0])

                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled traj: {A},{B},{C},{D},{E}'.format(A=coord_1,B=coord_2,C=coord_3,D=coord_4,E=coord_5))

                    dcf_1 = len(self.bdcf_temp)
                    dcf_2 = len(self.bdcf_temp[0])
                    dcf_3 = len(self.bdcf_temp[0][0])
                    dcf_4 = len(self.bdcf_temp[0][0][0])

                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled dcf: {A},{B},{C},{D}'.format(A=dcf_1,B=dcf_2,C=dcf_3,D=dcf_4))

                if undersampling >= self.E:
                    if comm.rank == 0:
                        logging.info('No TR overlap required')
                    TR_idx_e = TR_idx[e,:]
                    self.bksp_temp.append(np.transpose(bksp_motion_cut[e,:,:,TR_idx_e,:],[1,2,0,3]))
                    self.dual_q_temp.append(np.transpose(dual_q_motion_cut[e,:,:,TR_idx_e,:],[1,2,0,3]))
                    self.bcoord_temp.append(bcoord_motion_cut[:,TR_idx_e,:,:])    
                    self.bdcf_temp.append(bdcf_motion_cut[:,TR_idx_e,:])
                    
                    ksp_1 = len(self.bksp_temp)
                    ksp_2 = len(self.bksp_temp[0])
                    ksp_3 = len(self.bksp_temp[0][0])
                    ksp_4 = len(self.bksp_temp[0][0][0])
                    ksp_5 = len(self.bksp_temp[0][0][0][0])

                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled k-space: {A},{B},{C},{D},{E}'.format(A=ksp_1,B=ksp_2,C=ksp_3,D=ksp_4,E=ksp_5))

                    coord_1 = len(self.bcoord_temp)
                    coord_2 = len(self.bcoord_temp[0])
                    coord_3 = len(self.bcoord_temp[0][0])
                    coord_4 = len(self.bcoord_temp[0][0][0])
                    coord_5 = len(self.bcoord_temp[0][0][0][0])

                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled traj: {A},{B},{C},{D},{E}'.format(A=coord_1,B=coord_2,C=coord_3,D=coord_4,E=coord_5))

                    dcf_1 = len(self.bdcf_temp)
                    dcf_2 = len(self.bdcf_temp[0])
                    dcf_3 = len(self.bdcf_temp[0][0])
                    dcf_4 = len(self.bdcf_temp[0][0][0])
                    
                    if comm.rank == 0:
                        logging.info('Echo Random-Undersampled dcf: {A},{B},{C},{D}'.format(A=dcf_1,B=dcf_2,C=dcf_3,D=dcf_4))

        else: # Uniform
            TR_temp1 = np.round(np.linspace(0, temp_size-1, num=undersampled_TR)).astype(int)
            TR_idx = np.zeros(temp_size,dtype=bool)
            TR_idx[TR_temp1[:]]=1

            for e in range(self.E):
                
                self.bksp_temp.append(np.transpose(bksp_motion_cut[e,:,:,TR_idx,:],[1,2,0,3]))
                self.dual_q_temp.append(np.transpose(dual_q_motion_cut[e,:,:,TR_idx,:],[1,2,0,3]))
                self.bcoord_temp.append(bcoord_motion_cut[:,TR_idx,:,:])    
                self.bdcf_temp.append(bdcf_motion_cut[:,TR_idx,:])

                ksp_1 = len(self.bksp_temp)
                ksp_2 = len(self.bksp_temp[0])
                ksp_3 = len(self.bksp_temp[0][0])
                ksp_4 = len(self.bksp_temp[0][0][0])
                ksp_5 = len(self.bksp_temp[0][0][0][0])

                if comm.rank == 0:
                    logging.info('Echo Uniform-Undersampled k-space: {A},{B},{C},{D},{E}'.format(A=ksp_1,B=ksp_2,C=ksp_3,D=ksp_4,E=ksp_5))

                coord_1 = len(self.bcoord_temp)
                coord_2 = len(self.bcoord_temp[0])
                coord_3 = len(self.bcoord_temp[0][0])
                coord_4 = len(self.bcoord_temp[0][0][0])
                coord_5 = len(self.bcoord_temp[0][0][0][0])

                if comm.rank == 0:
                    logging.info('Echo Uniform-Undersampled traj: {A},{B},{C},{D},{E}'.format(A=coord_1,B=coord_2,C=coord_3,D=coord_4,E=coord_5))

                dcf_1 = len(self.bdcf_temp)
                dcf_2 = len(self.bdcf_temp[0])
                dcf_3 = len(self.bdcf_temp[0][0])
                dcf_4 = len(self.bdcf_temp[0][0][0])

                if comm.rank == 0:
                    logging.info('Echo Uniform-Undersampled dcf: {A},{B},{C},{D}'.format(A=dcf_1,B=dcf_2,C=dcf_3,D=dcf_4))


        self.bksp_temp = np.transpose(self.bksp_temp,[1,0,2,3,4])
        self.dual_q_temp = np.transpose(self.dual_q_temp,[1,0,2,3,4])
        self.bcoord_temp = np.transpose(self.bcoord_temp,[1,0,2,3,4])
        self.bdcf_temp = np.transpose(self.bdcf_temp,[1,0,2,3])


        
        for b in range(B):
            self.bksp_temp[b] = self.bksp_temp[b]/np.max(np.abs(self.bksp_temp[b].ravel()))*1000 ## k-space normalization
            self.bksp.append(
                sp.to_device(self.bksp_temp[b,...], self.device))
            self.dual_q_.append(
                sp.to_device(self.dual_q_temp[b,...], self.device))
            self.bcoord.append(
                sp.to_device(self.bcoord_temp[b,...], self.device))
            self.bdcf.append(
                sp.to_device(self.bdcf_temp[b,...], self.device))

        del self.bksp_temp
        del self.dual_q_temp
        del self.bcoord_temp
        del self.bdcf_temp

    # Primal-Dual Algorithm
    def pdinit(self, mrimg):

        dual_p_m = self.xp.zeros_like(mrimg)
        dual_p_w1 = []
        dual_p_w2 = []
        dual_q = self.dual_q_
        
        return dual_p_m, dual_p_w1, dual_p_w2, dual_q

    def pdhg(self, primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q, it):
        
        # p: dual variable for total variation 
        # q: dual variable for data term
        # u: primal variable
        
        primal_u_old = primal_u_tmp

        ### @Dual variable p ###
        ###
        # 1) update p
        # p^k+1 = prox_P (p^k + sigma * \partial_t u^k)
        diff = self.xp.zeros_like(dual_p_m)
        for b in range(self.B):
            if b < self.B - 1:
                diff[b] = primal_u[b + 1] - primal_u[b]
            if b == self.B - 1:
                diff[b] = 0
      
        # proximal operator
        dual_p_m = dual_p_m + self.sigma * diff

        absp = self.xp.abs(dual_p_m)
        dual_p_m = dual_p_m/self.xp.maximum(1, absp/self.lambda1)
            
        X1 = [0]
        X2 = [1,2,3]
        temp_range1 = tuple(X1)
        temp_range2 = tuple(X2)
        W1 = sp.linop.Wavelet(primal_u[0].shape, wave_name='db1', axes=temp_range1)
        W2 = sp.linop.Wavelet(primal_u[0].shape, wave_name='db6', axes=temp_range2)
        
        if it == 0:
            wav1_shape = []
            wav1_temp = W1 * primal_u[0]
            wav1_shape = self.xp.zeros([self.B] + list(wav1_temp.shape),dtype=wav1_temp.dtype)
            dual_p_w1 = self.xp.zeros_like(wav1_shape)
            
            wav2_shape = []
            wav2_temp = W2 * primal_u[0]
            wav2_shape = self.xp.zeros([self.B] + list(wav2_temp.shape),dtype=wav2_temp.dtype)
            dual_p_w2 = self.xp.zeros_like(wav2_shape) 

        wav1 = self.xp.zeros_like(dual_p_w1)
        wav2 = self.xp.zeros_like(dual_p_w2)
        
        
        for b in range(self.B):
            wav1[b] = W1 * primal_u[b]
            dual_p_w1[b] = dual_p_w1[b] + self.sigma * wav1[b]       

            wav2[b]  = W2 * primal_u[b]
            dual_p_w2[b] = dual_p_w2[b] + self.sigma * wav2[b]  

        
        for b in range(self.B):

            dual_p_w1[b] = dual_p_w1[b] - pywt.threshold(dual_p_w1[b], self.lambda2, 'soft')
            dual_p_w2[b] = dual_p_w2[b] - pywt.threshold(dual_p_w2[b], self.lambda3, 'soft')

        ### @Dual Variable q ###
        ###
        # 2) update q
        # q^k+1 = prox_Q (q^k + sigma * (FSu^k - y))
        #tmpp = self.xp.zeros_like(self.dual_q_)
        
        for b in range(self.B):
            tmp =  self.xp.zeros_like(self.dual_q_[b])
            for c in range(self.C):
                for e in range(self.E):
                    mps_c = sp.to_device(self.mps[c], self.device)
                    tmp[e, c] = sp.nufft(primal_u[b, e] * mps_c, self.bcoord[b][e]) - self.bksp[b][e][c] 
                    
            # proximal operator
            dual_q[b] = (dual_q[b] + self.sigma * tmp)/(1 + self.sigma)
        
        ### @PRIMAL VARIABLE ###
        ##
        # update u
        # u^k+1 = u^k - tau * ( (FS)^H q^k+1 + divp )
        
        # 2) Compute divergene w/backward gradient
        divp_m = self.xp.zeros_like(primal_u)
        for b in range(self.B):
            if b == 0:
                divp_m[b] = dual_p_m[b]
            if b > 0 and b < self.B - 1:
                divp_m[b] = dual_p_m[b] - dual_p_m[b - 1]
            if b == self.B - 1:
                divp_m[b] = -dual_p_m[b - 1]
        
        tmp = self.xp.zeros_like(primal_u)
        
        for b in range(self.B):
            for e in range(self.E):
                for c in range(self.C):
                    mps_c = sp.to_device(self.mps[c], self.device)
                    tmp[b, e] += sp.nufft_adjoint(self.bdcf[b][e]*dual_q[b][e][c], self.bcoord[b][e], oshape=primal_u.shape[2:]) * self.xp.conj(mps_c)

        divp_w1 = self.xp.zeros_like(primal_u)
        divp_w2 = self.xp.zeros_like(primal_u)
        
        for b in range(self.B):
            divp_w1[b] = W1.H(dual_p_w1[b])
            
        for b in range(self.B):
            divp_w2[b] = W2.H(dual_p_w2[b])

        if self.comm is not None:
            self.comm.allreduce(tmp)
 
        
        sp.axpy(primal_u_tmp, -self.tau, tmp  - divp_m  + divp_w1 + divp_w2)

        ### @AUXILIARY UPDATE ###
        primal_u = 2*primal_u_tmp - primal_u_old
        
        return primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q

    def run(self):
        done = False
        while not done:
            with tqdm(total=self.max_iter, desc='MotionResolvedRecon\n',
                      disable=not self.show_pbar) as pbar:

                with self.device:
                    mrimg = self.xp.zeros([self.B] + [self.E] + self.img_shape,
                                         dtype=self.mps.dtype)

                    dual_p_m, dual_p_w1,dual_p_w2, dual_q = self.pdinit(mrimg)

                    primal_u_old = self.xp.zeros_like(mrimg)
                    primal_u_tmp = self.xp.zeros_like(mrimg)
                    start_time = time.monotonic()  
                    tolerance = np.zeros(self.max_iter)
                    for it in range(self.max_iter):

                        mrimg_od = mrimg
                        mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q = \
                        self.pdhg(mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q, it)
                        _tol = self.xp.linalg.norm(abs(mrimg_od - mrimg))/self.xp.linalg.norm(abs(mrimg_od))
                        pbar.set_postfix(tol=_tol)
                        tolerance[it] = _tol

                        if (_tol < self.tol):
                            break
                        pbar.update()
                    done = True
        end_time = time.monotonic()
        total_time = end_time - start_time
        time_file = os.path.join(args.input_dir, args.img_file)
        np.savetxt(time_file+'_total_time.txt',np.repeat(total_time,2),fmt='%4.4f')
        return mrimg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Motion Resolved Reconstruction (PDHG).')
    parser.add_argument('--num_bins', type=int, default=6, help='Number of phases.')
    parser.add_argument('--lambda1', type=float, default=1e-6,
                        help='Regularization for motion.')
    parser.add_argument('--lambda2', type=float, default=1e-6,
                        help='Regularization for echo.')
    parser.add_argument('--lambda3', type=float, default=1e-6,
                        help='Regularization for image.')
    parser.add_argument('--undersampling', type=float, default=3,
                        help='Undersampling factor.')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum iteration.')
    parser.add_argument('--random', action='store_true', help='Random undersampling')
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

    with h5py.File('Gd_Phantom_Cones_With_Motion_1.h5', 'r') as hf:
        ksp_1   = hf["ksp"][:]
        coord = hf["coord"][:]
        dcf  = hf["dcf"][:]
        img_shape = hf["imageDim"][:]
        voxelSize = hf["voxelSize"][:]
        te    = hf["te"][:]
        tr    = hf["tr"][...]
        cf    = hf["cf"][...]
    img_shape = img_shape.tolist()

    with h5py.File('Gd_Phantom_Cones_With_Motion_2.h5', 'r') as hf:
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
        logging.info('Reading sensitivity map.')
    
    mps_file = os.path.join(args.input_dir, 'mps')
    mps      = np.squeeze(cfl.read_cfl(mps_file))
    
    if comm.rank == 0:
        logging.info('Reading resp signal.')
    
    resp_file = os.path.join(args.input_dir, 'resp')
    resp      = np.squeeze(cfl.read_cfl(resp_file)).real
    
    # Coil sensitivity map normalization
    if 1:
        mpsSOS = np.sum(abs(mps)**2, 0)**0.5
        for c in range(mps.shape[0]):
            mps[c] /= mpsSOS
    
    # Split between MPI nodes: make sure we split COILS of ksp and mps
    dual_q = np.zeros((ksp.shape[0], ksp.shape[1], ksp.shape[2], ksp.shape[3]), dtype=np.complex64)
    dual_q = np.array_split(np.transpose(dual_q, (1, 0, 2, 3)), comm.size)[comm.rank]
    dual_q = np.transpose(dual_q, (1, 0, 2, 3))
    
    ksp = np.array_split(np.transpose(ksp, (1, 0, 2, 3)), comm.size)[comm.rank] # coil, echo, readout
    mps = np.array_split(mps, comm.size)[comm.rank]
    ksp = np.transpose(ksp, (1, 0, 2, 3)) # echo, coil, readout

    #import pdb; pdb.set_trace() # breakpoint
    img = np.empty([args.num_bins, 1, 1, num_echoes, 1, 1] + img_shape, dtype=np.complex64)
    
    if comm.rank == 0:
        logging.info('Running motion-resolved reconstruction (PDHG): #echos={E}, #coils={C}, #bins={B}'.format(
            E=num_echoes, C=num_coils, B=args.num_bins))

    mrimg = MotionResolvedRecon(ksp, coord, dcf, mps, resp, dual_q, args.num_bins, args.random,
                            max_iter=9999, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3, undersampling=args.undersampling, sigma=0.1, tau=0.1, tol=0.01, device=device, margin=2, comm=comm).run()
    
    with device:
        img[:, 0, 0, :, 0, 0, :, :, :] = sp.to_device(mrimg)
        
    # Writing.
    if comm.rank == 0:
        logging.info('Writing as CFL files.')
        
        img_file = os.path.join(args.input_dir, args.img_file)
        cfl.write_cfl(img_file, img)
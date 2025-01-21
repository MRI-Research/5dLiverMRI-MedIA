# 5D Image Reconstruction for Acceleraetd Free-Breathing Liver MRI

This repository contains the reconstruction code and datasets of the following manuscript currently under minor revision.

> Kang et al. 5D Image Reconstruction Exploiting Space-Motion-Echo Sparsity for Accelerated Free-Breathing Quantitative Liver MRI. <em>Medical Image Analysis</em>.

Contact: MungSoo Kang (<kangms0511@gmail.com>)

# Contents
```bash
$ tree
.
├── README.md
├── recon_4D_motion_resolved_PDHG.py               # 4D recon w/o undersampling
├── recon_4D_motion_resolved_PDHG_undersampling.py # 4D recon w/undersampling
├── recon_5D_motion_resolved_PDHG.py               # 5D recon w/o undersampling
├── recon_5D_motion_resolved_PDHG_undersampling.py # 5D recon w/undersampling
├── recon_coil_sensitivity.py                      # JSENSE for coil sensitivity estimation
├── recon_gridding_motion_averaged.py              # motion averaged recon
└── recon_respiratory_signal.py                    # respiratory motion estimation

1 directory, 8 files
```

## Tested Environment

Python packages:
```
python=3.9.18
scipy=1.11.4
cupy=8.3.0
sigpy=0.1.26 (pywt.waverecn: mode = 'reflect') 
numpy=1.23.1
mpi4py=3.1.4
pywt=1.5.0
```
Nvidia A100 GPUs 

## Example Usage

```bash
# 1. Motion signal estimation
python3 recon_respiratory_signal.py --verbose 'input_dir' 'output_dir/resp'

# 2. Coil sensitivity estimation
python3 recon_coil_sensitivity.py --device 0 --show_pbar --verbose 'input_dir' 'output_dir/mps'

# 3. Gridding (motion-averaged) reconstruction
python3 recon_gridding_motion_averaged.py --device 0 --verbose 'input_dir' 'output_dir'

# 4. 4D motion resolved reconstruction
NUM_GPUS=4
## 4.1. No undersampling
mpiexec -n $NUM_GPUS python3 recon_4D_motion_resolved_PDHG.py --num_bins 4 --lambda1 1e-6 \
   --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
## 4.2. with random undersampling
mpiexec -n $NUM_GPUS python3 recon_4D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 \
   --undersampling 60 --random --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
## 4.3. with uniform undersampling
mpiexec -n $NUM_GPUS python3 recon_4D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 \
   --undersampling 60 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'

# 5. 5D motion resolved reconstruction
## 4.1. No undersampling
mpiexec -n $NUM_GPUS python3 recon_5D_motion_resolved_PDHG.py --num_bins 4 --lambda1 1e-6 \
   --lambda2 8e-7 --lambda3 3e-6 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
## 4.2. with random undersampling
mpiexec -n $NUM_GPUS python3 recon_5D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 \
   --lambda2 8e-7 --lambda3 3e-6 --undersampling 60 --random --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
## 4.3. with uniform undersampling
mpiexec -n $NUM_GPUS python3 recon_5D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 \
   --lambda2 8e-7 --lambda3 3e-6 --undersampling 60 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
```

# Datasets

Gadolinium phantom MRI k-space raw datasets are available on Zenodo:

1. Cones without motion: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14707963.svg)](https://doi.org/10.5281/zenodo.14707963), [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14707967.svg)](https://doi.org/10.5281/zenodo.14707967)
2. Cones with motion: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14691117.svg)](https://doi.org/10.5281/zenodo.14691117), [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14691162.svg)](https://doi.org/10.5281/zenodo.14691162)
3. Cartesian: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14691167.svg)](https://doi.org/10.5281/zenodo.14691167)

## File I/O

To read cones data:
```python
import h5py

# Read the stored h5 file
with h5py.File('Gd_Phantom_Cones_With_Motion_1.h5', 'r') as hf:
    ksp_1     = hf["ksp"][:]
    coord     = hf["coord"][:]
    dcf       = hf["dcf"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 

with h5py.File('Gd_Phantom_Cones_With_Motion_2.h5', 'r') as hf:
    ksp_2     = hf["ksp"][:]
    coord     = hf["coord"][:]
    dcf       = hf["dcf"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 

# Combine
ksp = np.concatenate((ksp_1, ksp_2), axis=1)
```

To read Cartesian data:
```python
import h5py

# Read the stored h5 file
with h5py.File('Gd_Phantom_Cartesian_WO_Motion.h5', 'r') as hf:
    ksp       = hf["ksp"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 
```

## Recon script for Cartesian data

To reconstruct image from Cartesian k-space raw data:
```matlab
% MATLAB
data = h5read('Gd_Phantom_Cartesian_WO_Motion.h5','/ksp');
ksp = complex(data.r,data.i);

myfft = @(func,x) func(func(func(x,[],1),[],2),[],3);
myfftshift1 = @(func,x) func(func(func(x,1),2),3);
myfftshift2 = @(func,x) func(func(x,1),2);

for echo = 1 : 6
    img(:,:,:,:,echo) = myfftshift2(@ifftshift,myfft(@ifft,myfftshift1(@fftshift,ksp(:,:,:,:,echo))));
end
```

## Quantitative Susceptibility Mapping (QSM)

For QSM, please refer to the details of the processing steps in our paper as well as the [MEDI toolbox](https://pre.weill.cornell.edu/mri/pages/qsm.html).

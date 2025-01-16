# 5dLiverMRI-MedIA
Repository for Kang et al. 5D Image Reconstruction Exploiting Space-Motion-Echo Sparsity for Accelerated Free-Breathing Quantitative Liver MRI. Medical Image Analysis

# Code

## Recon code
1. gridding
2. respiratory motion esimatimating code
3. 

## Dependencies

## Usage
```bash
python3 ...
```

# Dataset

## Description 

1. Cartesian w/o motion - kspace data
2. Cones w/o motion - ksp + coord + dcf + sens
3. Cones w/motion - ksp + coord + dcf + sens

For QSM/PDFF/R2* - voxelsize, cf, deltaTE, B0 direction, TEs

Zenodo link - .npy or .h5 

## File I/O

To read multi-echo cones data:
```python
import h5py

# Read the stored h5 file
with h5py.File('phantom.h5', 'r') as hf:
    ksp       = hf["ksp"][:]
    coord     = hf["coord"][:]
    dcf       = hf["dcf"][:]
    sens      = hf["sens"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 
```


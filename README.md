# 5dLiverMRI-MedIA
Repository for Kang et al. 5D Image Reconstruction Exploiting Space-Motion-Echo Sparsity for Accelerated Free-Breathing Quantitative Liver MRI. Medical Image Analysis

# Code

## Recon code
1. gridding
2. respiratory motion estimation code
3. coil sensitivity estimation code
4. motion-resolved reconstruction code (4D/5D)
   
## Dependencies

## Usage
```bash
python3 ...
```

# Dataset

## Description 

The following list of h5 files can be downloaded from...

1. `Gd_Phantom_Cartesian_WO_Motion.h5`
2. `Gd_Phantom_Cones_WO_Motion.h5`
3. `Gd_Phantom_Cones_With_Motion.h5`

## File I/O

To read cones data:
```python
import h5py

# Read the stored h5 file
with h5py.File('phantom.h5', 'r') as hf:
    ksp       = hf["ksp"][:]
    coord     = hf["coord"][:]
    dcf       = hf["dcf"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 
```

To read Cartesian data:
```python
import h5py

# Read the stored h5 file
with h5py.File('phantom.h5', 'r') as hf:
    ksp       = hf["ksp"][:]
    imageDim  = hf["imageDim"][:] # matrix size
    voxelSize = hf["voxelSize"][:] # spatial resolution in [cm]
    te        = hf["te"][:] # TE in [sec]
    tr        = hf["tr"][...] # TR in [sec]
    cf        = hf["cf"][...] # center frequency in [Hz] 
```

To reconstruct image from Cartesian k-space:
```matlab

data = h5read('Gd_Phantom_Cartesian_WO_Motion.h5','/ksp');
ksp = complex(data.r,data.i);

myfft = @(func,x) func(func(func(x,[],1),[],2),[],3);
myfftshift1 = @(func,x) func(func(func(x,1),2),3);
myfftshift2 = @(func,x) func(func(x,1),2);

for echo = 1 : 6
    img(:,:,:,:,echo) = myfftshift2(@ifftshift,myfft(@ifft,myfftshift1(@fftshift,ksp(:,:,:,:,echo))));
end
```


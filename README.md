# 5dLiverMRI-MedIA
Repository for Kang et al. 5D Image Reconstruction Exploiting Space-Motion-Echo Sparsity for Accelerated Free-Breathing Quantitative Liver MRI. Medical Image Analysis

# Code

## Recon code

1. respiratory motion estimation code
2. coil sensitivity estimation code
3. gridding (motion-averaged) reconstruction code
4. motion-resolved reconstruction code (4D/5D)
   
## Dependencies

## Usage

1. Motion signal estimation.
```bash 
python3 recon_respiratory_signal.py --verbose 'input_dir' 'output_dir/resp'
```

2. Coil sensitivity estimation.
```bash 
python3 recon_coil_sensitivity.py --device 1 --show_pbar --verbose 'input_dir' 'output_dir/mps'
```

3. Gridding (motion-averaged) reconstruction.
```bash 
python3 recon_gridding_motion_averaged.py --device 1 --verbose 'input_dir' 'output_dir'
```

4. 4D motion-resolved reconstruction.
```bash
(1) No undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_4D_motion_resolved_PDHG.py --num_bins 4 --lambda1 1e-6 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'

(2) Random undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_4D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 --undersampling 60 --random --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'

(3) Uniform undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_4D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 --undersampling 60 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
```

5. 5D motion-resolved reconstruction.
```bash
(1) No undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_5D_motion_resolved_PDHG.py --num_bins 4 --lambda1 1e-6 --lambda2 8e-7 --lambda3 3e-6 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'

(2) Random undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_5D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 --lambda2 8e-7 --lambda3 3e-6 --undersampling 60 --random --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'

(3) Uniform undersampling: mpiexec -n 4 python3 /otazolab_ess/mungsoo-data/motion_resolved_recon_tools/Cones/recon/bin/recon_5D_motion_resolved_PDHG_undersampling.py --num_bins 4 --lambda1 1e-6 --lambda2 8e-7 --lambda3 3e-6 --undersampling 60 --multi_gpu --show_pbar --verbose 'input_dir' 'output_dir'
```

# Dataset

## Description 

The following list of h5 files can be downloaded from...

1. `Gd_Phantom_Cartesian_WO_Motion.h5`
   
2-1. `Gd_Phantom_Cones_WO_Motion_1.h5`
   
2-2. `Gd_Phantom_Cones_WO_Motion_2.h5`

3-1. `Gd_Phantom_Cones_With_Motion_1.h5`

3-2. `Gd_Phantom_Cones_With_Motion_2.h5`

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
# Matlab
data = h5read('Gd_Phantom_Cartesian_WO_Motion.h5','/ksp');
ksp = complex(data.r,data.i);

myfft = @(func,x) func(func(func(x,[],1),[],2),[],3);
myfftshift1 = @(func,x) func(func(func(x,1),2),3);
myfftshift2 = @(func,x) func(func(x,1),2);

for echo = 1 : 6
    img(:,:,:,:,echo) = myfftshift2(@ifftshift,myfft(@ifft,myfftshift1(@fftshift,ksp(:,:,:,:,echo))));
end
```


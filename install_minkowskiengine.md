# How to install MinkowskiEngine on Jitendra's cluster

## Prerequisites

### 1. Install System Dependencies (BLAS/LAPACK)
```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev liblapack-dev libblas-dev
```

### 2. Install Compatible GCC Version
CUDA 11.7 is not compatible with GCC 13. Install GCC 11:
```bash
sudo apt-get install -y gcc-11 g++-11
```

## Required Code Modifications

### 1. Modify `setup.py`

**a) Add conda include directory for cblas.h (after line 215):**
```python
# Add conda include directory for cblas.h
import sys
conda_prefix = sys.prefix
if os.path.exists(os.path.join(conda_prefix, 'include')):
    include_dirs.append(os.path.join(conda_prefix, 'include'))
    print(f"Added conda include directory: {os.path.join(conda_prefix, 'include')}")
```

**b) Add OpenMP flags for NVCC (lines 172-174):**
```python
CC_FLAGS += ["-fopenmp"]
# Pass OpenMP flag to host compiler when nvcc compiles .cpp files
NVCC_FLAGS += ["-Xcompiler", "-fopenmp"]
```

**c) Replace .cpp files with .cu wrappers in GPU source list (lines 266-268):**
```python
"gpu.cu",
"quantization_gpu.cu",      # Instead of "quantization.cpp"
"direct_max_pool_gpu.cu",   # Instead of "direct_max_pool.cpp"
```

### 2. Create Wrapper CUDA Files

**Create `src/quantization_gpu.cu`:**
```cuda
/*
 * Wrapper to include quantization.cpp for GPU builds
 */

#include "quantization.cpp"
```

**Create `src/direct_max_pool_gpu.cu`:**
```cuda
/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * 
 * Wrapper to include direct_max_pool.cpp for GPU builds
 */

#include "direct_max_pool.cpp"
```

### 3. Fix Include Order in `src/spmm.cu`

Move torch headers to the top (lines 27-28):
```cpp
#include <torch/extension.h>
#include <torch/script.h>

#include "gpu.cuh"
#include "math_functions.cuh"
```

### 4. Fix `MinkowskiEngineBackend/__init__.py`

The file is empty by default. Add:
```python
from . import _C
```

## Installation Command

After making all the above modifications:

```bash
CUDA_HOME=/usr/local/cuda-11.7 CC=gcc-11 CXX=g++-11 python setup.py install --user --cuda_home=/usr/local/cuda-11.7 --blas=openblas
```
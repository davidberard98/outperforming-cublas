#include "ATen/ATen.h"
#include "torch/extension.h"

#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>

using bf16 = __nv_bfloat16;

#include "mm_v1.cuh"

TORCH_LIBRARY(gemm, m) { m.def("mm_v1(Tensor a, Tensor b) -> Tensor", &mm_v1); }

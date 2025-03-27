#include "ATen/ATen.h"
#include "torch/extension.h"

#include "mm_v1.cuh"

TORCH_LIBRARY(gemm, m) {
  m.def("mm_v1(Tensor a, Tensor b) -> Tensor", &mm_v1);
}

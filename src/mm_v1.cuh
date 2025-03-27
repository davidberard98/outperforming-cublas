#include "ATen/ATen.h"

namespace MMV1 {

template <int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K>
__global__ void mmv1Kernel(int N, int M, int K);

} // namespace MMV1

at::Tensor mm_v1(at::Tensor a, at::Tensor b) {
  return a.matmul(b);
  auto c = a.new_empty({a.size(0), b.size(1)});
  return c;
}

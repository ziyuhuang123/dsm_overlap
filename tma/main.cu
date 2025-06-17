#include "cutlass/util/command_line.h"

#include "scale_tma_kernel.h"
#include "tma_copy.h"
#include "tma_copy_dsm.h"
#include "tma_copy_multicast.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M = 16384, N = 16384, iterations = 1;
  int copy_mode = 0;                         // 0: G   1: G+G   2: G+DSM
                                             // 3: G+DSM(reg)  4: DSM
  cmd.get_cmd_line_argument("M",   M);
  cmd.get_cmd_line_argument("N",   N);
  cmd.get_cmd_line_argument("iterations", iterations);
  cmd.get_cmd_line_argument("copy_mode", copy_mode);

  std::cout << "(M,N)=(" << M << "," << N << "), copy_mode=" << copy_mode << '\n';

auto mode_str = [&]() -> const char* {
  switch (copy_mode) {
    case 0:  return "only globl";
    case 1:  return "G + G";
    case 2:  return "G + DSM";
    case 3:  return "G + DSM(reg)";
    case 4:  return "only DSM";
    default: return "INVALID";
  }
}();

std::cout << "Matrix (" << M << ", " << N << "),  mode " << copy_mode
          << "  [" << mode_str << "],  iterations = " << iterations << '\n';


  // copy_host_tma_load_and_store_kernel<128, 256>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_dsm<128, 128>(M, N, iterations);
  // scaleTmaKernelHost(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 2>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 2>(M, N, iterations);
  copy_host_tma_load_and_store_kernel_multicast<6, false, 2, 64, 256>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 4>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 4>(M, N, iterations);

  return 0;
}

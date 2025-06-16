#pragma once

#include "cutlass/detail/layout.hpp"

// template <class Element, class SmemLayout> struct SharedStorageTMA {
//   cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
//                       cutlass::detail::alignment_for_swizzle(SmemLayout{})>
//       smem;
//   // alignas(16) uint64_t tma_load_mbar[1];
//   cutlass::arch::ClusterTransactionBarrier mbarrier;
// };

template <class Element, class SmemLayout>
struct SharedStorageTMA {
  // 1. TMA 使用的缓冲区和屏障
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem; // sS 将指向这里
  cutlass::arch::ClusterTransactionBarrier mbarrier;

  // 2. DSM 使用的本地发送缓冲区
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem_dsm_local; // sS_dsm_local 将指向这里

  // 3. DSM 使用的本地接收缓冲区和屏障
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem_dsm_remote; // sS_dsm_remote 将指向这里
  cutlass::arch::ClusterTransactionBarrier mbarrier_dsm;
};
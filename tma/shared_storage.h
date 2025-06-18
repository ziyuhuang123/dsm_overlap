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

  cute::array_aligned<Element, cute::cosize_v<SmemLayout>, cutlass::detail::alignment_for_swizzle(SmemLayout{})> smem_D; // sS 将指向这里
  cutlass::arch::ClusterTransactionBarrier mbarrier;

template <int N_STAGES, class Element, class SmemLayout>
struct SharedStorageTMA {
  // 定义每个“阶段”包含的资源：一个屏障和一个缓冲区
  struct Stage {
    cutlass::arch::ClusterTransactionBarrier barrier;
    cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                        cutlass::detail::alignment_for_swizzle(SmemLayout{})> smem;
  };

  // 创建一个包含 N_STAGES 个“阶段”的数组
  cute::array<Stage, N_STAGES> stages;
};
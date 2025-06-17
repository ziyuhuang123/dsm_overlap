// #pragma once

// #include <cassert>
// #include <cstdio>
// #include <cstdlib>

// #include <chrono>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

// #include "cutlass/numeric_types.h"
// #include <cute/arch/cluster_sm90.hpp>
// #include <cute/tensor.hpp>
// #include <cutlass/arch/barrier.h>
// #include <cutlass/cluster_launch.hpp>
// #include <cutlass/cutlass.h>

// #include "cutlass/util/GPU_Clock.hpp"
// #include "cutlass/util/command_line.h"
// #include "cutlass/util/helper_cuda.hpp"
// #include "cutlass/util/print_error.hpp"
// #include <cute/arch/util.hpp>
// #include "cutlass/detail/layout.hpp"

// #include "cuda_launch.hpp"
// #include "shared_storage.h"
// #include "smem_helper.hpp"

// // template <typename _TiledCopyS, typename _TiledCopyD, typename _GmemLayout,
// //           typename _SmemLayout, typename _TileShape>
// // struct Params {
// //   using TiledCopyS = _TiledCopyS;
// //   using TiledCopyD = _TiledCopyD;
// //   using GmemLayout = _GmemLayout;
// //   using SmemLayout = _SmemLayout;
// //   using TileShape = _TileShape;

// //   TiledCopyS const tmaLoad;
// //   TiledCopyD const tmaStore;
// //   GmemLayout const gmemLayout;
// //   SmemLayout const smemLayout;
// //   TileShape const tileShape;

// //   Params(_TiledCopyS const &tmaLoad, _TiledCopyD const &tmaStore,
// //          _GmemLayout const &gmemLayout, _SmemLayout const &smemLayout,
// //          _TileShape const &tileShape)
// //       : tmaLoad(tmaLoad), tmaStore(tmaStore), gmemLayout(gmemLayout),
// //         smemLayout(smemLayout), tileShape(tileShape) {}
// // };


// template <typename BarrierT, typename SourceT, typename DstT>
// CUTLASS_DEVICE void copy_dsm(BarrierT& barrier,
//                             SourceT source,
//                             DstT dst,
//                             uint32_t size,
//                             uint32_t dst_block_rank) {

//   using namespace cute;                      

//   uint32_t bar_ptr;
//   if constexpr (std::is_integral_v<BarrierT>) {
//     bar_ptr = barrier;
//   }
//   else {
//     // bar_ptr = cast_smem_ptr_to_uint(cute::raw_pointer_cast(barrier.data()));
//     bar_ptr = cast_smem_ptr_to_uint(&barrier); // 这里和之前写法还稍有不同。包括函数定义的时候对barrier也要加上&
//   }

//   uint32_t src_addr;
//   if constexpr (std::is_integral_v<SourceT>) {
//     src_addr = source;
//   }
//   else {
//     src_addr = cast_smem_ptr_to_uint(cute::raw_pointer_cast(source.data()));
//   }

//   uint32_t dst_addr;
//   if constexpr (std::is_integral_v<DstT>) {
//     dst_addr = dst;
//   }
//   else {
//     dst_addr = cast_smem_ptr_to_uint(cute::raw_pointer_cast(dst.data()));
//   }

//   uint32_t neighbor_dst_addr;
//   // 利用 PTX 指令，将共享内存地址映射到目标 CTA 的共享内存地址
//   asm volatile (
//     "mapa.shared::cluster.u32 %0, %1, %2;\n"
//     : "=r"(neighbor_dst_addr)
//     : "r"(dst_addr), "r"(dst_block_rank)
//   );

//   // 发射 cp.async.bulk 指令，完成从 src_addr 到 neighbor_dst_addr 的数据复制
//   asm volatile (
//     "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
//     :
//     : "r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(bar_ptr)
//     : "memory"
//   );
//   asm volatile("cp.async.commit_group;" ::: "memory"); // 也许不需要这句？
// }


// template <int kNumThreads, class Element, class Params>
// __global__ static void __launch_bounds__(kNumThreads, 1)
//     copyTMAKernel_dsm(CUTE_GRID_CONSTANT Params const params) {
//   using namespace cute;

//   //
//   // Get layouts and tiled copies from Params struct
//   //
//   using GmemLayout = typename Params::GmemLayout;
//   using SmemLayout = typename Params::SmemLayout;
//   using TileShape = typename Params::TileShape;

//   auto &tmaLoad = params.tmaLoad;
//   auto &tmaStore = params.tmaStore;
//   auto &gmemLayout = params.gmemLayout;
//   auto &smemLayout = params.smemLayout;
//   auto &tileShape = params.tileShape;

//   // Use Shared Storage structure to allocate aligned SMEM addresses.
//   extern __shared__ char shared_memory[];
//   using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
//   SharedStorage &shared_storage =
//       *reinterpret_cast<SharedStorage *>(shared_memory);

//   // Define smem tensor
//   Tensor sS =
//       make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);
//   Tensor sS_dsm_local =
//       make_tensor(make_smem_ptr(shared_storage.smem_dsm_local.data()), smemLayout);
//   // 用于DSM接收
//   Tensor sS_dsm_remote =
//       make_tensor(make_smem_ptr(shared_storage.smem_dsm_remote.data()), smemLayout);


//   // Get mbarrier object and its value type
//   auto &mbarrier = shared_storage.mbarrier;

//   auto &mbarrier_dsm = shared_storage.mbarrier_dsm; // 现在 mbarrier_dsm 已经在此定义

//   using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
//   static_assert(cute::is_same_v<BarrierType, uint64_t>,
//                 "Value type of mbarrier is uint64_t.");

//   // Constants used for TMA
//   const int warp_idx = cutlass::canonical_warp_idx_sync();
//   const bool lane_predicate = cute::elect_one_sync();
//   constexpr int kTmaTransactionBytes =
//       sizeof(ArrayEngine<Element, size(SmemLayout{})>);

//   // Prefetch TMA descriptors for load and store
//   if (warp_idx == 0 && lane_predicate) {
//     prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
//     prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
//   }

//   // Get CTA view of gmem tensor
//   Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
//   auto blkCoord = make_coord(blockIdx.x, blockIdx.y);
//   Tensor gS = local_tile(mS, tileShape, blkCoord);

//   auto cta_tmaS = tmaLoad.get_slice(Int<0>{});

//   if (warp_idx == 0 and lane_predicate) {
//     mbarrier.init(1 /* arrive count */);
//     mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
//     copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
//          cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));

//     // 武装DSM屏障
//     mbarrier_dsm.init(1);
//     mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);

//     // 计算源CTA的ID（使用您的环形通信逻辑）
//     uint32_t src_block_rank = 
//         (cute::block_id_in_cluster().x + cute::cluster_shape().x - 1) % cute::cluster_shape().x;
//     copy_dsm(mbarrier_dsm, sS_dsm_local, sS_dsm_remote, kTmaTransactionBytes, src_block_rank);

//     asm volatile("cp.async.commit_group;" ::: "memory");
//   }
//   __syncthreads();

//   mbarrier.wait(0 /* phase */);
//   mbarrier_dsm.wait(0);
//   cutlass::arch::fence_view_async_shared();

//   // Get CTA view of gmem out tensor
//   auto mD = tmaStore.get_tma_tensor(shape(gmemLayout));
//   auto gD = local_tile(mD, tileShape, blkCoord);

//   auto cta_tmaD = tmaStore.get_slice(Int<0>{});

//   if (warp_idx == 0 and lane_predicate) {
//     cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
//     // cute::tma_store_arrive();
//   }
//   // cute::tma_store_wait<0>();
// }

// template <int TILE_M = 128, int TILE_N = 128, int THREADS = 32>
// int copy_host_tma_load_and_store_kernel_dsm(int M, int N, int iterations = 1) {
//   using namespace cute;

//   printf("Copy with TMA load and store -- no swizzling.\n");

//   using Element = cutlass::half_t;

//   auto tensor_shape = make_shape(M, N);

//   // Allocate and initialize
//   thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
//   thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

//   for (size_t i = 0; i < h_S.size(); ++i)
//     h_S[i] = static_cast<Element>(float(i));

//   thrust::device_vector<Element> d_S = h_S;
//   thrust::device_vector<Element> d_D = h_D;

//   //
//   // Make tensors
//   //

//   auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
//   auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
//   Tensor tensor_S = make_tensor(
//       make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
//   Tensor tensor_D = make_tensor(
//       make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

//   using bM = Int<TILE_M>;
//   using bN = Int<TILE_N>;

//   auto tileShape = make_shape(bM{}, bN{});
//   // NOTE: same smem layout for TMA load and store
//   auto smemLayout = make_layout(tileShape, LayoutRight{});
//   auto tma_load =
//       make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout);
//   // print(tma_load);

//   auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout);
//   // print(tma_store);

//   Params params(tma_load, tma_store, gmemLayoutS, smemLayout, tileShape);

//   dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
//   dim3 blockDim(THREADS);

//   int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
//   printf("smem size: %d.\n", smem_size);

//   void const *kernel =
//       (void const *)copyTMAKernel_dsm<THREADS, Element, decltype(params)>;
//   cfk::utils::set_smem_size(smem_size, kernel);

//   dim3 cluster_dims(2,1,1);
//   // dim3 cluster_dims(1);

//   // Define the cluster launch parameter structure.
//   cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
//                                              smem_size};

//   // for (int i = 0; i < iterations; i++) {
//   //   auto t1 = std::chrono::high_resolution_clock::now();    
//   //   cutlass::Status status =
//   //       cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
//   //   cudaError result = cudaDeviceSynchronize();
//   //   auto t2 = std::chrono::high_resolution_clock::now();
//   //   if (result != cudaSuccess) {
//   //     std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
//   //               << std::endl;
//   //     return -1;
//   //   }
//   //   std::chrono::duration<double, std::milli> tDiff = t2 - t1;
//   //   double time_ms = tDiff.count();
//   //   std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
//   //             << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s)"
//   //             << std::endl;
//   // }

//   // 用于累加总耗时的变量
//   double total_time_ms = 0.0;

//   for (int i = 0; i < iterations; i++) {
//       auto t1 = std::chrono::high_resolution_clock::now();
      
//       // 启动CUDA Kernel
//       cutlass::Status status =
//           cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
          
//       // 等待Kernel执行完毕
//       cudaError result = cudaDeviceSynchronize();
      
//       auto t2 = std::chrono::high_resolution_clock::now();

//       // 检查CUDA错误
//       if (result != cudaSuccess) {
//           std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
//                     << std::endl;
//           return -1;
//       }
      
//       // 计算单次运行时间并累加
//       std::chrono::duration<double, std::milli> tDiff = t2 - t1;
//       total_time_ms += tDiff.count();
//   }

//   // 确保迭代次数大于0，避免除零错误
//   if (iterations > 0) {
//       // 计算平均耗时
//       double average_time_ms = total_time_ms / iterations;
      
//       // 基于平均耗时计算平均性能（吞吐率）
//       double gb_per_s = 2e-6 * M * N * sizeof(Element) / average_time_ms;

//       // 打印最终的平均结果
//       std::cout << "Completed " << iterations << " trials." << std::endl;
//       std::cout << "Average time: " << average_time_ms << " ms" << std::endl;
//       std::cout << "Average throughput: " << gb_per_s << " GB/s" << std::endl;
//   }

//   //
//   // Verify
//   //

//   h_D = d_D;

//   int good = 0, bad = 0;

//   for (size_t i = 0; i < h_D.size(); ++i) {
//     if (h_D[i] == h_S[i])
//       good++;
//     else
//       bad++;
//   }

//   std::cout << "Success " << good << ", Fail " << bad << std::endl;

//   return 0;
// }

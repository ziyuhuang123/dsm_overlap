#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "cuda_launch.hpp"
#include "shared_storage.h"
#include "smem_helper.hpp"
#include <cooperative_groups.h>
template <typename _TiledCopyS, typename _TiledCopyD, typename _GmemLayout,
          typename _GmemLayoutOut, typename _SmemLayout, typename _TileShape,
          typename _ClusterShape>
struct ParamsMulticast {
  using TiledCopyS = _TiledCopyS;
  using TiledCopyD = _TiledCopyD;
  using GmemLayout = _GmemLayout;
  using GmemLayoutOut = _GmemLayoutOut;
  using SmemLayout = _SmemLayout;
  using TileShape = _TileShape;
  using ClusterShape = _ClusterShape;

  TiledCopyS const tmaLoad;
  TiledCopyD const tmaStore;
  GmemLayout const gmemLayout;
  GmemLayoutOut const gmemLayoutOut;
  SmemLayout const smemLayout;
  TileShape const tileShape;
  ClusterShape const cluster_shape;

  ParamsMulticast(_TiledCopyS const &tmaLoad, _TiledCopyD const &tmaStore,
                  _GmemLayout const &gmemLayout,
                  _GmemLayoutOut const &gmemLayoutOut,
                  _SmemLayout const &smemLayout, _TileShape const &tileShape,
                  _ClusterShape const &cluster_shape)
      : tmaLoad(tmaLoad), tmaStore(tmaStore), gmemLayout(gmemLayout),
        gmemLayoutOut(gmemLayoutOut), smemLayout(smemLayout),
        tileShape(tileShape), cluster_shape(cluster_shape) {}
};

template <int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernelMulticast(CUTE_GRID_CONSTANT Params const params) {
  // using namespace cute;

  // //
  // // Get layouts and tiled copies from Params struct
  // //
  // using GmemLayout = typename Params::GmemLayout;
  // using GmemLayoutOut = typename Params::GmemLayoutOut;
  // using SmemLayout = typename Params::SmemLayout;
  // using TileShape = typename Params::TileShape;
  // using ClusterShape = typename Params::ClusterShape;

  // auto &tmaLoad = params.tmaLoad;
  // auto &tmaStore = params.tmaStore;
  // auto &gmemLayout = params.gmemLayout;
  // auto &gmemLayoutOut = params.gmemLayoutOut;
  // auto &smemLayout = params.smemLayout;
  // auto &tileShape = params.tileShape;
  // auto &cluster_shape = params.cluster_shape;

  // uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  // constexpr uint32_t cluster_size = size(ClusterShape{});

  // uint16_t tma_mcast_mask = ((uint16_t(1) << cluster_size) - 1);

  // // Use Shared Storage structure to allocate aligned SMEM addresses.
  // extern __shared__ char shared_memory[];
  // using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
  // SharedStorage &shared_storage =
  //     *reinterpret_cast<SharedStorage *>(shared_memory);

  // // Define smem tensor
  // Tensor sS =
  //     make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);
  // Tensor sS_dsm_local =
  //     make_tensor(make_smem_ptr(shared_storage.smem_dsm_local.data()), smemLayout);
  // // 用于DSM接收
  // Tensor sS_dsm_remote =
  //     make_tensor(make_smem_ptr(shared_storage.smem_dsm_remote.data()), smemLayout);
  // auto &mbarrier_dsm = shared_storage.mbarrier_dsm; // 现在 mbarrier_dsm 已经在此定义


  // // Get mbarrier object and its value type
  // auto &mbarrier = shared_storage.mbarrier;
  // using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
  // static_assert(cute::is_same_v<BarrierType, uint64_t>,
  //               "Value type of mbarrier is uint64_t.");

  // // Constants used for TMA
  // const int warp_idx = cutlass::canonical_warp_idx_sync();
  // const bool lane_predicate = cute::elect_one_sync();
  // constexpr int kTmaTransactionBytes =
  //     sizeof(ArrayEngine<Element, size(SmemLayout{})>);

  // // Prefetch TMA descriptors for load and store
  // if (warp_idx == 0 && lane_predicate) {
  //   prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
  //   prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
  // }

  // // Get CTA view of gmem tensor
  // Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
  // auto blkCoord = make_coord(blockIdx.x, blockIdx.y);
  // Tensor gS = local_tile(mS, tileShape, blkCoord);

  // auto cta_tmaS = tmaLoad.get_slice(block_rank_in_cluster);
  // auto tSgSX = cta_tmaS.partition_S(gS);
  // auto tSgS = group_modes<1, rank(tSgSX)>(tSgSX);
  // auto tSsSX = cta_tmaS.partition_D(sS);
  // auto tSsS = group_modes<1, rank(tSsSX)>(tSsSX);

  // if (warp_idx == 0 and lane_predicate) {
  //   // for(int i=0; i<128*128;i++){
  //   //   sS_dsm_local[i]=blockIdx.z+1;
  //   // }
  //   mbarrier.init(1 /* arrive count */);
  //   mbarrier_dsm.init(1);
  // }
  // __syncthreads();
  // cute::cluster_sync();
  // cutlass::arch::fence_barrier_init();

  // if (warp_idx == 0 and lane_predicate) {
  //   mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
  //   copy(
  //       tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier), tma_mcast_mask),
  //       tSgS(_, 0), tSsS(_, 0));


        
  //   mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);

  //   // 计算源CTA的ID（使用您的环形通信逻辑）
  //   uint32_t src_block_rank = 
  //       (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
  //   // copy_dsm(mbarrier_dsm, sS_dsm_local, sS_dsm_remote, kTmaTransactionBytes, src_block_rank);
  //   copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier_dsm), tma_mcast_mask), tSgS(_, 0), tSsS(_, 0));

  // }
  // __syncthreads();

  // mbarrier.wait(0 /* phase */);
  // mbarrier_dsm.wait(0);
  // cutlass::arch::fence_view_async_shared();


  // if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0){
  //   print_tensor(sS_dsm_local);
  //   print_tensor(sS_dsm_remote);
  // }

  // // Get CTA view of gmem out tensor
  // auto blkCoordOut = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  // auto mD = tmaStore.get_tma_tensor(shape(gmemLayoutOut));
  // auto gD = local_tile(mD, tileShape, blkCoordOut);

  // auto cta_tmaD = tmaStore.get_slice(Int<0>{});

  // if (warp_idx == 0 and lane_predicate) {
  //   cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
  //   // cute::tma_store_arrive();
  // }
  // // cute::tma_store_wait<0>();
  // cute::cluster_sync();
}

template <int kConcurrentCopies, int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernelNoMulticast(CUTE_GRID_CONSTANT Params const params) {
  using namespace cute;

  //
  // Get layouts and tiled copies from Params struct
  //
  using GmemLayout = typename Params::GmemLayout;
  using GmemLayoutOut = typename Params::GmemLayoutOut;
  using SmemLayout = typename Params::SmemLayout;
  using TileShape = typename Params::TileShape;

  auto &tmaLoad = params.tmaLoad;
  auto &tmaStore = params.tmaStore;
  auto &gmemLayout = params.gmemLayout;
  auto &gmemLayoutOut = params.gmemLayoutOut;
  auto &smemLayout = params.smemLayout;
  auto &tileShape = params.tileShape;
  auto &cluster_shape = params.cluster_shape;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  // using SharedStorage = SharedStorageTMA<Element, SmemLayout>;

  // constexpr int CONCURRENT_COPIES = 7;

  // --- 2. 定义和获取流水线式共享内存 ---
  using SharedStorage = SharedStorageTMA<kConcurrentCopies, Element, SmemLayout>;

  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Define smem tensor
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);
  Tensor sS_dsm_local =
      make_tensor(make_smem_ptr(shared_storage.smem_dsm_local.data()), smemLayout);
  // 用于DSM接收
  Tensor sS_dsm_remote =
      make_tensor(make_smem_ptr(shared_storage.smem_dsm_remote.data()), smemLayout);


  uint32_t src_block_rank = (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  auto *dst_smem = cluster.map_shared_rank(shared_storage.smem_dsm_remote.data(), src_block_rank);
  Tensor sS_dsm_remote_dsm = make_tensor(make_smem_ptr(dst_smem), smemLayout);


  auto &mbarrier_dsm = shared_storage.mbarrier_dsm; // 现在 mbarrier_dsm 已经在此定义

  // Get mbarrier object and its value type
  auto &mbarrier = shared_storage.mbarrier;
  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
  static_assert(cute::is_same_v<BarrierType, uint64_t>,
                "Value type of mbarrier is uint64_t.");

  // Constants used for TMA
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const bool lane_predicate = cute::elect_one_sync();
  constexpr int kTmaTransactionBytes =
      sizeof(ArrayEngine<Element, size(SmemLayout{})>);

  // Prefetch TMA descriptors for load and store
  if (warp_idx == 0 && lane_predicate) {
    prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
    prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
  }

  // Get CTA view of gmem tensor
  Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
  auto blkCoord = make_coord(blockIdx.x, blockIdx.y);
  Tensor gS = local_tile(mS, tileShape, blkCoord);

  auto cta_tmaS = tmaLoad.get_slice(0);
  auto tSgSX = cta_tmaS.partition_S(gS);
  auto tSgS = group_modes<1, rank(tSgSX)>(tSgSX);
  // auto tSsSX = cta_tmaS.partition_D(sS);
  // auto tSsS = group_modes<1, rank(tSsSX)>(tSsSX);

  // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&blockIdx.z==0){

  //   print(gS); printf("  gS\n");
  //   print(tSgSX); printf("  tSgSX\n");
  //   print(tSgS); printf("  tSgS\n");
  //   print(tSgS(_, 0)); printf("  tSgS(_, 0)\n");
  //   print(tSsS(_, 0)); printf("  tSsS(_, 0)\n");

  // }

  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem_D.data()), smemLayout);
  auto mD = tmaStore.get_tma_tensor(shape(gmemLayout));
  auto gD = local_tile(mD, tileShape, blkCoord);
  auto cta_tmaD = tmaStore.get_slice(Int<0>{});
  auto tSgSX_d = cta_tmaD.partition_S(gD);
  auto tSgS_d = group_modes<1, rank(tSgSX_d)>(tSgSX_d);
  auto tSsSX_d = cta_tmaD.partition_D(sD);
  auto tSsS_d = group_modes<1, rank(tSsSX_d)>(tSsSX_d);


  
  
  if (warp_idx == 0 and lane_predicate) {
    #pragma unroll
    for (int i = 0; i < kConcurrentCopies; ++i) {
      shared_storage.stages[i].barrier.init(1);
    }
  }
  __syncthreads();
  cutlass::arch::fence_barrier_init();

  if (warp_idx == 0 and lane_predicate) {



    // 用一个循环，一次性发出 kConcurrentCopies 个独立的TMA加载指令
    #pragma unroll
    for (int i = 0; i < kConcurrentCopies; ++i) {
      // 获取当前阶段的屏障和SMEM缓冲区的引用/指针
      auto& stage_barrier = shared_storage.stages[i].barrier;
      Tensor stage_smem = make_tensor(make_smem_ptr(shared_storage.stages[i].smem.data()), smemLayout);
      
      // 核心修改：为每个拷贝任务计算不同的源数据瓦片
      // 我们从gS中，沿着M维度（行）切出第i个小瓦片
      auto coord_offset_i = make_coord(i * size<0>(TileShape{}), 0);
      Tensor gS_tile_i = local_tile(gS, tileShape, coord_offset_i);

      // 为第 i 个拷贝任务武装第 i 个屏障
      stage_barrier.arrive_and_expect_tx(kTmaTransactionBytes);

      // 发出第 i 个拷贝指令，关联第 i 个屏障
      copy(tmaLoad.with(reinterpret_cast<BarrierType&>(stage_barrier)), 
           tmaLoad.get_slice(0).partition_S(gS_tile_i),
           tmaLoad.get_slice(0).partition_D(stage_smem));
    }
    // 所有指令都发出后，用一个commit统一提交
    asm volatile("cp.async.commit_group;" ::: "memory");

    // mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
    // copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
    //      tSsS(_, 0));


bool need_wait_g   = false;
bool need_wait_dsm = false;

  uint32_t src_rank =
      (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) %
       cute::cluster_shape().z;

  switch (copy_mode) {

    case 0: {                           // 仅 global
      if (warp_idx == 0 and lane_predicate) {
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
            tSsS(_, 0));
      }
      need_wait_g = true;
      break;
    }

    case 1: {                           // global + global
      if (warp_idx == 0 and lane_predicate) {
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
            tSsS(_, 0));


        mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaStore.with(reinterpret_cast<BarrierType &>(mbarrier_dsm)), tSgS_d(_, 0), tSsS_d(_, 0));
        // copy(tmaStore.with(reinterpret_cast<BarrierType &>(mbarrier_dsm)), tSgS(_, 0), tSsS(_, 0));
        // copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier_dsm)), tSgS(_, 0),
        //     tSsS(_, 0));
        // copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier_dsm)), tSgS_d(_, 0), tSsS_d(_, 0));
      }
      need_wait_g = true;
      break;
    }

    case 2: {                           // global + DSM
      if (warp_idx == 0 and lane_predicate) {
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
            tSsS(_, 0));


        mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);
        // // 计算源CTA的ID（使用您的环形通信逻辑）
        uint32_t src_block_rank = 
            (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
        copy_dsm(mbarrier_dsm, sS_dsm_local, sS_dsm_remote_dsm, kTmaTransactionBytes, src_block_rank);
      }

      need_wait_g = need_wait_dsm = true;
      break;
    }

    case 3: {                           // global + DSM → reg
      if (warp_idx == 0 and lane_predicate) {
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
            tSsS(_, 0));
      }
      // 把 DSM 片段直接拉进寄存器
      // dsm2s_reg<kNumThreads>(sS_dsm_local, sS_dsm_remote);

      Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx
      Layout val_layout = make_layout(make_shape(Int<8>{}, Int<1>{}));   // (4,1) -> val_idx
      // using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;     // A very specific access width copy instruction
      // using Atom = Copy_Atom<CopyOp, Element>;
      using Atom = Copy_Atom<AutoVectorizingCopy, Element>;

      TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);
      ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
      Tensor thr_tile_S = thr_copy.partition_S(sS_dsm_local);             // (CopyOp, CopyM, CopyN)
      Tensor thr_tile_D = thr_copy.partition_D(sS_dsm_remote_dsm);
      Tensor fragment = make_fragment_like(thr_tile_D);
      copy(tiled_copy, thr_tile_S, fragment);
      copy(tiled_copy, fragment, thr_tile_D);


      need_wait_g = true;              // DSM-REG 路径不依赖 mbarrier
      break;
    }

    case 4: {                           // 仅dsm
      if (warp_idx == 0 and lane_predicate) {
        mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);
        // // 计算源CTA的ID（使用您的环形通信逻辑）
        uint32_t src_block_rank = 
            (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
        copy_dsm(mbarrier_dsm, sS_dsm_local, sS_dsm_remote_dsm, kTmaTransactionBytes, src_block_rank);
      }
      need_wait_dsm = true;
      break;
    }

} // CTA-leader 结束

__syncthreads();
cluster.sync();
if (need_wait_g)   mbarrier.wait(0);
if (need_wait_dsm) mbarrier_dsm.wait(0);


  // if (warp_idx == 0 and lane_predicate) {
  //   mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
  //   copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
  //        tSsS(_, 0));

  //   mbarrier_dsm.arrive_and_expect_tx(kTmaTransactionBytes);

  //   // // 计算源CTA的ID（使用您的环形通信逻辑）
  //   uint32_t src_block_rank = 
  //       (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
  //   copy_dsm(mbarrier_dsm, sS_dsm_local, sS_dsm_remote, kTmaTransactionBytes, src_block_rank);
  //   copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier_dsm)), tSgS(_, 0), tSsS(_, 0));
  // }

  // uint32_t src_block_rank = (cute::block_id_in_cluster().z + cute::cluster_shape().z - 1) % cute::cluster_shape().z;
  // namespace cg = cooperative_groups;
  // cg::cluster_group cluster = cg::this_cluster();
  // auto *dst_smem = cluster.map_shared_rank(shared_storage.smem_dsm_remote.data(), src_block_rank);
  // Tensor sS_dsm_remote_dsm = make_tensor(make_smem_ptr(dst_smem), smemLayout);

  // cooperative_copy<kNumThreads, 128>(threadIdx.x, sS_dsm_local, sS_dsm_remote, AutoCopyAsync{});
  // cooperative_copy<kNumThreads, 128>(threadIdx.x, sS_dsm_local, sS_dsm_remote, AutoVectorizingCopy{});

  // __syncthreads();

  #pragma unroll
  for (int i = 0; i < kConcurrentCopies; ++i) {
    shared_storage.stages[i].barrier.wait(0);
  }
  // mbarrier_dsm.wait(0 /* phase */);
  cutlass::arch::fence_view_async_shared();


  // // Get CTA view of gmem out tensor
  // auto blkCoordOut = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  // auto mD = tmaStore.get_tma_tensor(shape(gmemLayoutOut));
  // auto gD = local_tile(mD, tileShape, blkCoordOut);

  // auto cta_tmaD = tmaStore.get_slice(Int<0>{});

  // if (warp_idx == 0 and lane_predicate) {
  //   cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
  //   // cute::tma_store_arrive();
  // }
  // cute::tma_store_wait<0>();
}

template<bool use_multicast = true, int COPYN = 2, int TILE_M = 128,
          int TILE_N = 128, int THREADS = 256>
int copy_host_tma_load_and_store_kernel_multicast(int M, int N,
                                                  int iterations, int copy_mode) {
                                                  
  using namespace cute;


  std::cout << "Deep copy " << COPYN << "X." << std::endl;

  if constexpr (use_multicast)
    printf("Copy with TMA Multicast load and store.\n");
  else
    printf("Copy with TMA load and store, NO multicast.\n");

  // using Element = float;
  using Element = cutlass::half_t;

  using ClusterShape = Shape<_1, _1, Int<COPYN>>;
  ClusterShape cluster_shape;

  auto tensor_shape = make_shape(M, N);
  // auto tensor_shape_out = make_shape(M, N, Int<COPYN>{});
  auto tensor_shape_out = make_shape(M, N);

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape));     // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape_out)); // (M, N, COPYN)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(float(i));

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  // auto gmemLayoutD = make_ordered_layout(tensor_shape_out, Step<_1, _0, _2>{});
  auto gmemLayoutD = make_layout(tensor_shape_out, LayoutRight{});
  //   print(gmemLayoutD);

  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});
  // NOTE: same smem layout for TMA load and store
  auto smemLayout =
      tile_to_shape(cfx::getSmemLayoutK<Element, TILE_N>(), tileShape);
  auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, tensor_S, smemLayout,
                                tileShape, size(cluster_shape));
  auto tma_load_no_multicast =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout, tileShape, _1());

  // print(tma_load);
  // auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout, tileShape, Int<1>{});
  auto tma_store = make_tma_copy(SM90_TMA_LOAD{}, tensor_D, smemLayout, tileShape, Int<1>{});
  // print(tma_store);

  ParamsMulticast params(tma_load, tma_store, gmemLayoutS, gmemLayoutD,
                         smemLayout, tileShape, cluster_shape);
  ParamsMulticast params_no_multicast(tma_load_no_multicast, tma_store,
                                      gmemLayoutS, gmemLayoutD, smemLayout,
                                      tileShape, cluster_shape);

  dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N), COPYN);
  dim3 blockDim(THREADS);
  dim3 cluster_dims(size<0>(cluster_shape), size<1>(cluster_shape),
                    size<2>(cluster_shape));

  // int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
  int smem_size = 227*1024; // 这样一个block占一个SM
  printf("smem size: %d.\n", smem_size);

  void const *kernel;
  if constexpr (use_multicast)
    kernel = (void const *)
        copyTMAKernelMulticast<THREADS, Element, decltype(params)>;
  else
    kernel =
        (void const *)copyTMAKernelNoMulticast<CONCURRENT_COPIES, THREADS, Element,
                                               decltype(params_no_multicast)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};

  // for (int i = 0; i < iterations; i++) {
  //   auto t1 = std::chrono::high_resolution_clock::now();
  //   if constexpr (use_multicast)
  //     cutlass::Status status =
  //         cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
  //   else
  //     cutlass::Status status = cutlass::launch_kernel_on_cluster(
  //         launch_params, kernel, params_no_multicast);
  //   cudaError result = cudaDeviceSynchronize();
  //   auto t2 = std::chrono::high_resolution_clock::now();
  //   if (result != cudaSuccess) {
  //     std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
  //               << std::endl;
  //     return -1;
  //   }
  //   std::chrono::duration<double, std::milli> tDiff = t2 - t1;
  //   double time_ms = tDiff.count();
  //   std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
  //             << (COPYN + 1) * 1e-6 * M * N * sizeof(Element) / time_ms
  //             << " GB/s)" << std::endl;
  // }


  // 1. 在循环外定义一个变量，用于累加总耗时
  double total_time_ms = 0.0;

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // 2. 启动Kernel的逻辑保持不变，包括 if constexpr
    if constexpr (use_multicast) {
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
    }
    else {
      cutlass::Status status = cutlass::launch_kernel_on_cluster(
          launch_params, kernel, params_no_multicast);
    }
    
    // 等待Kernel执行完毕
    cudaError result = cudaDeviceSynchronize();
    
    auto t2 = std::chrono::high_resolution_clock::now();

    // 检查CUDA错误
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    
    // 3. 计算单次运行时间并累加，而不是直接打印
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    total_time_ms += tDiff.count();
  }

  // 4. 循环结束后，计算平均值并统一打印结果
  if (iterations > 0) {
    // 计算平均耗时
    double average_time_ms = total_time_ms / iterations;
    
    // 基于平均耗时计算平均性能（吞吐率）
    double gb_per_s = (COPYN + 1) * 1e-6 * M * N * sizeof(Element) / average_time_ms;

    // 打印最终的平均结果
    std::cout << "Completed " << iterations << " trials." << std::endl;
    std::cout << "Average time: " << average_time_ms << " ms" << std::endl;
    std::cout << "Average throughput: " << gb_per_s << " GB/s" << std::endl;
  }


  //
  // Verify
  //

  h_D = d_D;

  int good = 0, bad = 0;

  int offset = size(tensor_shape);

  for (size_t i = 0; i < h_S.size(); ++i) {
    for (int j = 0; j < COPYN; ++j) {
      if (h_D[i + j * offset] == h_S[i])
        good++;
      else
        bad++;
    }
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;

  return 0;
}

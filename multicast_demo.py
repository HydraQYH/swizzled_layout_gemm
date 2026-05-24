import argparse
from typing import Type, Union
import cuda.bindings.driver as cuda

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils

count = 128
bytes_per_element = 4

@cute.struct
class SharedStorage:
  barrier_storage: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
  smem_data: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, count], 16]

@cute.kernel
def copy_kernel(tensor: cute.Tensor):
  data_size_clutser = cute.size(tensor)
  bidx, _, _ = cute.arch.block_idx()
  tidx, _, _ = cute.arch.thread_idx()
  data_size_cta = data_size_clutser // 2

  cta_rank_in_cluster = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster()
  )
  cta_layout_mnk = cute.make_layout((2, 1, 1))
  cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

  mcast_mask = cute.make_layout_image_mask(
    cta_layout_mnk, cluster_coord_mnk, mode=0
  )

  smem = utils.SmemAllocator()
  storage = smem.allocate(SharedStorage)

  # Initialize barrier for TMA synchronization
  barrier_ptr = storage.barrier_storage.data_ptr()
  print("CuTeDSL DEBUG barrier_ptr", barrier_ptr)

  # Initialize the barrier
  with cute.arch.elect_one():
    cute.arch.mbarrier_init(barrier_ptr, 1)
    cute.arch.mbarrier_init_fence()
  cute.arch.sync_threads()

  # Cluster arrive after barrier init
  pipeline_init_arrive(cluster_shape_mn=(2, 1, 1), is_relaxed=True)

  # Signal arrival on the barrier after TMA is issued
  with cute.arch.elect_one():
    cute.arch.mbarrier_expect_tx(barrier_ptr, count * bytes_per_element)

  # Cluster wait for barrier init
  pipeline_init_wait(cluster_shape_mn=(2, 1, 1),)

  copy_bluk_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyBulkG2SMulticastOp(),
    cutlass.Int32,
    num_bits_per_copy=data_size_cta * 4 * 8
  )
  print("CuTeDSL DEBUG copy_bluk_atom", copy_bluk_atom)

  cluster_dst = cute.make_tensor(storage.smem_data.data_ptr(), cute.make_layout((data_size_clutser,)))
  cta_src = cute.local_tile(tensor, (data_size_cta,), (cta_rank_in_cluster,))
  cta_dst = cute.local_tile(cluster_dst, (data_size_cta,), (cta_rank_in_cluster,))
  print("CuTeDSL DEBUG cta_src", cta_src)
  print("CuTeDSL DEBUG cta_dst", cta_dst)

  with cute.arch.elect_one():
    cute.copy(
      copy_bluk_atom,
      cute.group_modes(cta_src, 0, cute.rank(cta_src)),
      cute.group_modes(cta_dst, 0, cute.rank(cta_dst)),
      mcast_mask=mcast_mask,
      mbar_ptr=barrier_ptr,
    )

  # Signal arrival on the barrier after TMA is issued
  with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(barrier_ptr)

  # Wait for TMA to complete
  cute.arch.mbarrier_wait(barrier_ptr, 0)

  with cute.arch.elect_one():
    if bidx == 1:
      cute.printf(f"bid {bidx} tid {tidx} mcast_mask {mcast_mask}, cluster_coord_mnk {cluster_coord_mnk}")
      cute.print_tensor(cluster_dst)

@cute.jit
def copy(
  tensor: cute.Tensor,
  stream: cuda.CUstream
):
  copy_kernel(tensor).launch(
    grid=(2, 1, 1),
    block=[32, 1, 1],
    cluster=(2, 1, 1),
    stream=stream,
  )


if __name__ == '__main__':
  tensor = torch.arange(count, dtype=torch.int32, device='cuda')
  cute_tensor = from_dlpack(tensor, assumed_align=16)

  torch_stream = torch.cuda.current_stream()
  stream = cuda.CUstream(torch_stream.cuda_stream)

  compiled_kernel = cute.compile(copy, cute_tensor, stream)
  compiled_kernel(cute_tensor, stream)


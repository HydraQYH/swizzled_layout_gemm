import argparse
from typing import Type, Union
import cuda.bindings.driver as cuda

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

class SwizzleCopyKernel:
  def __init__(self, block_m: int, block_n: int):
    self.block_m = block_m
    self.block_n = block_n
    self.buffer_align_bytes = 16
    self.mbarrier_align_bytes = 8

  @cute.jit
  def __call__(self, matrix: cute.Tensor, layout_enum: utils.LayoutEnum, swizzled_matrix: cute.Tensor, stream: cuda.CUstream,):
    self.dtype: Type[cutlass.Numeric] = matrix.element_type
    print("CuTeDSL DEBUG matrix", matrix)

    swizzle_mode = sm90_utils.get_smem_layout_atom(
      layout_enum,
      self.dtype,
      self.block_n if cutlass.const_expr(layout_enum.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K) else self.block_m
    )
    print("CuTeDSL DEBUG swizzle_mode", swizzle_mode)
    shared_memory_layout_atom = sm90_utils.make_smem_layout_atom(
      swizzle_mode,
      self.dtype
    )
    print("CuTeDSL DEBUG shared_memory_layout_atom", shared_memory_layout_atom)

    if cutlass.const_expr(layout_enum.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K):
      order=(0, 1)
    elif cutlass.const_expr(layout_enum.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.MN):
      order=(1, 0)
    self.shared_memory_layout = cute.tile_to_shape(shared_memory_layout_atom, (self.block_m, self.block_n), order=order)
    print("CuTeDSL DEBUG self.shared_memory_layout", self.shared_memory_layout)

    swizzled_matrix_layout = cute.tile_to_shape(self.shared_memory_layout, matrix.shape, order=order)
    print("CuTeDSL DEBUG swizzled_matrix_layout", swizzled_matrix_layout)

    # Use swizzled_matrix_tensor to avoid overriding the parameter name
    swizzled_matrix_tensor = cute.make_tensor(swizzled_matrix.iterator, swizzled_matrix_layout)
    print("CuTeDSL DEBUG swizzled_matrix_tensor", swizzled_matrix_tensor)

    grid_shape = (swizzled_matrix_tensor.shape[0][1], swizzled_matrix_tensor.shape[1][1], 1)
    print("CuTeDSL DEBUG grid_shape", grid_shape)

    tma_atom, tma_matirx = cpasync.make_tiled_tma_atom(
      cpasync.CopyBulkTensorTileG2SOp(), matrix, self.shared_memory_layout, (self.block_m, self.block_n)
    )
    print("CuTeDSL DEBUG tma_matirx", tma_matirx)

    @cute.struct
    class SharedStorage:
      smem_data: cute.struct.Align[
        cute.struct.MemRange[self.dtype, cute.cosize(self.shared_memory_layout)],
        self.buffer_align_bytes,
      ]
      barrier_storage: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1],
        self.mbarrier_align_bytes,
      ]
    self.shared_storage = SharedStorage
    self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, self.shared_memory_layout)
    print("CuTeDSL DEBUG self.num_tma_load_bytes", self.num_tma_load_bytes)

    # Launch the kernel synchronously
    self.swizzle_kernel(tma_atom, tma_matirx, self.shared_memory_layout, swizzled_matrix_tensor).launch(
      grid=grid_shape,
      block=[32, 1, 1],
      cluster=(1, 1, 1),
      stream=stream,
    )

  @cute.kernel
  def kernel(self, swizzled_matrix: cute.Tensor, smem_layout: Union[cute.Layout, cute.ComposedLayout],):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    # Allocate Shared Memory
    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # Initialize barrier for TMA synchronization
    barrier_ptr = storage.barrier_storage.data_ptr()
    print("CuTeDSL DEBUG barrier_ptr", barrier_ptr)

    # Initialize the barrier
    with cute.arch.elect_one():
      cute.arch.mbarrier_init(barrier_ptr, 1)
      cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # Signal arrival on the barrier after TMA is issued
    with cute.arch.elect_one():
      cute.arch.mbarrier_arrive_and_expect_tx(barrier_ptr, self.num_tma_load_bytes)

    target_tile = swizzled_matrix[((None, bidx), (None, bidy))]
    copy_bluk_atom = cute.make_copy_atom(
      cute.nvgpu.cpasync.CopyBulkG2SOp(),
      self.dtype,
      num_bits_per_copy=self.num_tma_load_bytes * 8
    )
    print("CuTeDSL DEBUG copy_bluk_atom", copy_bluk_atom)

    src = cute.make_tensor(
      target_tile.iterator,
      cute.make_layout(
        (cute.cosize(smem_layout),)
      )
    )

    dst = cute.make_tensor(
      storage.smem_data.data_ptr(),
      cute.make_layout(
        (cute.cosize(smem_layout),)
      )
    )

    with cute.arch.elect_one():
      cute.copy(
        copy_bluk_atom,
        cute.group_modes(src, 0, cute.rank(src)),
        cute.group_modes(dst, 0, cute.rank(dst)),
        mbar_ptr=barrier_ptr
      )

    # Wait for TMA to complete
    cute.arch.mbarrier_wait(barrier_ptr, 0)

  @cute.kernel
  def swizzle_kernel(
    self,
    tma_atom: cute.CopyAtom,
    tma_matrix: cute.Tensor,
    smem_layout: Union[cute.Layout, cute.ComposedLayout],
    swizzled_matrix: cute.Tensor,
  ):
    bidx, bidy, _ = cute.arch.block_idx()

    # Allocate Shared Memory
    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # Initialize barrier for TMA synchronization
    barrier_ptr = storage.barrier_storage.data_ptr()
    print("CuTeDSL DEBUG barrier_ptr", barrier_ptr)

    # Initialize the barrier: elect_one ensures only one thread executes this
    # Note: We must use elect_one() instead of "if tid == 0" because:
    # - elect_one() provides proper synchronization semantics
    # - It ensures all threads are aware that exactly one thread is executing
    # - It prevents race conditions and provides memory ordering guarantees
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier_ptr, 1)
        cute.arch.mbarrier_expect_tx(barrier_ptr, self.num_tma_load_bytes)

    # Fence ensures init/expect_tx are visible before proceeding
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # Tile the (M, N) tensor: ((TileM, TileN), M/TileM, N/TileN)
    gSrc_tiled = cute.local_tile(
      tma_matrix, (self.block_m, self.block_n), (bidx, bidy)
    )
    print("CuTeDSL DEBUG gSrc_tiled", gSrc_tiled)

    # TMA Load partition
    # Here we only use 1x1 cluster, so cta_id is 0 and cta_layout is (1).
    # More details about how to set cta_coord and cta_layout can be found in the tma_v4.py
    # Note: Smem and gemm should have the same size (atom element size) in the first rank
    smem_tensor = storage.smem_data.get_tensor(smem_layout)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_tensor, 0, 2),
        cute.group_modes(gSrc_tiled, 0, 2),
    )

    # ---------- TMA Load: Global -> Shared ----------
    cute.copy(
        tma_atom,
        tAgA,  # Source (TMA Tensor View)
        tAsA,  # Dest (SMEM Tensor View)
        tma_bar_ptr=barrier_ptr,
    )

    # Signal arrival on the barrier after TMA is issued
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier_ptr)

    # Wait for TMA to complete
    cute.arch.mbarrier_wait(barrier_ptr, 0)
    
    copy_bluk_atom = cute.make_copy_atom(
      cute.nvgpu.cpasync.CopyBulkS2GOp(),
      self.dtype,
      num_bits_per_copy=self.num_tma_load_bytes * 8
    )
    print("CuTeDSL DEBUG copy_bluk_atom", copy_bluk_atom)

    src = cute.make_tensor(
      storage.smem_data.data_ptr(),
      cute.make_layout(
        (cute.cosize(smem_layout),)
      )
    )

    target_tile = swizzled_matrix[((None, bidx), (None, bidy))]
    dst = cute.make_tensor(
      target_tile.iterator,
      cute.make_layout(
        (cute.cosize(smem_layout),)
      )
    )
    
    with cute.arch.elect_one():
      cute.copy(
        copy_bluk_atom,
        cute.group_modes(src, 0, cute.rank(src)),
        cute.group_modes(dst, 0, cute.rank(dst)),
      )
    
      cute.arch.cp_async_bulk_commit_group()
      cute.arch.cp_async_bulk_wait_group(0, read=True)


def run(
    M: int,
    N: int,
    k_major: bool
):
  matrix = torch.randn((M, N), dtype=torch.float16, device="cuda") if k_major else torch.randn((N, M), dtype=torch.float16, device="cuda").t()
  swizzled_matrix = torch.empty_like(matrix)
  cute_matrix = from_dlpack(matrix, assumed_align=16)
  cute_swizzled_matrix = from_dlpack(swizzled_matrix, assumed_align=16)
  copy_kernel = SwizzleCopyKernel(128, 64)
  major = utils.LayoutEnum.ROW_MAJOR if k_major else utils.LayoutEnum.COL_MAJOR
  
  torch_stream = torch.cuda.current_stream()
  stream = cuda.CUstream(torch_stream.cuda_stream)

  compiled_kernel = cute.compile(copy_kernel, cute_matrix, major, cute_swizzled_matrix, stream, options="--generate-line-info")
  compiled_kernel(cute_matrix, major, cute_swizzled_matrix, stream)

  print(matrix)
  print(swizzled_matrix)

  for _ in range(3):
    compiled_kernel(cute_matrix, major, cute_swizzled_matrix, stream)

  torch.cuda.nvtx.range_push('swizzle_copy')
  compiled_kernel(cute_matrix, major, cute_swizzled_matrix, stream)
  torch.cuda.nvtx.range_pop()

if __name__ == '__main__':
  run(128, 64, True)

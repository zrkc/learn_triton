###### submission begin ######

import triton
import triton.language as tl

@triton.jit
def my_matmul_trival(a_ptr, b_ptr, c_ptr,
              M: tl.constexpr,
              N: tl.constexpr,
              K: tl.constexpr,
              BLOCK_SIZE: tl.constexpr,
              BLOCK_K: tl.constexpr):
    grid_x = tl.program_id(axis=0)
    grid_y = tl.program_id(axis=1)
    row_offsets = grid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = grid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        a_offsets = row_offsets[:, None] * K + k_offsets[None, :]
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)

        b_offsets = col_offsets[:, None] * K + k_offsets[None, :]
        b_mask = (col_offsets[:, None] < N) & (k_offsets[None, :] < K)

        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        acc += tl.dot(a, tl.trans(b))

    c_offsets = row_offsets[:, None] * N + col_offsets[None, :]
    c_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(c_ptr + c_offsets, acc.to(tl.bfloat16), mask=c_mask)

@triton.jit
def my_matmul_grouped(a_ptr, b_ptr, c_ptr,
                M: tl.constexpr,
                N: tl.constexpr,
                K: tl.constexpr,
                BLOCK_SIZE: tl.constexpr,
                GROUP_SIZE: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr):
    """
    Intuition: Grouped ordering is a launch-order strategy.
    It tends to shorten the reuse distance of tiles in L2.
    Inside a group we enumerate in column-major order (n first, then m),
    so nearby programs are more likely to reuse recently loaded B tiles
    before they are evicted from cache.
    And overall we are indexing in row-major order outside the group, so the reuse distance of A tiles is also shortened.

    TODO: here we assume stride = (M, 1) for A and stride = (N, 1) for B, which is the case for our test but may not be true in general (e.g. if A and B are views of larger tensors).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE) # total number of pids (blocks) = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE * num_pid_n # in one group: GROUP_SIZE rows of blocks in A, all columns of blocks in B
    group_id = pid // num_pid_in_group # which group
    first_pid_m = group_id * GROUP_SIZE # the first row of blocks in A for this group
    this_group_num_pid_m = min(GROUP_SIZE, num_pid_m - first_pid_m) # the actual number of rows of blocks in A for this group

    # therefore, there are this_group_num_pid_m rows of blocks in A, and all num_pid_n columns of blocks in B for this group
    # within group, if we iterate row-wise first: for each row in A, we need to load *all* columns of blocks in B, which seems not " local"
    # instead, we iterate column-wise first: for each column in B, we only need to load *this_group_num_pid_m* rows of blocks in A
    # so within group, we index the blocks in column-major order.
    pid_m = first_pid_m + ((pid % num_pid_in_group) % this_group_num_pid_m) # which row of blocks in A for this pid
    pid_n = (pid % num_pid_in_group) // this_group_num_pid_m # which column in B

    # pointers for first A block and B block
    # we will advance this pointer as we move in the K direction and accumulate
    offsets_am = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # row offsets for first A block
    offsets_bn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offsets_am[:, None] * K + offsets_k[None, :])
    b_ptrs = b_ptr + (offsets_bn[:, None] * K + offsets_k[None, :])

    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offsets_am[:, None] < M) & (offsets_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offsets_bn[:, None] < N) & (offsets_k[None, :] < (K - k)), other=0.0)
        accumulator += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(tl.bfloat16)
    offsets_cm = offsets_am
    offsets_cn = offsets_bn
    c_ptrs = c_ptr + (offsets_cm[:, None] * N + offsets_cn[None, :])
    tl.store(c_ptrs, c, mask=(offsets_cm[:, None] < M) & (offsets_cn[None, :] < N))


def run_kernel(
    A,  # Tensor[bfloat16]
    B,  # Tensor[bfloat16]
    C,  # Tensor[bfloat16]
    M,  # int64
    N,  # int64
    K,  # int64
):
    
    # trivial implementation
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']), )
    # my_matmul_trival[grid](A, B, C, M, N, K, BLOCK_SIZE=128, BLOCK_K=64)

    # group implementation to hit cache
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']) * triton.cdiv(N, meta['BLOCK_SIZE']), ) # one-dimensional grid, total number of blocks of size BLOCK_SIZE x BLOCK_SIZE
    # A is split into cdiv(M, BLOCK_SIZE) rows of blocks of size (BLOCK_SIZE, K)
    my_matmul_grouped[grid](A, B, C, M, N, K, BLOCK_SIZE=64, GROUP_SIZE=8, BLOCK_SIZE_K=64)

###### submission end ######

import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(DEVICE)
torch.manual_seed(0)

M, N, K = 233, 666, 777
A = torch.randn((M, K), dtype=torch.bfloat16, device=DEVICE)
B = torch.randn((N, K), dtype=torch.bfloat16, device=DEVICE)
C = torch.zeros((M, N), dtype=torch.bfloat16, device=DEVICE)

run_kernel(A, B, C, M, N, K)

answer = torch.matmul(A, B.T)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(answer - C))}')
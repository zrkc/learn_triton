###### submission begin ######

import torch

import triton
import triton.language as tl

@triton.jit
def your_kernel(x_ptr,
                y_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x = x + y
    # Write x + y back to DRAM.
    tl.store(x_ptr + offsets, x, mask=mask)

def run_kernel(
    A,  # Tensor[fp16]
    B,  # Tensor[fp16]
    numel,  # int64
):
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']), )

    your_kernel[grid](A, B, numel, BLOCK_SIZE=1024)

###### submission end ######

DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(DEVICE)

torch.manual_seed(0)
size = (8192, 8192)
numel = size[0] * size[1]
A = torch.rand(size, device=DEVICE)
B = torch.rand(size, device=DEVICE)
answer = A.add(B)
run_kernel(A, B, numel)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(answer - A))}')
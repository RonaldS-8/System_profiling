import torch
import triton
import triton.language as tl


@triton.jit
def weighted_sum_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_m,
    x_stride_k,
    weight_stride_k,
    output_stride_m,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, BLOCK_K)

    weight_ptrs = weight_ptr + offset_k * weight_stride_k
    x_ptrs = x_ptr + offset_m[:, None] * x_stride_m + offset_k[None, :] * x_stride_k

    # 每个 Program 累加 BLOCK_M 个输出
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):

        row = tl.load(x_ptrs, mask=(offset_m[:, None] < M) & (offset_k[None, :] + k < K), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offset_k + k < K), other=0.0)

        acc += tl.sum(row * weight[None, :], axis=-1)

        x_ptrs += BLOCK_K * x_stride_k
        weight_ptrs += BLOCK_K * weight_stride_k

    output_ptrs = output_ptr + offset_m * output_stride_m
    tl.store(output_ptrs, acc, mask=(offset_m < M))

@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_m,
    x_stride_k,
    weight_stride_k,
    output_stride_m,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(M, K),
        strides=(x_stride_m, x_stride_k),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(K,),
        strides=(weight_stride_k,),
        offsets=(0,),
        block_shape=(BLOCK_K,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(M,),
        strides=(output_stride_m,),
        offsets=(pid_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        acc += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, BLOCK_K))
        weight_block_ptr = weight_block_ptr.advance((BLOCK_K,))

    tl.store(
        output_block_ptr,
        acc,
        boundary_check=(0,),
    )
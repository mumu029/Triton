import triton
import triton.language as tl
import torch
import torch.nn as nn
from typing import Any

import torch.cuda.nvtx as nvtx

# 定义矩阵乘法内核
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    矩阵乘法 Triton 内核: C = A x B
    A: (M, K), B: (K, N), C: (M, N)
    """
    # 查找此程序实例的处理块
    pid = tl.program_id(0)
    

    # 沿 M 维度对程序实例进行分组
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算用于加载 A 和 B 的指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 循环 K 维度
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 将结果写回 C
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)

def matmul(a :torch.Tensor, b :torch.Tensor):
    # 检查输入张量的维度
    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    M, K = a.shape
    K, N = b.shape

    # 确保张量在 GPU 上
    a = a.cuda()
    b = b.cuda()
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # 计算网格大小
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, # 可以根据 GPU 和矩阵大小调整这些参数
        GROUP_SIZE_M=8,
        num_warps=4, # 可以根据 GPU 调整
        num_stages=3, # 可以根据 GPU 调整
    )
    return c

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        # 可以为不同的 BLOCK_SIZE 组合尝试不同的 num_stages 和 num_warps
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
    ],
    # 也可以定义一些条件，例如根据 M, N, K 的大小来筛选适用的配置
    key=['M', 'N', 'K'], # 指定哪些参数用于缓存调优结果
)
@triton.jit
def matmul_bmm_transposed_b_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    B, M, N, K,
    # Strides for moving through matrices
    stride_ab, stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    # Meta-parameters for tuning
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Kernel for computing C = A @ B.T where A is (B, M, K) and B is (N, K)
    This kernel is launched in a 2D grid of programs, where each program computes a BxMxN block of C.
    """
    pid_b = tl.program_id(axis=2)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A
    a_ptrs = a_ptr + (pid_b * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    b_ptrs = b_ptr + (offs_an[None, :] * stride_bn + offs_k[:, None] * stride_bk)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, handling leftovers in K gracefully.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

        # Advance the pointers for the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optional: apply activation function
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cb * pid_b + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (pid_b < B) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_bmm(a, b, activation=""):
    """
    Compute C = A @ B.T
    A is a 3D tensor (B, M, K)
    B is a 2D tensor (N, K)
    C will be a 3D tensor (B, M, N)
    """
    # Check constraints.
    assert a.shape[2] == b.shape[1], "Incompatible dimensions K"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    B, M, K = a.shape
    N, _ = b.shape
    
    # Allocates output.
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    if M == 0:
        pass
    else:

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            1,
            B
        )

        # -----------------------------------------------------------
        # Kernel Launch
        matmul_bmm_transposed_b_kernel[grid](
            a, b, c,
            B, M, N, K,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), c.stride(2)
        )
    return c

class ZZULinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: Any | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = bias
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, device= device, dtype=dtype))
        nn.init.kaiming_normal_(self.weight,mode="fan_in",nonlinearity="leaky_relu")
        if self.need_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        x = matmul_bmm(x, self.weight)
        if self.bias != None:
            x += self.bias
        return x

def eval_rumtime(func, x, times, message, warm_up = 20):
    # for _ in range(warm_up):
    #     func(x)
    torch.cuda.synchronize()
    start_event_linear = torch.cuda.Event(enable_timing=True)
    end_event_linear = torch.cuda.Event(enable_timing=True)
    start_event_linear.record()
    for _ in range(times):
        _ = func(x)
    end_event_linear.record()
    torch.cuda.synchronize()
    time_linear_ms = start_event_linear.elapsed_time(end_event_linear)
    print(f"{message} 平均执行时间: {time_linear_ms / times:.6f} ms/次\n")
    return x.shape[1] / (time_linear_ms / times)



if __name__ == "__main__":
    pass
    # runtime = [[], [], []]
    # test_count = 499
    # for i in range(499, test_count + 1):
    #     B, M, K, N = 8, i, 7298, 2560
    #     a = torch.randint(1, 10, (B, M, K), device='cuda').to(torch.float32)
    #     # b = torch.randint(1, 10, (N, K), device='cuda').to(torch.float32)
    #     times = 1

    #     with nvtx.range("ZZULinear"):
    #         model_triton = ZZULinear(K, N).to("cuda")
    #         runtime[0].append(eval_rumtime(model_triton, a, times, "ZZULinear"))


    #     with nvtx.range("Linear"):
    #         model_torch = nn.Linear(K, N, bias=True).to("cuda")
    #         runtime[1].append(eval_rumtime(model_torch, a, times, "Eager"))

    #     with nvtx.range("Compile"):
    #         model_compile = torch.compile(model_torch)
    #         runtime[2].append(eval_rumtime(model_compile, a, times, "Compile"))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 6), dpi=300)
    # plt.plot(range(1, test_count + 1), runtime[0], label='ZZULinear', marker='o')
    # plt.plot(range(1, test_count + 1), runtime[1], label='Eager Linear', marker='x')
    # plt.plot(range(1, test_count + 1), runtime[2], label='Compile Linear', marker='s')
    # plt.xlabel('M (High of Input Features)')
    # plt.ylabel('Average Speed (ms / M)')
    # plt.title('Performance Comparison of ZZULinear vs Eager Linear vs Compile Linear')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('performance_comparison.png')
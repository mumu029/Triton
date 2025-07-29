import triton.language as tl
import triton
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from ZZULinear import ZZULinear
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

def ACT_Multi_Matmul(act_fn, x : torch.Tensor, weight:torch.Tensor):
    assert act_fn != None
    assert x.shape[2] // 2 == weight.shape[1]

    B, M, K = x.shape
    N, _ = weight.shape
    c = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * B,
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )
    
    # Launch the kernel
    ACT_Multi_Matmul_kernel[grid](
        x, weight, c,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), 
        c.stride(0), c.stride(1), c.stride(2),
        # 注意：这里不再有 BLOCK_SIZE 和 GROUP_SIZE 参数
        act_fn=act_fn
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
        ],
    # 也可以定义一些条件，例如根据 M, N, K 的大小来筛选适用的配置
    key=['M', 'N', 'K'], # 指定哪些参数用于缓存调优结果
)
@triton.jit
def ACT_Multi_Matmul_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, c_ptr,
    # Matrix dimensions
    B, M, N, K,
    # Strides for X tensor (B, M, K)
    stride_xb, stride_xm, stride_xk,
    # Strides for Weight tensor (N, K/2)
    stride_wn, stride_wk,
    # Strides for C tensor (B, M, N)
    stride_cb, stride_cm, stride_cn,
    # Meta-parameters for tuning
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Activation function
    act_fn: tl.constexpr
):
    """
    Kernel for computing C = dot(act_fn(x[..., :K//2]) * x[..., K//2:], weight.T)
    where x is (B, M, K) and weight is (N, K//2), producing C of shape (B, M, N).
    
    This kernel is launched in a 2D grid for performance:
    - Grid Axis 0: B * cdiv(M, BLOCK_SIZE_M) -> Handles batch and M dimensions
    - Grid Axis 1: cdiv(N, BLOCK_SIZE_N)     -> Handles N dimension
    """
    # -----------------------------------------------------------
    # 1. Map program IDs to batch, M, and N blocks
    # -----------------------------------------------------------
    pid_bm = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    num_m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    pid_b = pid_bm // num_m_blocks
    pid_m_raw = pid_bm % num_m_blocks

    # Group M-blocks for better L2 cache locality
    group_id = pid_m_raw // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_m_blocks - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m_raw % group_size)

    # -----------------------------------------------------------
    # 2. Compute pointers and offsets
    # -----------------------------------------------------------
    K_half = K // 2
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for input x (split into two halves)
    x_base_ptr = x_ptr + pid_b * stride_xb + offs_m[:, None] * stride_xm
    x1_ptrs = x_base_ptr + offs_k[None, :] * stride_xk
    x2_ptrs = x_base_ptr + (offs_k[None, :] + K_half) * stride_xk

    # Pointers for weight. The layout handles the transpose operation `weight.T`.
    # We load a (BLOCK_SIZE_K, BLOCK_SIZE_N) tile.
    # offs_k accesses the K-dim (in_features) and offs_n accesses the N-dim (out_features).
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # -----------------------------------------------------------
    # 3. Main loop over the K_half dimension
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    for k in range(0, K_half, BLOCK_SIZE_K):
        # Boundary masks
        k_mask = (k + offs_k) < K_half
        
        
        # Load and compute the activated `A` matrix block
        x1 = tl.load(x1_ptrs + k * stride_xk, mask=m_mask & k_mask[None, :], other=0.0)
        x2 = tl.load(x2_ptrs + k * stride_xk, mask=m_mask & k_mask[None, :], other=0.0)
        

        # Load the weight matrix block
        
        w = tl.load(weight_ptrs + k * stride_wk, mask=k_mask[:, None] & n_mask, other=0.0)
        
        # Matrix multiply and accumulate
        
        accumulator += tl.dot(act_fn(x1) * x2, w)
        
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # 4. Write the result to the output tensor C
    # -----------------------------------------------------------
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, c, mask=c_mask)

class ZZUSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _silu

    def forward(self, x):
        
        x = self.gate_up_proj(x)
        x = ACT_Multi_Matmul(self.act_fn, x, self.down_proj.weight)
        return x

    def tool(self):
        self.gate_up_proj = ZZULinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.gate_up_proj.weight = nn.Parameter(torch.cat([self.gate_proj.weight, self.up_proj.weight]))
    
@triton.jit    
def _silu(x):
    return x * tl.sigmoid(x)

def eval_runtime(model, x, times, message="Model forward pass"):
    for _ in range(20):  # Warm-up
        output = model(x)
    torch.cuda.synchronize()
    start_event_linear = torch.cuda.Event(enable_timing=True)
    end_event_linear = torch.cuda.Event(enable_timing=True)
    start_event_linear.record()
    with nvtx.range(message):
        for _ in range(times):
            model(x)
    end_event_linear.record()
    torch.cuda.synchronize()
    time_linear_ms = start_event_linear.elapsed_time(end_event_linear)
    print(f"{message} 平均执行时间: {time_linear_ms / times:.6f} ms/次\n")
    return x.shape[1] / (time_linear_ms / times)

class Qwen3MLP_big_linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ZZULinear import ZZULinear
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        with nvtx.range("gate_up_proj"):
            x = self.gate_proj(x)
            x1 = x[...,:self.intermediate_size]
            x2 = x[...,self.intermediate_size:]
        with nvtx.range("down_proj"):

            down_proj = self.down_proj(self.act_fn(x1) * x2)
        return down_proj

class Qwen3MLP_big_linear_triton(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ZZULinear import ZZULinear
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = ZZULinear(self.hidden_size, self.intermediate_size * 2)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        with nvtx.range("gate_up_proj"):
            x = self.gate_proj(x)
            x1 = x[...,:self.intermediate_size]
            x2 = x[...,self.intermediate_size:]
        with nvtx.range("down_proj"):

            down_proj = self.down_proj(self.act_fn(x1) * x2)
        return down_proj
    
class Qwen3MLP_op_fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ZZULinear import ZZULinear
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _silu

    def forward(self, x):
        with nvtx.range("gate_proj"):
            x1 = self.gate_proj(x)
        with nvtx.range("up_proj"):
            x2 = self.up_proj(x)
        x = torch.cat([x1, x2], dim=-1)
        with nvtx.range("down_proj"):
            down_proj = ACT_Multi_Matmul(self.act_fn, x, self.down_proj.weight)
        return down_proj

class Qwen3MLP_big_linear_op_fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ZZULinear import ZZULinear
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _silu

    def forward(self, x):
        with nvtx.range("gate_up_proj"):
            x = self.gate_proj(x)
        with nvtx.range("down_proj"):
            down_proj = ACT_Multi_Matmul(self.act_fn, x, self.down_proj.weight)
        return down_proj

class Config:
    def __init__(self):
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.hidden_act = "silu"
        self.dtype = torch.float32

def main():
    config = Config()
    times = 200; seq_len = 125
    naive = Qwen3MLP(config).to("cuda")
    big_linear = Qwen3MLP_big_linear(config).to("cuda")
    big_linear_triton = Qwen3MLP_big_linear_triton(config).to("cuda")
    op_fusion = Qwen3MLP_op_fusion(config).to('cuda')
    big_linear_op_fusion = Qwen3MLP_big_linear_op_fusion(config).to("cuda")
    demo = ZZUSwiGLU(config).to("cuda:0");demo.tool()
    compile_naive = torch.compile(naive)

    result_time = [[],[], [], [], [],[], []]
    for i in range(1, seq_len + 1):
        print(f"Testing sequence length: {i}")
        x = torch.randn((8, i, config.hidden_size), device="cuda", dtype=config.dtype)
        # x = torch.randint(1,10, (8, 128, config.hidden_size), device="cuda").to(config.dtype)

        with nvtx.range("Naive MLP"):
            naive_speed = eval_runtime(naive, x, times, "Naive MLP")
            result_time[0].append(naive_speed)
        with nvtx.range("Big Linear MLP"):
            big_linear_speed = eval_runtime(big_linear, x, times, "Big Linear MLP")
            result_time[1].append(big_linear_speed)
        with nvtx.range("Big Linear triton MLP"):
            big_linear_triton_speed = eval_runtime(big_linear_triton, x, times, "Big Linear triton MLP")
            result_time[2].append(big_linear_triton_speed)
        # with nvtx.range("Op Fusion MLP"):
        #     op_fusion_speed = eval_runtime(op_fusion, x, times, "Op Fusion MLP")
        #     result_time[3].append(op_fusion_speed)
        with nvtx.range("Big Linear Op Fusion MLP"):
            big_linear_op_fusion_speed = eval_runtime(big_linear_op_fusion, x, times, "Big Linear Op Fusion MLP")
            result_time[4].append(big_linear_op_fusion_speed)
        with nvtx.range("ZZUSwiGLU"):
            demo_speed = eval_runtime(demo, x, times, "ZZUSwiGLU")
            result_time[5].append(demo_speed)
        with nvtx.range("compile Naive MLP"):
            compile_naive_speed = eval_runtime(compile_naive, x, times, "compile Naive MLP")
            result_time[6].append(compile_naive_speed)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6),dpi=200)
    plt.plot(range(1, seq_len + 1), result_time[0], label='Naive MLP', marker='o')
    plt.plot(range(1, seq_len + 1), result_time[6], label='Compile Naive MLP', marker='o')
    plt.plot(range(1, seq_len + 1), result_time[1], label='Big Linear MLP', marker='o')
    plt.plot(range(1, seq_len + 1), result_time[2], label='Big Linear triton MLP', marker='o')
    # plt.plot(range(1, seq_len + 1), result_time[3], label='Op Fusion MLP', marker='o')
    plt.plot(range(1, seq_len + 1), result_time[4], label='Big Linear Op Fusion MLP', marker='o')
    plt.plot(range(1, seq_len + 1), result_time[5], label='ZZUSwiGLU', marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speed (tokens/ms)')
    plt.title('MLP Speed Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("mlp_speed_comparison.png")

   

if __name__ == "__main__":
    main()
    

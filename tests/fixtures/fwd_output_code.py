# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.bmm

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/vn/cvn7syoy6ts72prseykobuvj7yiz75x6l4akupm4r4cathivnqwt.py
# Topologically Sorted Source Nodes: [inputs_t], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   inputs_t => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton(
    "triton_poi_fused_clone_0",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/uz/cuzcs76gl437efoes5lijhkiy5tzppipgbqhardbfconcr3lxbk3.py
# Topologically Sorted Source Nodes: [inputs_t, gates], Original ATen: [aten.clone, aten.bmm]
# Source node to ATen node mapping:
#   gates => bmm
#   inputs_t => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %bmm : [num_users=2] = call_function[target=torch.ops.aten.bmm.default](args = (%clone, %primals_2), kwargs = {})
triton_tem_fused_bmm_clone_1 = async_compile.triton(
    "triton_tem_fused_bmm_clone_1",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_clone_1', 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
)
@triton.jit
def triton_tem_fused_bmm_clone_1(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 5000
    N = 1536
    K = 512

    stride_aq = 512
    stride_am = 16384
    stride_ak = 1

    stride_bq = 786432
    stride_bk = 1536
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1536*idx_m + 7680000*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
""",
    device_str="cuda",
)
meta0 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": True,
    "ACC_TYPE": "tl.float32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
}


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/od/codyo7esvlyme5nhmpezopjgrpnkiultq4tx32r44zvfaaj7lc4s.py
# Topologically Sorted Source Nodes: [gates_1, gates_2, vals_1, outputs], Original ATen: [aten.add, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   gates_1 => add
#   gates_2 => mul, sigmoid
#   outputs => mul_1
#   vals_1 => add_1
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm, %unsqueeze), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %sigmoid), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_1, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add_1), kwargs = {})
triton_poi_fused_add_mul_silu_2 = async_compile.triton(
    "triton_poi_fused_add_mul_silu_2",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_silu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1536)
    x2 = xindex // 7680000
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 1536*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x0 + 1536*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/lq/clq4adyix33wlnqqlwdoftu4tpznhn2etyur5azgy2un4j3pw62o.py
# Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   outputs_1 => bmm_2
# Graph fragment:
#   %bmm_2 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%mul_1, %primals_6), kwargs = {})
triton_tem_fused_bmm_3 = async_compile.triton(
    "triton_tem_fused_bmm_3",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_3', 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
)
@triton.jit
def triton_tem_fused_bmm_3(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 5000
    N = 512
    K = 1536

    stride_aq = 7680000
    stride_am = 1536
    stride_ak = 1

    stride_bq = 786432
    stride_bk = 512
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m + 2560000*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/k4/ck4ue7jbymrot33gosytvwibgqii33norxvs4a33jrgenzsdc5zc.py
# Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   outputs_3 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton(
    "triton_poi_fused_clone_4",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 32)
    x2 = xindex // 16384
    x3 = (xindex % 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 2560000*x1), None)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_hadoop-scale-pmf/sq/csqtru4ageunwbeew22n5czrmkbpibznhfokbuocnx67xeljgvke.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [0, 2, 1]), kwargs = {})
triton_poi_fused_transpose_5 = async_compile.triton(
    "triton_poi_fused_transpose_5",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_transpose_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '99EA89A0A9D95986274BE7B6C65D57A1F61568D41D73439B0A728D361AAE6E8F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_transpose_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 5000)
    x2 = xindex // 2560000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 16384*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
""",
    device_str="cuda",
)


async_compile.wait(globals())
del async_compile


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (5000, 32, 512), (16384, 512, 1))
    assert_size_stride(primals_2, (32, 512, 1536), (786432, 1536, 1))
    assert_size_stride(primals_3, (32, 1536), (1536, 1))
    assert_size_stride(primals_4, (32, 512, 1536), (786432, 1536, 1))
    assert_size_stride(primals_5, (32, 1536), (1536, 1))
    assert_size_stride(primals_6, (32, 1536, 512), (786432, 512, 1))
    assert_size_stride(primals_7, (32, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 5000, 512), (512, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_t], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(primals_1, buf0, 81920000, grid=grid(81920000), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((32, 5000, 1536), (7680000, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_t, gates], Original ATen: [aten.clone, aten.bmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_bmm_clone_1.run(
            buf0, primals_2, buf1, grid=torch._inductor.kernel.bmm.bmm_grid(32, 5000, 1536, meta0), stream=stream0
        )
        buf2 = empty_strided_cuda((32, 5000, 1536), (7680000, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [vals], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_bmm_clone_1.run(
            buf0, primals_4, buf2, grid=torch._inductor.kernel.bmm.bmm_grid(32, 5000, 1536, meta0), stream=stream0
        )
        buf3 = empty_strided_cuda((32, 5000, 1536), (7680000, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gates_1, gates_2, vals_1, outputs], Original ATen: [aten.add, aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_silu_2.run(
            buf1, primals_3, buf2, primals_5, buf3, 245760000, grid=grid(245760000), stream=stream0
        )
        buf4 = empty_strided_cuda((32, 5000, 512), (2560000, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_bmm_3.run(
            buf3, primals_6, buf4, grid=torch._inductor.kernel.bmm.bmm_grid(32, 5000, 512, meta0), stream=stream0
        )
        buf5 = empty_strided_cuda((5000, 32, 512), (16384, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf4, primals_7, buf5, 81920000, grid=grid(81920000), stream=stream0)
        del buf4
        del primals_7
        buf6 = empty_strided_cuda((32, 512, 5000), (2560000, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_transpose_5.run(buf0, buf6, 81920000, grid=grid(81920000), stream=stream0)
        del buf0
    return (
        buf5,
        primals_3,
        primals_5,
        buf1,
        buf2,
        reinterpret_tensor(buf3, (32, 1536, 5000), (7680000, 1, 1536), 0),
        reinterpret_tensor(primals_6, (32, 512, 1536), (786432, 1, 512), 0),
        buf6,
        reinterpret_tensor(primals_4, (32, 1536, 512), (786432, 1, 1536), 0),
        reinterpret_tensor(primals_2, (32, 1536, 512), (786432, 1, 1536), 0),
    )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided((5000, 32, 512), (16384, 512, 1), device="cuda:0", dtype=torch.float32)
    primals_2 = rand_strided((32, 512, 1536), (786432, 1536, 1), device="cuda:0", dtype=torch.float32)
    primals_3 = rand_strided((32, 1536), (1536, 1), device="cuda:0", dtype=torch.float32)
    primals_4 = rand_strided((32, 512, 1536), (786432, 1536, 1), device="cuda:0", dtype=torch.float32)
    primals_5 = rand_strided((32, 1536), (1536, 1), device="cuda:0", dtype=torch.float32)
    primals_6 = rand_strided((32, 1536, 512), (786432, 512, 1), device="cuda:0", dtype=torch.float32)
    primals_7 = rand_strided((32, 512), (512, 1), device="cuda:0", dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)

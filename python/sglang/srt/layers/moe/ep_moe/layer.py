from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from sglang.srt.batch_overlap.per_expert_overlap import PeoOverlapArgs

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    is_peo_enabled,
    get_peo_overlap_method,
    get_peo_num_rounds,
    get_peo_deepep_num_sms,
    get_peo_up_deepgemm_num_sms,
    get_peo_down_deepgemm_num_sms,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
elif _is_npu:
    import torch_npu


logger = logging.getLogger(__name__)


if _is_npu:
    import torch_npu


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False

        self.deepep_mode = get_deepep_mode()

        if (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            )
        ):
            # NPU supports low_latency deepep without deepgemm
            # FP4 quantization with flashinfer_cutedsl also supports low_latency deepep without deepgemm
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        if is_in_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return torch.ops.sglang.moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        # TODO: can we call super().forward here?
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert (
            envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
                if not input_quant and not isinstance(
                    self.quant_method, NPUW4A16Int4DynamicMoEMethod
                ):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")

        return hidden_states


class NpuFuseEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        self.quant_method.process_weights_after_loading = (
            self._process_weights_after_loading
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
            gmm1_permuted_weight=self.w13_weight,
            gmm1_permuted_weight_scale=self.w13_weight_scale,
            gmm2_weight=self.w2_weight,
            gmm2_weight_scale=self.w2_weight_scale,
        ).hidden_state

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def permute_w13_weight_scale(self, w: torch.Tensor, tile_n: int):
        if tile_n % 2 != 0:
            raise ValueError(f"tile_n must be even, got {tile_n}")

        *dims, n = w.shape
        if n % tile_n != 0:
            raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

        w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)

        # Permute the last two dimensions.
        perm_order = list(range(len(dims))) + [-2, -3, -1]
        w_permuted = w_reshaped.permute(perm_order)

        return w_permuted.reshape(*dims, n)

    def reshape_w13_weight(self, weight: torch.Tensor, dim: int, chunk_size: int = 64):
        # Achieving greater computing power through reshape on Ascend.
        original_shape = weight.shape
        if dim < 0:
            dim += len(original_shape)

        if original_shape[dim] % (2 * chunk_size) != 0:
            raise ValueError(
                f"Dimension {dim} size {original_shape[dim]} must be divisible by {2 * chunk_size}"
            )

        new_shape = (
            *original_shape[:dim],
            2,
            original_shape[dim] // (2 * chunk_size),
            chunk_size,
            *original_shape[dim + 1 :],
        )

        weight = weight.view(new_shape)
        weight = weight.transpose(dim, dim + 1).contiguous()

        return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13 = self.release_weight_cache(layer.w13_weight)
        torch_npu.npu_format_cast_(w13, 2)
        cpu_w13 = w13.cpu()
        w13 = self.reshape_w13_weight(cpu_w13, -1).npu()
        torch_npu.npu_format_cast_(w13, 29)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)

        w2 = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
        w13_scale = self.permute_w13_weight_scale(w13_scale, 128)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale.to(torch.float32), requires_grad=False
        )

        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )

        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )


def get_num_device_sms():
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count

class PeoDeepEPMoE(DeepEPMoE):
    gemm_stream = torch.cuda.Stream()

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(num_experts, top_k, hidden_size, intermediate_size,
                         layer_id, num_fused_shared_experts, params_dtype, quant_config, prefix, activation,
                         routed_scaling_factor, **kwargs)
        self.overlap_method = get_peo_overlap_method()
        self.num_rounds = get_peo_num_rounds()
        self.num_device_sms = get_num_device_sms()
        self.num_deepep_sms = get_peo_deepep_num_sms()
        self.num_up_deepgemm_sms = get_peo_up_deepgemm_num_sms()
        self.num_down_deepgemm_sms = get_peo_down_deepgemm_num_sms()
        self.num_ranks = dist.get_world_size()

        assert self.num_deepep_sms <= self.num_device_sms, f"num_deepep_sms {self.num_deepep_sms} > num_device_sms {self.num_device_sms}"
        assert self.num_up_deepgemm_sms is None or self.num_up_deepgemm_sms <= self.num_device_sms, f"num_up_deepgemm_sms {self.num_up_deepgemm_sms} > num_device_sms {self.num_device_sms}"
        assert self.num_down_deepgemm_sms is None or self.num_down_deepgemm_sms <= self.num_device_sms, f"num_down_deepgemm_sms {self.num_down_deepgemm_sms} > num_device_sms {self.num_device_sms}"

        self.comm_stream = self.dispatcher.get_buffer().get_comm_stream()
        self.local_rank = dist.get_rank()

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
        start_idx: int,
        end_idx: int,
    ):
        if self.deprecate_flag:
            return FusedMoE.run_moe_core(
                self,
                dispatch_output,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        else:
            raise NotImplementedError("Not supported for PEO")

    def forward_overlap_1(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        current_stream: torch.cuda.Stream,
    ):
        global combine_state, dispatch_output
        states = list()
        gemm_done_events = list()
        moe_hidden_states = list()

        torch.set_printoptions(
            threshold=float('inf'),  # 显示所有元素
            precision=4,  # 小数点后4位
            linewidth=120  # 每行宽度
        )
        if self.layer_id == 0:
            hidden_states_device = hidden_states.device
            layer_0 = [[-1.9824e-01, -1.0352e-01, -2.5269e-02,  9.4727e-02,  9.8145e-02,  4.4336e-01, -3.4714e-04,  2.2095e-02,
             -9.4727e-02, -1.8921e-02, -6.8359e-02,  1.3379e-01,  1.3086e-01,  8.1787e-03, -5.5847e-03, -7.1777e-02,
             -4.4678e-02,  5.2490e-02,  9.1553e-03,  4.1260e-02, -4.4678e-02, -1.0742e-02,  1.6556e-03,  5.2490e-02,
              1.1047e-02, -1.9531e-02, -5.5695e-04, -4.0627e-04, -7.8125e-02,  1.0071e-03,  2.5635e-03,  3.1250e-01,
             -2.2583e-03,  8.3496e-02,  4.8584e-02,  8.6914e-02, -3.3936e-02, -2.8198e-02,  2.5177e-03,  1.8692e-03,
             -1.0315e-02, -1.8883e-04, -4.1504e-02,  2.2125e-03,  2.3682e-02, -3.4668e-02, -3.7842e-03, -1.7212e-02,
              8.9169e-05, -5.2979e-02,  4.5410e-02,  4.2915e-04, -9.0820e-02,  2.1057e-03,  2.8038e-04, -2.8809e-02,
             -6.2180e-04,  3.0640e-02,  3.6430e-04, -1.0303e-01,  5.2261e-04,  3.1281e-04, -1.4893e-02,  3.6377e-02,
             -2.4170e-02, -5.8594e-03,  2.4658e-02,  1.3574e-01,  1.0156e-01,  1.2970e-04,  7.9727e-04,  3.8910e-04,
             -5.6885e-02,  8.5449e-04,  5.9128e-04, -2.8687e-03, -3.8757e-03, -1.6113e-01,  3.8330e-02, -1.2634e-02,
             -1.9684e-03,  3.1738e-02,  3.8818e-02, -5.4932e-03, -5.1880e-04,  4.2725e-04,  9.9182e-04, -4.7852e-02,
             -5.4199e-02, -1.0254e-01,  3.6865e-02, -4.7607e-02,  6.7383e-02, -1.5137e-01, -1.1133e-01,  1.1279e-01,
              1.0315e-02,  2.8076e-02, -4.1016e-02,  4.1809e-03,  3.2196e-03, -4.1016e-01, -9.5215e-02, -1.3672e-01,
              8.9722e-03, -4.6349e-04,  3.7695e-01, -8.0566e-02, -3.2959e-02,  2.5195e-01, -4.7607e-02,  2.9182e-04,
             -1.4038e-02, -2.0264e-02, -2.3071e-02,  3.7354e-02,  2.0215e-01,  4.1260e-02, -5.7129e-02,  1.3794e-02,
              3.8574e-02,  1.9043e-02, -9.1553e-03,  7.9956e-03,  9.8633e-02, -6.6528e-03,  1.3733e-02, -1.0376e-02,
              1.1658e-02,  1.2390e-02, -6.2500e-02, -1.5945e-03,  5.7220e-05,  2.2656e-01, -2.3340e-01, -1.6846e-02,
              2.7466e-02,  1.0925e-02, -5.0293e-02, -2.3041e-03, -2.4796e-04, -4.8828e-02, -1.3828e-04, -2.2602e-04,
              3.9062e-02, -1.5945e-03,  2.4780e-02, -4.7607e-02,  8.2016e-04, -2.2095e-02, -1.8750e-01, -9.9659e-05,
             -1.7776e-03, -8.4473e-02, -1.3574e-01,  1.8883e-04, -1.1768e-01,  6.5430e-02,  3.7537e-03,  2.1118e-02,
             -4.7607e-02,  2.6978e-02, -8.0490e-04, -5.3955e-02,  8.6426e-02, -1.7188e-01,  3.2715e-02, -3.9062e-02,
              2.9144e-03, -4.4336e-01, -7.9102e-02, -9.6560e-06,  1.2354e-01,  3.0640e-02, -6.4941e-02, -4.0039e-02,
              7.4158e-03,  6.9336e-02,  1.0645e-01, -2.5269e-02, -5.7373e-03,  2.0885e-04,  8.0566e-02, -4.8340e-02,
             -5.2979e-02, -1.9455e-04, -1.5234e-01, -2.8809e-02, -4.2725e-04,  9.5315e-10, -1.4062e-01,  7.6660e-02,
              1.2695e-01,  1.4551e-01, -1.2756e-02,  1.4404e-02,  9.9487e-03, -3.1006e-02,  6.3477e-02, -2.0996e-01,
              2.0447e-03,  2.5146e-02, -1.3828e-04, -2.2266e-01, -1.8799e-02, -9.0942e-03, -2.4567e-03, -1.5831e-04,
             -2.8564e-02, -8.6426e-02,  1.1353e-02,  7.2002e-05,  1.3199e-03, -2.2030e-04, -7.6172e-02,  5.2490e-02,
              3.7003e-04, -3.4790e-03,  8.3008e-02, -1.3351e-04, -1.4420e-03,  1.0803e-02,  3.3569e-04,  1.5717e-03,
             -1.5320e-02,  3.8147e-03, -3.8818e-02, -7.1777e-02, -1.0109e-04, -1.0742e-01, -5.8105e-02,  5.6641e-02,
              7.6172e-02, -7.2266e-02,  3.5645e-02,  4.5410e-02, -2.5000e-01, -5.1575e-03,  4.0234e-01,  5.6505e-05,
              7.4219e-02,  1.4258e-01, -4.9210e-04, -4.6875e-02, -9.3994e-03,  6.3782e-03, -1.6327e-03, -1.9684e-03,
             -1.7871e-01, -1.9727e-01, -5.6458e-04, -8.5449e-04,  4.2725e-02,  1.1826e-03,  1.0986e-01,  7.2754e-02,
             -2.4292e-02, -7.8964e-04, -1.0529e-03,  1.6357e-02, -3.6523e-01, -1.9684e-03, -2.8564e-02,  3.2227e-02,
              2.3535e-01, -1.4420e-03,  1.0254e-01,  2.2339e-02,  5.8838e-02, -9.7752e-05,  6.3965e-02, -1.3809e-03,
             -9.7168e-02, -1.2305e-01,  1.1963e-01,  8.3984e-01,  4.7852e-02,  1.2360e-03, -1.9908e-05, -4.0894e-03,
              1.8433e-02,  1.1426e-01,  5.2002e-02,  3.0396e-02, -1.8692e-03,  8.2397e-03, -7.7057e-04, -1.5625e-02,
              1.6724e-02, -2.1606e-02, -5.6396e-02,  5.0049e-02, -1.4355e-01,  6.1512e-05, -2.2168e-01, -3.6865e-02,
             -6.5625e-01, -7.3624e-04, -5.0781e-02, -3.5400e-02,  7.9632e-05, -6.7871e-02, -1.0156e-01, -3.1433e-03,
              4.5471e-03,  2.6093e-03,  1.7548e-04, -1.4551e-01, -3.2227e-02, -6.3477e-03, -5.5664e-02, -5.3955e-02,
             -2.6245e-02,  4.0527e-02, -7.6172e-02, -5.6396e-02, -5.9509e-03,  2.4512e-01, -1.0061e-04,  2.4170e-02,
              7.7820e-04, -4.5776e-03,  1.7700e-02,  9.6436e-03,  1.2061e-01,  3.5645e-02,  1.5869e-03, -1.1063e-03,
             -3.8605e-03, -2.4121e-01,  4.2725e-03,  5.9326e-02, -6.3419e-05, -1.2793e-01,  1.9336e-01, -4.4189e-02,
             -4.6875e-02, -1.5625e-01,  2.7344e-02, -2.6855e-02, -2.0508e-02,  9.6191e-02, -2.9785e-02,  3.8338e-04,
             -2.6562e-01, -5.9814e-02, -4.7852e-02, -1.1444e-03,  5.5542e-03,  4.2383e-01, -6.8359e-02,  1.5918e-01,
             -1.0693e-01, -7.7820e-04,  1.0529e-03, -8.7891e-02, -8.2520e-02,  1.1444e-03, -1.0840e-01,  5.6458e-04,
             -4.0234e-01,  3.2715e-02, -9.6436e-03, -7.5684e-02,  1.3504e-03,  5.9128e-04,  2.1973e-02, -5.8838e-02,
              4.5538e-05,  2.3842e-04, -2.5482e-03, -9.4238e-02, -4.0527e-02,  2.4536e-02, -1.1426e-01, -1.1536e-02,
              1.7456e-02,  3.4790e-03, -9.9945e-04, -3.2471e-02,  2.6855e-03,  3.6926e-03,  2.0703e-01,  2.3556e-04,
             -3.2471e-02, -1.4496e-04,  1.6602e-01,  7.1289e-02, -3.1982e-02, -1.6479e-03, -1.0864e-02, -2.2339e-02,
             -3.2997e-04, -2.2363e-01, -4.0039e-01,  2.9053e-02,  1.1719e-01,  8.1177e-03,  2.5879e-02, -1.3477e-01,
              1.7334e-02, -1.3550e-02,  3.6133e-02, -6.3896e-05, -4.5312e-01, -3.2471e-02, -4.1992e-01, -1.4343e-03,
             -2.9297e-02, -3.9673e-03, -2.4902e-01,  3.1055e-01,  1.3379e-01,  2.4128e-04,  3.0029e-02, -3.0640e-02,
              1.9836e-04,  2.5269e-02, -5.5176e-02, -9.1797e-02,  8.5449e-03, -7.8735e-03,  3.7842e-02, -3.7842e-02,
             -7.7438e-04,  2.0117e-01, -1.4099e-02, -1.2158e-01,  1.3924e-04, -8.8120e-04,  2.0508e-02,  3.3203e-02,
             -5.1270e-02, -3.9307e-02, -7.3242e-04,  1.0395e-04, -2.2984e-04, -1.0303e-01, -1.0908e-05, -1.2695e-02,
              5.3223e-02, -1.2634e-02,  1.0400e-01,  9.4986e-04, -9.1797e-02, -9.3079e-04,  4.3457e-02, -4.3030e-03,
              1.0925e-02, -8.1253e-04,  5.1025e-02,  8.8501e-03,  8.4839e-03, -3.2227e-01, -2.4872e-03,  1.6570e-05,
             -1.7578e-01,  8.1253e-04, -1.2891e-01,  2.4223e-04, -4.9805e-01,  1.9727e-01,  4.2725e-02,  4.1504e-02,
             -6.2500e-02, -2.0264e-02,  3.0975e-03, -1.3306e-02, -1.8799e-02,  1.6504e-01, -1.0449e-01,  6.7383e-02,
             -1.0315e-02,  2.1973e-03, -9.6680e-02, -7.7438e-04, -3.3264e-03,  1.9775e-02, -1.0986e-03,  4.9316e-02,
              8.0872e-04,  2.1851e-02,  1.2064e-04,  2.8839e-03,  5.8350e-02,  1.4954e-02,  3.6430e-04,  1.7090e-01,
             -3.1738e-02,  9.6191e-02,  7.7148e-02,  8.2520e-02,  4.7363e-02,  1.7383e-01,  6.6833e-03, -6.2988e-02,
              2.7466e-02,  2.0020e-02, -1.5039e-01, -9.7168e-02,  2.1484e-02,  1.0910e-03, -2.3560e-02,  5.3644e-06,
             -4.1504e-02,  2.5391e-02, -8.7402e-02, -1.2500e-01, -1.2054e-03,  1.2598e-01,  4.4922e-02, -1.7700e-02,
              1.1084e-01, -1.9165e-02, -1.8311e-04,  6.8359e-01, -1.9073e-03, -2.3071e-02, -1.2305e-01,  6.3782e-03,
              3.0640e-02,  1.3977e-02,  1.9775e-02, -1.5503e-02,  2.7924e-03, -3.2715e-02, -1.7090e-02, -1.3086e-01,
              3.3691e-02,  3.9368e-03, -9.2983e-05, -5.8594e-03, -2.0264e-02, -1.2598e-01, -1.7456e-02, -1.8311e-03,
              5.1880e-04,  2.8906e-01,  1.3672e-01,  5.8594e-02, -1.4709e-02, -8.4961e-02, -5.6458e-04,  3.2471e-02,
              1.2451e-02, -6.7383e-02,  1.7834e-04,  3.7231e-03, -6.0797e-05, -2.3438e-02, -1.8799e-02,  8.7500e-01,
              3.0136e-04,  9.9121e-02,  7.8613e-02,  2.7657e-04,  1.1902e-03, -9.3750e-02, -3.7598e-02,  4.0817e-04,
              1.2695e-01, -2.1582e-01, -8.9169e-05, -5.2734e-02, -1.1047e-02, -3.2959e-02,  4.4922e-02, -1.7262e-04,
              1.3885e-03, -3.5706e-03, -4.5471e-03,  2.2888e-05,  4.0283e-03,  3.9795e-02,  3.7689e-03,  6.2866e-03,
              1.1169e-02,  5.2246e-02,  9.6680e-02, -5.7678e-03, -2.6512e-04, -9.9609e-02,  1.9312e-05, -3.6621e-02,
              9.7656e-02,  3.9062e-03,  3.9307e-02, -9.4727e-02,  6.8359e-03, -1.9836e-03, -4.4441e-04,  5.7068e-03,
              7.4219e-02,  1.8997e-03, -5.5176e-02,  3.8818e-02,  2.9373e-04,  8.1055e-02, -8.7402e-02, -8.3160e-04,
             -3.8818e-02, -2.2697e-04, -1.7242e-03,  4.1016e-02,  1.5442e-02,  4.4189e-02, -1.2207e-03, -9.4604e-03,
              1.4258e-01,  5.7861e-02,  3.9291e-04, -2.3535e-01,  3.8574e-02, -1.6016e-01,  1.5625e-02,  5.3711e-02,
             -1.4099e-02,  8.4229e-03, -6.6406e-02,  9.2773e-03, -2.4292e-02,  3.0151e-02, -9.6893e-04,  1.0834e-03,
              2.4780e-02, -6.4941e-02, -3.5400e-02,  3.3447e-02, -1.5869e-02,  3.7598e-02, -3.0518e-04, -1.0529e-03,
             -6.4850e-05,  3.0273e-01,  2.1289e-01, -1.9238e-01,  1.3962e-03,  1.2493e-04, -4.0588e-03, -5.3223e-02,
             -3.9795e-02, -3.4904e-04,  2.8839e-03,  4.2480e-02,  7.8613e-02, -4.0771e-02,  5.2261e-04, -7.2632e-03,
              3.6865e-02,  3.9795e-02,  8.6784e-05, -2.9297e-03, -9.1553e-05,  7.9590e-02,  5.0781e-02, -2.5635e-03,
             -9.1309e-02, -4.2419e-03, -1.4258e-01,  1.7395e-03, -1.5198e-02, -8.2493e-05, -5.4199e-02,  6.7871e-02,
              1.4355e-01,  2.5391e-01, -3.1494e-02,  2.8419e-04, -1.7242e-03, -1.6327e-03, -1.3574e-01,  1.6937e-03,
             -1.0132e-02, -4.1748e-02,  9.6512e-04,  2.6398e-03, -1.2451e-01, -5.2490e-02,  9.9945e-04, -7.8516e-01,
              7.2266e-02, -2.0630e-02, -6.6528e-03,  2.3651e-03,  5.2490e-02,  2.7954e-02,  1.9775e-02, -2.3804e-03,
             -1.3962e-03, -2.4414e-01, -2.4261e-03,  5.1172e-01, -3.7384e-03, -9.2773e-02, -8.5449e-03, -1.2741e-03,
             -3.0060e-03, -4.7302e-04, -2.2125e-03, -1.6504e-01, -1.0986e-03, -2.5757e-02,  2.1289e-01, -8.8120e-04,
             -2.4316e-01, -2.1362e-04, -6.0303e-02,  1.4954e-02, -2.4223e-04,  4.2534e-04,  6.5918e-02, -5.1270e-02,
              4.6387e-02, -5.5664e-02, -5.2643e-04, -6.4453e-02, -4.1008e-05, -2.2852e-01,  5.6152e-02, -2.6703e-03,
              2.4414e-01, -7.3242e-03, -6.5308e-03,  1.4551e-01,  9.8877e-03,  2.8809e-02,  1.0437e-02,  1.3574e-01,
              9.1016e-01, -5.1758e-02, -1.6937e-03, -5.7983e-04, -3.7956e-04,  2.3460e-04, -2.9297e-01, -8.1543e-02,
              2.0020e-02,  4.0527e-02, -1.5918e-01, -6.9336e-02,  4.6692e-03, -2.0264e-02,  2.6001e-02,  2.6245e-02,
              3.4332e-03,  3.1433e-03, -3.8818e-02,  7.0953e-04, -2.4658e-02, -2.0142e-02, -5.4169e-04,  1.4954e-02,
              3.4332e-03,  1.0400e-01, -2.8076e-03, -1.6992e-01, -9.3994e-03, -1.1816e-01,  3.2715e-02, -1.0254e-02,
              1.7090e-02,  1.5068e-04, -2.0752e-03,  1.8262e-01, -1.2207e-03, -4.2419e-03,  8.8379e-02, -1.0205e-01,
              3.0151e-02, -6.5430e-02, -1.7383e-01,  4.1504e-02, -1.0986e-01,  1.1133e-01,  5.4932e-04,  8.3496e-02,
              5.1514e-02, -1.1749e-03, -3.2471e-02,  9.7046e-03,  4.2915e-04, -1.9836e-04, -4.6143e-02,  1.3965e-01,
              1.1673e-03,  3.4180e-02,  3.1494e-02, -4.9133e-03,  1.9836e-04, -2.6733e-02,  1.2695e-01, -2.6123e-02,
             -2.8992e-04, -2.5940e-04,  3.7193e-05,  4.1199e-04, -5.8746e-04, -7.4005e-04,  1.3489e-02, -6.3782e-03,
              4.5013e-04, -1.1914e-01, -3.9062e-02,  2.2583e-03, -9.9182e-04,  5.0306e-05, -6.5918e-02,  4.3945e-01,
              5.7861e-02,  1.7090e-01,  3.3112e-03, -9.0820e-02, -1.8848e-01, -9.4727e-02, -4.6875e-02, -6.6833e-03,
              5.6458e-04,  2.3193e-03,  5.0293e-02, -1.0010e-01,  7.4707e-02, -4.8828e-04, -8.6060e-03,  1.2695e-01,
             -5.8365e-04, -4.2480e-02, -6.8970e-03,  2.2030e-04,  2.7466e-03,  2.5195e-01,  8.3984e-02, -1.2402e-01,
              4.1962e-05,  3.5156e-01, -5.0391e-01, -2.7275e-04, -1.2207e-04,  8.3923e-04, -1.5234e-01,  2.3926e-02,
              3.6316e-03,  1.1230e-02, -1.2695e-01,  1.5234e-01, -5.2490e-02,  6.8665e-05, -3.1250e-02, -1.1230e-01,
             -1.4258e-01, -8.3496e-02, -7.0801e-02, -1.7357e-04,  1.1816e-01, -1.1673e-03,  6.8359e-03, -4.9316e-02,
              5.5695e-04,  3.0078e-01,  4.3945e-02, -7.9590e-02,  1.5625e-01,  8.8867e-02,  3.2043e-03, -2.8906e-01,
             -5.4443e-02, -5.4169e-04,  1.2329e-02,  4.6082e-03, -1.4941e-01, -2.5511e-05, -5.3644e-05,  1.0986e-03,
              3.5400e-02,  2.2266e-01,  1.3962e-03, -1.2451e-01, -4.6921e-04,  9.2773e-03, -6.5918e-02,  7.9102e-02,
              1.2207e-04, -7.1526e-06, -2.2705e-02, -1.6861e-03,  1.8597e-05,  1.7700e-02, -6.6280e-05, -6.6833e-03,
              6.0303e-02,  2.8711e-01,  1.3733e-03, -3.2715e-02, -1.6968e-02, -4.6631e-02,  8.9355e-02, -4.2114e-03,
             -5.6152e-02, -1.6403e-03,  8.6426e-02,  2.9373e-04,  3.2227e-02, -9.6191e-02, -7.2861e-04, -2.7222e-02,
             -2.8534e-03, -1.5918e-01,  1.0107e-01, -6.3965e-02, -7.0801e-02, -4.9072e-02,  6.7871e-02,  3.0212e-03,
              3.5889e-02, -2.5781e-01, -1.0352e-01, -1.2451e-01, -4.0625e-01,  1.0071e-03,  6.3324e-04,  7.9102e-02,
              3.6240e-05,  1.1621e-01, -2.3828e-01, -1.9653e-02, -1.5076e-02, -6.2256e-03,  5.1758e-02, -2.1606e-02,
              1.0693e-01,  6.5430e-02, -3.1250e-02,  6.3672e-01,  1.1182e-01, -8.2397e-03, -3.8477e-01, -2.1606e-02,
             -6.6528e-03,  3.1006e-02, -8.7402e-02,  8.9645e-04, -9.7656e-04,  6.6280e-05, -1.9684e-03, -4.6143e-02,
             -3.7384e-03,  1.1719e-02, -1.2741e-03, -6.6406e-02,  2.7100e-02, -8.4961e-02, -2.1118e-02,  3.0136e-04,
              7.0312e-02, -2.1118e-02,  2.7930e-01,  2.7930e-01, -7.2266e-02, -6.2866e-03, -2.9907e-02,  2.5757e-02,
              3.6049e-04, -2.0117e-01,  1.9531e-01, -3.4485e-03, -3.1055e-01,  6.5430e-02,  1.7090e-03, -2.2292e-05,
             -6.0730e-03,  2.2888e-03, -9.4727e-02, -8.1055e-02,  1.4973e-04,  2.4121e-01,  3.5400e-02,  3.1233e-05,
              8.6914e-02,  2.5269e-02,  8.0566e-02, -9.2773e-03,  1.2695e-01, -5.5237e-03, -4.7913e-03,  5.3711e-03,
             -1.3065e-04,  3.6621e-04, -6.2561e-04, -7.5195e-02,  1.2817e-02, -1.0059e-01,  2.9883e-01,  1.2695e-02,
             -3.6926e-03, -1.6699e-01, -9.7656e-03, -5.1025e-02, -7.7637e-02, -7.3242e-02,  1.8677e-02, -1.0864e-02,
             -2.3174e-04,  4.1016e-02, -7.9956e-03,  1.0547e-01, -1.6357e-02, -1.8082e-03, -1.1768e-01,  8.4473e-02,
              1.4160e-02,  2.1729e-02, -1.8945e-01,  2.5940e-03, -1.7578e-02, -3.9482e-04, -1.1749e-03,  1.7090e-03,
             -4.6997e-03,  3.8757e-03, -4.2534e-04,  3.1982e-02, -3.0029e-02, -1.2207e-04, -6.3477e-02, -1.3428e-03,
             -3.5400e-02,  2.4780e-02, -1.7456e-02, -5.0293e-02,  2.0874e-02,  4.2480e-02, -1.8311e-02, -1.6235e-02,
              8.4961e-02, -2.7222e-02,  1.6098e-03,  1.5354e-04, -1.4156e-06,  8.8692e-05,  2.9653e-06, -4.5896e-06,
              7.5340e-05,  5.7373e-02,  1.1749e-03, -3.6865e-02,  2.1240e-02, -6.0272e-04,  6.6406e-02,  6.8283e-04,
             -1.4648e-01,  4.8447e-04,  2.5787e-03,  8.5831e-05,  4.0527e-02,  1.2988e-01, -5.7373e-02, -2.9373e-04,
              1.9531e-03,  1.8066e-02,  1.2329e-02,  1.5640e-03, -1.0147e-03,  6.1768e-02,  1.9629e-01, -2.6093e-03,
              9.1309e-02,  9.9487e-03,  1.3770e-01,  2.6733e-02, -4.3945e-03,  4.8637e-05, -2.5757e-02, -4.1260e-02,
             -1.0193e-02,  5.9570e-02, -1.2695e-02,  2.8687e-02,  1.5015e-02, -7.1289e-02,  1.4355e-01,  8.8501e-03,
             -7.1777e-02, -1.0303e-01, -1.0254e-02,  1.0559e-02, -1.2894e-03, -1.2302e-04,  2.5635e-03, -5.3223e-02,
              7.1289e-02, -3.4904e-04, -9.3262e-02, -1.8158e-03, -2.4121e-01,  4.0527e-02, -2.1777e-01,  2.8906e-01,
             -1.9193e-05, -1.3672e-01,  3.9062e-02,  6.3705e-04, -1.8066e-02, -1.0010e-02,  8.8379e-02, -1.8262e-01,
              5.9082e-02, -1.2988e-01, -1.5137e-01,  2.2221e-04,  1.4099e-02,  8.2016e-04, -6.0425e-03,  1.2451e-01,
              1.7871e-01, -3.2471e-02,  6.1279e-02, -2.0905e-03, -2.0027e-04,  5.9128e-04, -2.2705e-02,  1.3428e-02,
              5.3223e-02,  1.0681e-02, -9.1171e-04, -5.6396e-02, -1.0742e-01,  2.4219e-01, -9.7656e-02,  2.6703e-05,
             -7.8583e-04, -1.4258e-01, -7.3242e-03,  1.2988e-01, -2.0142e-03,  5.8746e-04,  1.7676e-01,  2.5000e-01,
             -4.0527e-02,  1.3794e-02, -1.1873e-04, -5.9326e-02, -2.0386e-02, -2.6001e-02,  1.4832e-02,  8.0872e-04,
              3.2227e-02, -9.5215e-02, -1.0193e-02, -2.8076e-03,  1.5234e-01, -1.5182e-03,  8.3496e-02,  2.4605e-04,
              1.3477e-01, -1.3428e-02, -3.7537e-03, -1.3428e-02, -4.8523e-03, -4.5471e-03,  7.0312e-02, -8.3008e-02,
             -3.6133e-02,  1.1206e-04, -2.0996e-02,  7.4219e-02, -1.4648e-01,  5.5664e-02, -2.9297e-03, -2.6733e-02,
              6.0791e-02, -6.1417e-04,  5.7129e-02, -3.2959e-02, -1.0303e-01,  3.5156e-02, -3.3379e-04,  1.8768e-03,
             -5.5313e-04,  1.0791e-01, -3.6316e-03, -5.2344e-01, -5.3406e-03, -2.9663e-02,  2.2583e-02, -3.4180e-02,
             -4.5410e-02, -4.2969e-02, -4.1992e-02,  9.5215e-02, -1.8188e-02, -1.7334e-02, -1.1719e-01,  8.4961e-02,
             -1.2793e-01, -1.9434e-01,  1.2360e-03,  7.7637e-02, -1.5723e-01, -4.2725e-03,  1.9043e-02, -9.9945e-04,
             -2.4557e-05,  2.9419e-02,  1.1673e-03,  1.8188e-02,  1.4160e-02, -5.8899e-03, -8.6914e-02, -9.4727e-02,
              3.3691e-02,  5.6641e-02, -6.8359e-02, -4.6158e-04, -2.3174e-04, -8.0078e-02,  3.7384e-04, -9.5215e-02,
             -6.7871e-02,  4.8828e-04,  3.3936e-02,  5.5908e-02, -1.3867e-01, -1.3477e-01, -9.6680e-02, -3.1853e-04,
              2.7710e-02, -1.0645e-01, -1.8677e-02, -4.8584e-02, -3.5889e-02, -8.9844e-02, -4.6082e-03, -1.7285e-01,
             -2.7657e-04, -2.7344e-01, -6.7444e-03,  2.4796e-04,  1.6504e-01, -1.8005e-03, -8.0872e-04, -1.6113e-02,
              2.3723e-05, -3.4180e-03,  4.9438e-03,  4.3488e-04, -1.2812e+00,  1.8359e-01,  2.0264e-02,  3.4523e-04,
             -2.6001e-02, -5.0293e-02, -1.5503e-02,  6.8359e-03,  1.0693e-01,  6.4453e-02,  4.5410e-02,  3.7537e-03,
              1.7773e-01,  1.4160e-02,  8.4961e-02,  6.2561e-04, -3.5048e-05,  2.0504e-05,  6.6223e-03,  4.1504e-02,
             -7.2021e-03, -2.2949e-02, -1.2054e-03, -2.6001e-02,  9.1797e-02, -8.4961e-02,  1.1719e-02, -1.0864e-02,
              4.9973e-04,  1.5234e-01,  5.0049e-03, -1.1328e-01, -1.1368e-03,  1.0010e-02, -3.7384e-04,  3.9551e-02,
              1.3770e-01,  4.1809e-03,  2.0905e-03, -1.9824e-01,  1.5991e-02,  4.4632e-04, -1.9150e-03,  2.2697e-04,
             -9.3994e-03, -1.8120e-04,  7.5684e-02,  3.3691e-02, -1.2779e-04, -2.1191e-01,  1.1621e-01, -6.7871e-02,
             -5.2452e-05,  9.9487e-03, -3.5889e-02,  8.6670e-03, -6.3782e-03,  1.3611e-02, -1.3000e-02, -8.3542e-04,
              2.9907e-03,  1.8164e-01, -1.9531e-02,  6.2988e-02, -2.2583e-02, -2.1219e-05, -1.7480e-01, -5.4688e-02,
             -1.6479e-02,  5.8594e-02, -1.7334e-02, -1.4941e-01, -5.3787e-04,  2.2705e-02,  6.3171e-03, -5.6396e-02,
              2.3145e-01, -8.6426e-02,  1.4648e-03,  8.9722e-03, -1.4343e-03,  6.5625e-01,  2.3956e-03, -5.6885e-02,
             -3.0640e-02, -5.7422e-01, -1.7188e-01, -6.5994e-04, -3.9795e-02, -7.5378e-03,  4.4189e-02,  7.4158e-03,
             -4.9316e-02,  5.7068e-03,  2.9802e-05, -1.0529e-03,  1.1230e-01, -2.0386e-02, -4.1199e-04,  5.3711e-03,
              1.3672e-02, -1.3245e-02,  3.2471e-02,  5.0964e-03, -2.0410e-01, -2.3651e-03,  6.3705e-04, -2.9373e-04,
             -6.5994e-04,  1.6211e-01, -8.3618e-03, -9.0332e-03,  4.3213e-02,  7.1335e-04, -4.2236e-02,  3.3722e-03,
              2.3926e-01,  5.1880e-04, -6.0425e-03, -7.2479e-04,  1.9775e-02,  3.0884e-02, -3.5400e-03, -7.7148e-02,
              1.5411e-03,  1.4114e-03, -9.0942e-03,  9.5703e-02,  8.5068e-04,  1.2159e-04,  8.1177e-03,  5.8350e-02,
             -3.8867e-01,  1.1780e-02,  4.2725e-03,  2.4567e-03,  6.3324e-04, -1.5855e-05, -1.1396e-04,  8.8120e-04,
             -2.3956e-03,  1.1063e-03, -6.0791e-02,  1.4258e-01, -1.7452e-04, -8.3008e-03, -6.5430e-02,  2.4261e-03,
             -5.1758e-02,  4.4189e-02,  6.3705e-04,  1.7822e-02,  2.8809e-02,  2.8229e-03,  1.7578e-01,  2.3079e-04,
              4.3213e-02,  4.7852e-02,  1.8262e-01, -8.5449e-03, -1.1719e-01, -1.8262e-01, -1.8433e-02, -6.3324e-04,
              1.0925e-02, -4.5166e-02,  1.7383e-01,  6.6895e-02,  1.4551e-01,  2.3991e-06, -8.7357e-04, -5.2734e-02,
             -6.6757e-05,  7.4707e-02, -1.6968e-02, -3.0078e-01,  4.7363e-02, -1.5747e-02, -1.3086e-01,  7.8125e-03,
              6.5231e-04, -4.8584e-02, -8.2520e-02,  1.3855e-02,  1.0986e-01, -1.8082e-03,  2.8076e-02,  1.0840e-01,
              5.2261e-04,  3.4912e-02, -2.8198e-02,  3.6716e-05,  3.1250e-02,  6.5308e-03,  2.4319e-04,  1.4343e-02,
             -8.1055e-02, -1.9043e-02,  3.2425e-04,  5.6885e-02,  6.9275e-03, -1.0254e-02,  1.0303e-01,  2.6245e-02,
              1.1444e-04,  4.7684e-04, -1.3885e-03, -3.9062e-02, -5.4932e-03,  5.0659e-03,  7.5195e-02, -1.7853e-03,
             -7.6172e-02,  1.3275e-03, -5.2344e-01,  6.2866e-03, -1.2012e-01,  7.6771e-05,  3.2616e-04,  1.6797e-01,
             -4.0283e-02, -8.1177e-03, -1.4591e-04, -9.8267e-03, -3.9062e-02,  4.0771e-02, -6.0303e-02, -3.6621e-04,
              2.2583e-02,  2.2461e-02, -1.9646e-04, -1.4954e-03,  2.4223e-04,  5.6152e-03,  5.9570e-02,  2.5757e-02,
              2.0142e-02, -1.9531e-03,  1.3367e-02,  9.5825e-03, -5.0735e-04, -8.0566e-02,  3.8281e-01,  1.3000e-02,
              5.1758e-02, -8.3008e-03, -5.9814e-02,  3.4668e-02, -1.2256e-01,  2.6489e-02,  1.0449e-01, -4.0039e-02,
             -2.5392e-05,  1.8066e-02,  3.0273e-02, -1.0371e-05,  2.4902e-02, -1.0059e-01, -2.1484e-02, -8.0872e-04,
              5.7220e-04,  7.5073e-03,  1.2085e-02,  3.9864e-04,  1.4746e-01, -6.4392e-03,  9.1309e-02,  1.0757e-03,
              2.1815e-05, -2.4121e-01, -3.4332e-03, -3.4424e-02,  4.7607e-02,  9.2163e-03, -3.3203e-02, -7.8125e-02,
             -5.4199e-02, -8.8867e-02, -8.1543e-02,  1.8457e-01,  2.5195e-01,  7.1777e-02,  5.5237e-03,  1.8750e-01,
              1.8158e-03,  1.4746e-01, -6.3777e-06,  4.4922e-02, -1.2329e-02, -4.1992e-02, -1.3947e-05, -1.2988e-01,
             -1.9150e-03, -5.4169e-04,  4.1260e-02, -2.6131e-04,  4.3030e-03, -5.2979e-02, -3.0151e-02, -7.6172e-02,
              1.9727e-01, -1.5234e-01, -9.6130e-04,  3.1006e-02,  2.6489e-02,  1.0147e-03, -3.1494e-02,  1.5259e-04,
             -1.9141e-01,  3.2349e-03, -4.6997e-03,  4.1246e-05, -4.4556e-03,  3.2471e-02,  1.1572e-01,  3.7598e-02,
             -6.6016e-01,  1.2219e-05, -6.5430e-02, -2.0386e-02,  1.1597e-03, -1.1253e-04, -1.9264e-04, -1.5039e-01,
              7.7148e-02, -7.5912e-04, -3.3379e-04,  8.4305e-04,  1.9989e-03, -3.2471e-02, -1.3367e-02,  7.6599e-03,
              3.8086e-02, -2.3560e-02,  8.2520e-02,  4.4531e-01, -2.8038e-04, -2.0142e-03, -2.3651e-03,  1.7071e-04,
              6.4373e-05, -1.1719e-01,  6.1035e-02,  1.8921e-02,  6.7520e-04,  3.7689e-03,  6.8283e-04, -1.7383e-01,
              2.2221e-04, -1.1621e-01, -5.3223e-02,  2.0020e-02,  1.0352e-01, -1.0498e-01, -2.8076e-03,  6.6406e-02,
              4.1504e-02, -1.6968e-02,  3.3447e-02, -2.8198e-02,  8.0566e-02,  5.1880e-03,  3.7842e-03,  2.4536e-02,
              2.1118e-02, -5.7697e-05, -3.5400e-02,  6.8188e-05, -4.3335e-03,  8.9355e-02,  5.1880e-03, -7.2754e-02,
              6.2500e-02,  5.2261e-04, -3.6914e-01,  1.2207e-02, -1.2695e-02, -4.4189e-02,  5.7983e-03, -1.2779e-04,
             -2.0630e-02, -4.6387e-02,  5.1498e-04,  5.7422e-01,  8.2016e-04, -1.9897e-02, -2.8968e-05, -4.4250e-04,
              1.6504e-01, -4.4434e-02, -2.3193e-03,  1.5039e-01, -9.2773e-02, -3.8086e-01, -2.6978e-02, -4.2480e-02,
              1.2891e-01, -6.2943e-04,  7.1716e-03,  4.0283e-03,  4.6387e-02, -7.1777e-02, -5.8594e-02,  1.1279e-01,
             -4.9072e-02,  4.2114e-03, -2.1387e-01, -4.2439e-05, -6.9046e-04, -6.8188e-05, -8.9844e-02, -5.8746e-04,
             -3.4424e-02, -1.5259e-03, -2.1484e-02,  4.1504e-02,  1.0498e-01,  1.6098e-03, -8.0566e-02,  1.4114e-03,
              1.9531e-02,  5.6982e-05, -2.8229e-03,  5.9128e-04,  1.6113e-02,  1.0071e-03, -1.2793e-01,  1.2422e+00,
              4.2419e-03, -4.1797e-01,  6.9046e-04, -9.4727e-02, -1.1597e-02,  1.4355e-01,  4.4250e-03, -3.0365e-03,
              1.1635e-04,  9.3994e-03, -1.2451e-02, -5.2185e-03, -2.1606e-02,  1.6861e-03, -7.4707e-02, -6.5613e-04,
             -7.0953e-04, -1.7334e-02,  1.2939e-02,  9.1797e-02, -5.3516e-01, -1.2793e-01, -5.2246e-02,  3.8574e-02,
              1.6602e-02, -7.4463e-03,  1.2695e-01,  4.0039e-02,  2.0508e-02,  7.2937e-03, -1.3611e-02, -2.1458e-04,
              5.8838e-02,  1.7014e-03,  1.4400e-04,  1.4591e-04, -2.2793e-04,  1.8787e-04,  3.5645e-02,  1.1292e-03,
              4.0283e-02, -2.1973e-01, -3.3203e-01,  1.2970e-03,  2.3535e-01,  5.4443e-02,  4.5166e-02, -6.8665e-04,
             -8.2970e-05,  8.7891e-02, -4.7363e-02,  4.4823e-04, -1.0254e-01, -1.8358e-05, -4.4922e-02, -1.2500e-01,
              1.8188e-02, -9.9487e-03, -1.7578e-01,  2.2095e-02,  9.9182e-04, -5.5176e-02,  2.5513e-02, -2.3346e-03,
             -9.1309e-02,  4.5166e-02, -4.2725e-03,  1.0147e-03,  1.6406e-01,  3.7432e-05, -1.3770e-01, -9.5215e-02,
              1.3924e-04,  1.0872e-04, -1.2451e-01,  3.7842e-03,  4.3213e-02,  1.3672e-01, -3.9648e-01, -8.0566e-03,
             -1.6846e-02,  2.1875e-01, -1.3611e-02, -2.4986e-04,  6.7871e-02,  3.8330e-02, -5.9814e-03, -2.9883e-01,
              1.5020e-05, -1.2158e-01, -7.9590e-02, -7.2021e-03, -6.9580e-03,  2.4170e-02,  3.1250e-02, -1.3184e-01,
              4.1199e-03,  4.4434e-02,  7.5195e-02, -2.2266e-01,  6.9824e-02, -2.9419e-02, -6.3965e-02,  2.4109e-03,
             -2.5757e-02,  1.6556e-03,  1.0681e-03, -2.1973e-01, -1.9922e-01, -1.6113e-01,  1.3504e-03,  3.0640e-02,
             -6.1768e-02,  2.8198e-02,  1.2891e-01,  8.9264e-04, -1.1377e-01,  2.7466e-02, -1.2158e-01, -3.9062e-03,
             -7.0190e-04,  1.0071e-02, -1.8359e-01, -7.7820e-04,  1.9531e-02,  1.9775e-02,  2.9785e-02, -2.8564e-02,
             -1.8311e-03, -3.6377e-02,  1.1279e-01,  1.8463e-03,  5.4321e-03, -3.1891e-03, -2.1362e-02, -5.3467e-02,
              2.0874e-02, -1.5717e-03,  4.0039e-02, -3.3936e-02,  6.3477e-02, -1.1902e-02, -1.6937e-03,  6.7711e-05,
             -4.2114e-03, -4.4434e-02, -4.1809e-03,  2.2583e-02, -1.5527e-01, -1.8164e-01, -9.5215e-03, -4.0283e-02,
             -6.8665e-05, -3.5095e-04,  1.2988e-01, -2.7618e-03,  2.1744e-04,  1.3770e-01, -3.6621e-02, -3.3691e-02,
              4.3945e-02, -2.0599e-03, -2.1515e-03,  3.4904e-04, -4.0293e-05, -1.0014e-04,  3.0823e-03,  7.9590e-02,
              9.3842e-04,  5.9204e-03, -6.6833e-03, -8.3984e-02,  8.9722e-03, -1.7090e-03,  3.2501e-03, -2.7466e-04,
             -5.7617e-02,  2.8442e-02,  2.4261e-03,  7.2754e-02,  4.0771e-02,  3.4668e-02,  5.2979e-02, -7.3730e-02,
              6.5918e-03, -2.2278e-03,  8.6914e-02, -1.3184e-01, -2.1973e-02,  2.0630e-02,  1.8188e-02, -7.1335e-04,
              2.6953e-01, -6.8359e-02,  8.3984e-02,  6.9824e-02, -6.8359e-02, -2.3730e-01,  1.4355e-01,  6.2988e-02,
             -3.2336e-06, -5.1758e-02,  6.5231e-04, -1.3672e-01,  1.8359e-01, -2.6978e-02,  1.4404e-02,  2.5749e-04,
              7.6660e-02,  2.5749e-04,  1.9727e-01,  2.4048e-02,  6.6895e-02, -1.0681e-03, -1.4114e-04,  6.6757e-04,
             -1.3306e-02,  1.0059e-01, -7.8735e-03, -1.2451e-02, -6.0547e-02,  1.3809e-03, -3.1250e-02, -2.6367e-02,
              5.0049e-02, -4.6143e-02,  4.5898e-02,  9.2773e-02,  8.4473e-02,  6.5002e-03, -4.1260e-02,  5.0783e-05,
             -1.7578e-02,  6.8665e-04,  1.5527e-01,  7.3242e-02,  3.2806e-04, -5.3955e-02,  1.4844e-01,  1.0254e-02,
              1.0059e-01,  1.8921e-02, -3.2806e-03, -1.9073e-03,  8.4043e-06,  1.0529e-03, -5.7817e-06,  4.6539e-04,
             -1.3574e-01, -1.7071e-04,  1.0071e-03, -3.0212e-03, -3.7354e-02, -7.4219e-02,  1.2779e-04,  6.0059e-02,
             -1.4160e-02,  3.9062e-02,  5.0537e-02,  1.7188e-01,  1.4954e-02,  1.2109e-01, -5.7373e-02, -1.1169e-02,
             -9.0332e-02,  6.1035e-02, -9.9609e-02,  7.2021e-03,  2.0123e-04, -7.2754e-02, -5.2246e-02, -1.6235e-02,
             -1.2085e-02,  9.8877e-03,  1.4160e-02,  1.0156e-01, -1.9312e-05,  6.1768e-02, -1.2665e-03, -4.8828e-02,
             -2.5195e-01,  8.2397e-03, -2.1362e-02, -2.1515e-03, -7.8964e-04, -1.1963e-01, -1.0925e-02,  1.6699e-01,
             -9.1553e-03,  6.1646e-03,  4.5967e-04, -1.2589e-03, -1.0986e-02,  1.1279e-01, -1.7014e-03, -2.0703e-01,
             -3.9551e-02, -1.4484e-05, -3.0518e-02, -1.8921e-02,  1.6785e-03,  6.9336e-02, -1.1475e-02,  1.0107e-01,
             -4.9174e-06, -5.7617e-02,  1.1426e-01, -3.8330e-02,  3.3936e-02, -3.0859e-01, -4.3164e-01, -2.3438e-01,
              1.8945e-01, -3.3203e-01, -1.3306e-02,  1.2436e-03,  1.7452e-04, -2.2316e-04,  7.3730e-02,  7.8678e-05,
             -2.0874e-02,  7.8678e-05, -1.3184e-01, -6.3324e-04, -2.7161e-03,  1.0840e-01, -2.9907e-02,  1.1444e-03,
             -6.8359e-02, -6.0654e-04,  1.4648e-01, -1.8311e-02, -6.5002e-03,  1.7676e-01,  2.0386e-02,  1.7700e-03,
              1.5488e-03,  4.7112e-04,  1.3885e-03,  2.0294e-03,  4.2152e-04, -6.2866e-03, -5.2261e-04,  1.0538e-04,
             -3.9673e-04, -2.4567e-03,  7.8613e-02, -1.7700e-02,  5.3467e-02, -5.0293e-02,  3.5858e-04,  6.0791e-02,
              3.5400e-02,  1.8359e-01, -1.3574e-01,  8.3984e-02,  7.4707e-02,  9.1797e-02,  1.5259e-02,  1.5869e-02,
              1.3809e-03, -7.7820e-04,  5.2643e-04, -2.0386e-02,  5.6458e-04, -4.2725e-03, -7.6904e-03,  9.0408e-04,
              1.0864e-02, -1.2360e-03, -9.3262e-02, -8.4961e-02,  3.3594e-01,  1.7285e-01, -1.2817e-02, -9.1553e-03,
              6.8359e-02,  6.9885e-03, -2.0142e-03,  2.3050e-08,  5.2734e-02, -1.8403e-06,  1.0925e-02,  4.6143e-02,
             -3.1641e-01, -3.4142e-04,  1.3794e-02, -2.4707e-01, -1.7700e-03, -7.6172e-02,  1.0742e-02,  2.6758e-01,
              1.7383e-01,  8.5449e-02, -1.2500e-01, -1.5793e-03, -9.1553e-03,  7.6294e-03,  2.9297e-01,  6.7871e-02,
              5.8594e-03, -5.2795e-03, -1.9775e-02, -1.4062e-01, -4.5898e-02, -5.0354e-04, -8.6975e-04,  4.1504e-02,
              2.7539e-01, -1.2875e-04,  2.0630e-02, -7.7820e-03,  2.6489e-02,  9.6436e-03, -2.9297e-03,  1.1719e-02,
             -4.0527e-02,  4.1962e-04,  1.3770e-01,  1.2793e-01, -9.5844e-05, -6.8359e-02,  4.4632e-04,  8.4473e-02,
             -2.1362e-02, -3.2227e-02,  1.9531e-02, -4.7607e-02,  2.5024e-02, -6.1035e-02, -5.9814e-03, -4.6387e-03,
              1.3281e-01, -2.0264e-02, -4.9133e-03, -1.9531e-02,  1.3086e-01,  4.3701e-02,  1.1292e-03,  6.6280e-05,
             -3.3936e-02,  1.3794e-02,  1.5991e-02,  5.0391e-01, -1.4062e-01,  8.9169e-05, -2.4902e-01, -3.4180e-02,
              4.7607e-02,  2.6978e-02, -6.0303e-02, -3.5889e-02,  6.6406e-02, -4.9744e-03, -5.5908e-02, -2.5024e-02,
             -5.9891e-04,  1.9165e-02,  5.5176e-02, -2.9297e-02,  2.8809e-02, -1.2500e-01,  1.4526e-02,  1.1377e-01,
             -3.2485e-06,  1.0254e-01, -1.4221e-02, -7.8735e-03, -1.3672e-02, -7.3242e-03, -6.7383e-02, -4.2480e-02,
             -2.3499e-03,  2.2705e-02,  1.4099e-02,  1.1978e-03,  6.7383e-02,  7.2632e-03,  1.0347e-04,  1.1475e-01,
             -7.0801e-03,  2.8906e-01,  6.7871e-02,  1.2329e-02,  5.8899e-03,  8.3984e-02,  1.7776e-03, -1.5381e-02,
             -1.6602e-02,  1.0156e-01, -4.5776e-05, -2.2316e-04, -7.2021e-03,  5.1737e-05,  1.5137e-02,  6.7871e-02,
              4.3701e-02, -1.0071e-02,  3.2471e-02, -9.1553e-05, -3.8574e-02, -8.6426e-02,  5.0293e-02,  7.0095e-05,
             -3.2959e-02, -6.5308e-03, -1.2390e-02, -1.2256e-01,  3.2227e-02, -8.1177e-03, -2.4261e-03, -4.0820e-01,
              1.3809e-03,  1.4355e-01,  1.3977e-02,  5.5075e-05,  8.4473e-02, -5.1514e-02,  8.2520e-02, -2.0142e-02,
              8.6426e-02, -5.2002e-02,  8.1177e-03, -5.3787e-04, -2.4048e-02, -3.7575e-04,  4.7607e-02,  6.9427e-04,
             -6.8054e-03, -8.1787e-03, -5.1498e-04, -1.3672e-01, -8.9722e-03, -6.4087e-03,  3.2616e-04,  7.8613e-02,
             -1.4526e-02,  4.5654e-02,  5.1758e-02,  9.3579e-06,  4.8096e-02, -4.2114e-03, -2.1362e-02, -3.0859e-01,
             -4.4250e-03, -4.7656e-01, -4.1504e-02, -5.0964e-03,  6.5430e-02, -2.7148e-01, -1.1621e-01,  3.5048e-05,
              5.4169e-04,  3.4961e-01,  8.6426e-02, -8.6914e-02,  1.4114e-04,  8.3008e-03,  1.1902e-03, -2.9907e-02,
              4.1992e-01,  2.0599e-03,  7.7438e-04, -1.5381e-02,  2.6245e-03, -1.5320e-02, -6.0547e-02, -1.6211e-01,
             -7.7637e-02, -1.2207e-02,  2.1057e-03,  1.8652e-01, -2.2827e-02,  4.6692e-03, -1.0586e-04, -7.3314e-06,
              2.0447e-03, -2.1606e-02,  3.3447e-02, -2.6367e-02, -1.7548e-03, -1.2305e-01,  4.7607e-02,  4.9072e-02,
             -2.2949e-02, -2.5269e-02, -8.6914e-02,  3.3379e-04,  5.2246e-02, -7.5195e-02,  6.2988e-02, -3.1090e-04,
             -1.2146e-02,  7.4506e-06, -1.2589e-03,  4.2725e-03, -2.8809e-02,  1.3379e-01,  3.9795e-02, -6.3477e-02,
              2.3340e-01,  7.3730e-02, -2.7466e-03, -2.8229e-03,  1.5430e-01, -3.0078e-01, -1.3281e-01,  2.4902e-02,
             -3.8086e-02,  3.4180e-02, -4.8637e-04,  5.2002e-02, -3.8574e-02,  1.0010e-02, -1.2779e-04,  1.3977e-02,
             -3.2715e-02, -2.7100e-02, -6.7902e-04, -1.1108e-02,  3.6430e-04,  7.0312e-02, -6.3477e-02, -1.0742e-02,
              9.6436e-03, -9.8145e-02,  6.7749e-03,  1.3428e-03,  2.3270e-04, -4.4727e-01, -7.7148e-02, -2.0630e-02,
              1.0547e-01,  1.3184e-02, -2.6978e-02,  3.8910e-03, -9.0820e-02, -5.9082e-02, -2.9602e-03,  6.5613e-03,
             -7.1777e-02,  2.1667e-03,  1.8677e-02,  4.5898e-02,  1.1670e-01,  4.0588e-03,  1.6113e-01,  8.4961e-02,
              1.4746e-01, -8.6670e-03, -4.4678e-02,  2.8038e-04,  7.3547e-03, -6.2988e-02,  5.8838e-02,  1.9824e-01,
              1.0059e-01, -3.3691e-02,  1.0693e-01,  0.0000e+00, -4.5166e-02, -4.4189e-02, -6.8359e-02,  3.2043e-03,
             -9.3460e-04, -3.9062e-03,  5.3406e-04,  4.2969e-01,  1.0071e-02, -1.0437e-02, -1.9336e-01, -1.6937e-03,
              7.9590e-02,  1.0596e-01,  2.7466e-02, -2.2705e-02,  6.4671e-06, -2.3560e-02, -1.6895e-01, -1.7456e-02,
              9.7656e-02,  3.1948e-05,  7.1289e-02, -5.3711e-02, -2.5177e-04, -1.3199e-03,  6.7520e-04, -1.6113e-02,
             -7.4158e-03, -2.3535e-01, -1.0107e-01,  2.8839e-03, -4.9805e-02,  3.8910e-04, -5.7983e-03, -5.0537e-02,
              1.3638e-04, -3.3379e-05, -1.9264e-04,  4.3701e-02,  2.3071e-02, -1.2939e-02,  3.5156e-02,  9.5215e-02,
              7.7637e-02,  4.3457e-02,  2.4170e-02,  5.3223e-02, -2.3926e-01,  3.1233e-05, -4.3213e-02,  6.2012e-02,
              6.8359e-02, -1.8066e-02, -2.5195e-01,  8.3923e-05,  1.9264e-04, -3.3447e-02,  1.3672e-02, -4.1199e-03,
             -3.2959e-03,  1.2634e-02, -5.8350e-02,  3.1494e-02, -1.4258e-01, -1.1182e-01, -1.8164e-01, -9.8877e-03,
             -1.8997e-03, -1.2158e-01, -8.3984e-02,  1.9653e-02, -2.0385e-05,  1.2054e-03, -3.7766e-04, -5.5908e-02,
             -1.3867e-01,  3.0975e-03,  5.7983e-04,  6.8359e-02,  1.0315e-02,  3.3203e-01, -1.5793e-03,  1.0757e-03,
             -9.8267e-03, -7.9956e-03,  1.2500e-01,  5.8746e-04, -3.3447e-02,  1.5564e-03,  1.2988e-01, -1.1328e-01,
             -5.3955e-02, -1.1523e-01, -9.7046e-03,  2.3071e-02,  1.3428e-02, -2.9663e-02,  1.4420e-03,  2.4567e-03,
              3.6926e-03,  8.0078e-02,  2.8419e-04, -2.4219e-01, -1.6479e-02, -2.4414e-02,  1.9455e-03,  1.5945e-03,
             -7.1777e-02,  8.1543e-02, -7.7148e-02,  1.6357e-02, -1.2109e-01,  3.1250e-02, -9.6680e-02, -7.0190e-03,
              4.4556e-03, -1.2207e-01, -1.7548e-03,  1.6022e-04,  5.4932e-02, -8.6426e-02,  4.8447e-04,  5.9814e-02,
             -6.2500e-02, -3.9291e-04,  1.3275e-03, -8.6426e-02,  5.9891e-04,  3.6865e-02,  1.5259e-02,  1.0303e-01,
             -9.2773e-02,  2.4780e-02,  8.0109e-05,  3.2654e-03, -6.8970e-03, -6.9885e-03, -5.3406e-03, -5.4932e-02,
              4.7302e-04, -3.1982e-02,  1.6937e-03,  1.0986e-02,  8.7738e-04,  4.3945e-02, -7.6660e-02,  1.1206e-04,
              2.5513e-02,  6.5918e-02,  1.0071e-02,  5.5075e-05, -1.5015e-02, -6.7749e-03, -2.9541e-02, -2.2339e-02,
              4.3945e-02,  1.6992e-01, -2.3937e-04,  5.7129e-02,  2.5928e-06,  2.2949e-02,  1.8311e-02, -2.8564e-02,
             -7.8613e-02,  5.0049e-02,  5.0537e-02,  5.2344e-01,  1.6235e-02, -3.6774e-03,  3.7891e-01,  2.5269e-02,
              2.7657e-05, -2.8038e-04, -1.7676e-01, -3.5400e-02,  7.2021e-03,  4.5471e-03,  2.6978e-02,  1.9379e-03,
              2.6245e-02,  4.3335e-03,  1.8158e-03,  4.3701e-02,  1.4400e-04,  3.7402e-06, -6.6757e-04, -6.9336e-02,
              1.2939e-02, -2.1118e-02, -6.2988e-02,  6.9427e-04, -1.8005e-03,  2.0410e-01, -1.1658e-02,  3.3691e-02,
              1.0681e-03,  3.4571e-05, -5.6396e-02, -2.8198e-02,  5.2734e-02, -3.3188e-04,  1.7676e-01,  1.5182e-03,
              3.4523e-04, -7.2861e-04, -8.5449e-04, -2.9688e-01,  1.5869e-03,  6.6528e-03, -4.8828e-02,  6.4453e-02,
              8.5449e-03, -4.2725e-04,  5.7617e-02, -1.9897e-02,  9.1309e-02,  1.9360e-04, -3.1738e-02,  1.1841e-02,
              1.3855e-02, -2.1606e-02, -2.3145e-01, -4.2188e-01, -1.6968e-02, -9.2773e-02, -3.6523e-01,  3.2902e-05,
              4.1771e-04,  2.8038e-04, -4.6692e-03,  2.4048e-02,  2.0386e-02,  8.5449e-03, -6.3477e-02,  1.0010e-02,
              3.9062e-02, -8.7357e-04, -3.9307e-02,  2.5024e-02, -1.0071e-02, -1.7776e-03,  1.2988e-01, -2.4292e-02,
             -2.9945e-04,  4.3869e-04, -3.0136e-04,  7.9346e-03,  6.6406e-02,  4.3640e-03,  5.2344e-01,  3.5889e-02,
              1.1621e-01, -2.7466e-03,  6.1035e-05, -4.3335e-03,  1.8652e-01,  1.0400e-01, -3.1471e-04,  1.5820e-01,
              1.2500e-01,  1.5793e-03, -2.6367e-01, -2.7418e-06,  8.3923e-05,  6.6528e-03, -2.9297e-01, -4.1992e-02,
             -3.2336e-06, -2.4780e-02, -3.9101e-05, -3.3008e-01,  4.7607e-02, -5.5313e-04,  7.4463e-03, -2.2125e-03,
              2.0630e-02,  3.2959e-02, -3.5400e-02, -7.6172e-02,  5.9307e-06, -3.9551e-02, -8.3496e-02,  9.1797e-01,
              5.5420e-02, -1.8677e-02, -3.1586e-03,  4.2236e-02,  1.2268e-02,  1.4062e-01, -7.2754e-02,  2.7847e-04,
             -6.5918e-02,  1.0205e-01,  2.1172e-04,  1.1749e-03,  1.8555e-01,  1.2793e-01, -6.7578e-01,  7.6675e-04,
              6.0303e-02,  2.0752e-03, -1.1621e-01,  3.3447e-02,  1.9434e-01, -5.0735e-04, -4.3457e-02, -2.1606e-02,
              4.5898e-02, -5.8350e-02,  4.5166e-02,  1.0681e-03, -1.7548e-03,  9.6191e-02, -2.0294e-03, -2.9419e-02,
             -3.6049e-04,  1.4191e-03, -8.0566e-03,  7.0953e-04,  1.4801e-03, -1.0071e-02,  3.0664e-01,  2.7100e-02,
             -8.5831e-04,  1.8164e-01,  1.3672e-01, -2.5177e-03,  1.1963e-02, -1.1035e-01,  5.2643e-04,  7.7057e-04,
             -7.3242e-02,  3.0273e-02, -1.5137e-01,  3.1738e-02, -4.2725e-03, -1.3574e-01, -4.4441e-04,  7.1716e-03,
             -2.0123e-04,  9.1553e-03, -6.7188e-01,  3.0823e-03,  3.2501e-03, -1.1426e-01,  2.9373e-04,  5.4626e-03,
             -6.2256e-02,  1.6235e-02,  8.8379e-02,  5.9082e-02,  4.9805e-02, -1.1484e+00,  9.5215e-02,  9.0332e-03,
              1.0156e-01,  7.5684e-02,  4.0625e-01,  2.3174e-04, -6.2866e-03,  3.1006e-02, -2.8125e-01,  4.4141e-01,
              3.2227e-02, -9.4223e-04,  1.9531e-01, -3.1738e-02, -9.5749e-04, -2.7100e-02, -1.3611e-02, -5.1117e-04,
              1.0620e-02, -4.2725e-03,  7.0312e-02, -7.2021e-03, -1.4160e-01,  3.5553e-03,  8.9645e-04, -6.3705e-04,
              7.8125e-02,  4.1962e-05,  2.1648e-04,  1.5869e-03, -5.3167e-05, -1.4038e-03,  1.4752e-06, -4.6730e-05,
              7.5195e-02,  2.0905e-03,  4.4189e-02, -1.1035e-01,  6.4850e-05,  6.7871e-02,  1.8066e-01,  3.8086e-02,
             -8.6914e-02,  1.8555e-02,  5.8350e-02, -5.8365e-04, -2.7084e-04, -9.3994e-03, -9.0820e-02,  4.0771e-02,
             -2.3556e-04, -1.2500e-01, -2.3560e-02,  4.2969e-02, -6.6406e-02, -2.1839e-04,  3.9062e-02, -1.0889e-01,
             -3.4180e-02, -5.1880e-04, -1.3867e-01,  1.3086e-01, -5.7983e-04,  7.8613e-02,  2.6367e-01, -2.4609e-01,
              3.2959e-02,  6.9336e-02,  3.4766e-01, -1.2512e-03,  6.3324e-04, -2.0752e-02,  1.7166e-05,  8.5449e-02,
              7.5340e-05,  4.5410e-02, -6.2988e-02,  7.3242e-02, -1.3123e-03,  4.3945e-02, -9.8145e-02,  1.6992e-01,
              8.8501e-03, -1.6708e-03, -8.6426e-02, -1.5564e-02, -2.1118e-02,  1.5527e-01, -6.4697e-03, -8.3618e-03,
              1.0824e-04,  2.8198e-02, -1.4404e-02,  7.0190e-04, -1.3733e-04,  3.2806e-04,  4.2534e-04,  1.8945e-01,
              3.5645e-02,  7.8201e-04, -2.1606e-02, -1.2109e-01,  5.5078e-01, -9.7046e-03, -3.4332e-04,  5.7129e-02,
             -9.0027e-04, -4.2236e-02,  1.4844e-01, -1.0010e-01, -9.9609e-02, -3.3691e-02, -3.1471e-04,  4.4556e-03,
              7.7637e-02, -5.5664e-02, -2.8491e-05,  1.9775e-02,  7.3730e-02, -1.0437e-02,  5.1880e-04,  8.6914e-02,
             -2.3842e-05, -1.8768e-03, -2.2339e-02, -8.2397e-03, -3.9062e-02, -2.7537e-05, -2.5757e-02,  9.6191e-02,
              3.6316e-03,  1.4973e-04, -6.2500e-02, -6.8188e-05, -1.9434e-01, -2.9053e-02,  1.7700e-03, -6.1035e-04,
             -4.0039e-02,  9.1309e-02, -1.2207e-02,  7.6172e-02,  4.1992e-02,  1.2112e-04, -1.9684e-03, -1.7334e-02,
             -1.4465e-02,  5.8289e-03,  5.0781e-02,  1.5527e-01,  5.9509e-03, -2.9785e-02, -2.7954e-02, -2.8038e-04,
             -2.5146e-02, -1.1353e-02, -1.8066e-01,  4.1504e-02,  4.0039e-02, -9.6798e-05,  7.6904e-03, -2.7954e-02,
             -5.2979e-02, -1.0395e-04,  1.7090e-02,  4.7852e-01, -1.0193e-02, -1.2634e-02,  3.3936e-02, -1.3672e-01,
              8.1956e-07, -1.0193e-02,  1.3489e-02, -1.6797e-01,  3.0396e-02,  4.5654e-02,  2.8564e-02, -9.4727e-02,
              9.5215e-02,  6.6376e-04, -3.0365e-03, -2.5635e-03,  4.4922e-02,  9.8877e-03, -3.0762e-02, -4.6997e-03,
              3.2422e-01,  3.4180e-02,  6.1523e-02, -3.6163e-03, -5.2002e-02,  5.6885e-02,  4.1199e-04,  2.2507e-04,
              4.9805e-02,  7.3433e-05,  5.9128e-04, -4.8523e-03,  2.3365e-05, -2.8809e-02,  2.7275e-04,  9.3262e-02,
             -5.2246e-02, -4.6492e-05, -1.0620e-02,  4.7302e-04, -2.2697e-04,  3.0762e-02, -4.1992e-02, -4.7119e-02,
             -9.3460e-04, -2.7771e-03,  6.1512e-05,  1.4941e-01, -1.1230e-01,  1.3580e-03, -3.8818e-02, -2.8801e-04,
             -1.1349e-04, -2.0386e-02, -2.7222e-02, -1.2354e-01,  1.7456e-02,  1.6327e-03, -4.0283e-02,  1.6212e-04,
             -1.2451e-01,  2.9373e-04, -1.8457e-01, -4.6921e-04, -6.6895e-02, -2.1777e-01,  1.7969e-01,  3.7994e-03,
              6.9141e-05,  8.5449e-02,  5.3467e-02,  8.1062e-05, -1.0254e-02, -2.4536e-02,  1.4343e-02, -1.5015e-02,
             -9.9609e-02, -1.3477e-01,  2.8229e-03, -1.1328e-01,  6.2466e-05,  8.6212e-04,  2.5146e-02,  4.3154e-05,
              2.9419e-02, -5.8594e-02,  1.2695e-01,  1.5625e-01, -6.4453e-02, -5.4321e-03, -1.0681e-02, -8.1055e-02,
             -1.6016e-01,  1.5259e-02,  7.5195e-02, -2.0599e-03,  8.2031e-02,  2.9907e-02,  2.0313e-04, -1.8358e-05,
              5.3467e-02, -1.1816e-01,  1.3580e-03, -2.1240e-02,  8.0078e-02,  1.2112e-04,  3.2617e-01,  8.9645e-05,
              4.2725e-04, -3.8605e-03, -9.8267e-03, -5.0781e-02, -2.0410e-01,  2.9297e-02,  1.8066e-01,  1.2061e-01,
              6.3477e-02,  1.4160e-01, -3.7598e-02, -1.0059e-01,  1.1978e-03,  5.7373e-02, -1.5640e-03, -1.6113e-02,
             -1.2146e-02,  1.1621e-01, -2.2430e-03, -8.4305e-04,  2.4170e-02,  1.2875e-04,  9.3750e-02,  9.5367e-04,
              5.4932e-03, -2.1210e-03, -1.2598e-01, -7.9155e-05,  1.7700e-03, -4.9316e-02, -6.1523e-02, -1.3809e-03,
             -5.0293e-02,  1.5076e-02, -6.9824e-02, -7.7820e-03, -8.9355e-02, -1.9043e-01,  1.2451e-01,  1.3379e-01,
             -4.4434e-02,  1.7242e-03, -6.5002e-03,  4.0627e-04,  1.9312e-05, -3.4668e-02, -2.0020e-01,  4.4434e-02,
              4.9438e-03,  4.5654e-02, -1.0605e-03, -2.2705e-02, -4.1748e-02, -1.3184e-02,  3.5645e-02,  2.6512e-04,
             -4.3297e-04, -1.3281e-01, -1.1658e-02, -2.6172e-01,  4.9316e-02,  1.3906e+00, -1.9629e-01, -5.3101e-03,
             -5.3955e-02,  8.2397e-04, -2.5024e-02,  2.4033e-04, -3.6011e-03,  7.6172e-02, -2.7344e-02,  5.6763e-03,
             -2.3145e-01,  1.3000e-02, -6.1035e-04, -5.0781e-01,  1.4877e-04, -1.9165e-02,  3.9307e-02,  1.3281e-01,
              4.2725e-02, -2.0123e-04, -1.0449e-01,  1.7212e-02,  2.6978e-02, -6.0059e-02,  2.0508e-01, -5.3467e-02,
             -1.9150e-03,  1.0834e-03,  5.7617e-02, -2.4414e-04,  1.3638e-04,  5.5313e-04,  3.7354e-02, -2.8809e-02,
             -8.4961e-02, -1.6403e-03, -1.0986e-03,  8.6914e-02, -1.9836e-03, -2.2278e-03,  4.2969e-02, -6.6895e-02,
             -8.2031e-02,  4.8584e-02,  2.1484e-02, -7.9590e-02, -1.0681e-03, -4.2969e-02,  9.3842e-04, -8.9844e-02,
              4.0527e-02, -2.2949e-01,  5.4169e-04,  2.5146e-02,  2.6093e-03, -3.9307e-02,  3.3951e-04,  1.1778e-04,
             -2.7084e-04, -4.1748e-02,  6.5308e-03,  2.9297e-01,  1.1597e-02, -1.2012e-01, -6.1768e-02, -2.4780e-02,
              1.3828e-04, -1.2891e-01, -2.8419e-04,  9.1309e-02,  8.2520e-02, -1.0193e-02, -8.9355e-02, -5.5908e-02,
             -6.7383e-02, -3.4570e-01,  1.2061e-01,  3.3203e-01, -2.6855e-03, -1.8848e-01, -1.3199e-03, -3.3140e-05,
             -3.2471e-02,  2.9492e-01, -6.0059e-02,  2.5000e-01,  1.0156e-01,  8.7357e-04,  3.1738e-02, -1.7969e-01,
              3.0640e-02,  1.2891e-01,  9.3262e-02, -7.6599e-03, -9.8145e-02,  1.2970e-04,  1.1536e-02, -9.2316e-04,
             -8.1543e-02, -1.0071e-03, -6.0059e-02, -7.1106e-03, -3.0859e-01, -3.4332e-04, -4.1748e-02, -9.1553e-03,
             -3.6011e-03,  3.5645e-02, -6.3324e-04, -1.1963e-02,  1.4160e-01, -4.0527e-02, -1.4771e-02, -3.3112e-03,
              1.0437e-02, -1.7334e-02, -3.6469e-03,  3.0518e-04,  6.1035e-03,  4.5776e-03, -2.3071e-02,  2.6855e-02,
              9.5367e-04, -3.8477e-01,  5.1117e-04,  5.9128e-04, -1.3550e-02, -3.1982e-02, -8.4961e-02,  3.3691e-02,
             -2.5635e-02, -1.8457e-01,  7.4707e-02, -1.0791e-01,  9.4727e-02,  8.0566e-02, -4.4727e-01,  2.0313e-04,
              9.3750e-02,  4.8429e-07,  2.3956e-03, -2.3956e-03,  6.6406e-01, -1.2988e-01, -2.3315e-02, -6.6895e-02,
              1.6504e-01, -5.9326e-02, -3.0640e-02,  5.4443e-02,  2.1118e-02, -6.3477e-03,  1.0620e-02,  1.4648e-02,
              3.4124e-06,  1.5820e-01, -3.8086e-02,  2.6703e-04,  1.3065e-04,  4.5410e-02, -8.9407e-06, -4.9591e-04,
              1.1826e-04,  1.1621e-01,  4.9561e-02, -2.5757e-02, -3.3951e-04, -1.7090e-02,  3.7109e-02,  5.4321e-03,
              1.0742e-02, -4.2578e-01, -1.0193e-02,  6.4850e-05,  5.8889e-05,  4.7922e-05,  6.5918e-02, -9.6893e-04,
              2.2949e-02, -3.9673e-04, -1.9836e-03, -8.3496e-02, -7.0312e-02,  5.5313e-04,  3.6469e-03,  6.0272e-04,
             -1.9336e-01, -6.5002e-03,  7.4219e-02,  1.3281e-01,  4.4678e-02, -4.0283e-02,  7.3624e-04, -4.3640e-03,
             -6.1523e-02,  1.4725e-03, -8.9407e-06, -4.9591e-04, -1.6309e-01, -1.5076e-02, -1.0071e-03, -7.4219e-02,
              3.9307e-02, -2.5749e-04, -1.6357e-02, -3.0762e-02, -3.7598e-02,  9.5825e-03, -5.4932e-02, -1.3428e-02,
             -1.5793e-03, -1.6308e-04,  1.3867e-01, -2.0264e-02,  3.2663e-05, -7.2956e-05,  2.8801e-04, -9.6436e-03,
              3.0884e-02, -1.8457e-01,  2.6245e-02,  2.1362e-03,  2.3560e-02,  1.5320e-02,  1.4954e-02, -2.4902e-02,
              1.4954e-03, -3.6865e-02,  1.6327e-03, -1.5430e-01, -1.1523e-01,  2.9907e-02,  2.4319e-04,  8.8379e-02,
             -1.2598e-01,  6.0791e-02,  2.4048e-02,  1.1523e-01, -7.1526e-05, -4.2578e-01, -3.5742e-01,  3.8477e-01,
              3.7842e-03,  7.3242e-02,  7.5684e-02,  2.5558e-04,  6.2180e-04,  3.3398e-01, -2.5749e-04,  1.7676e-01,
              1.2512e-03,  5.9509e-03,  5.8899e-03,  1.2512e-03, -8.5068e-04, -3.2196e-03, -6.6406e-02, -1.6309e-01,
              1.1816e-01,  2.1240e-02, -1.1536e-02, -3.4424e-02,  4.1016e-02, -7.2632e-03,  1.0498e-01,  6.8188e-05,
             -3.1891e-03, -3.1891e-03,  5.5908e-02, -5.5908e-02, -8.9355e-02,  5.7983e-04, -4.8637e-04,  1.3574e-01,
              9.9121e-02,  1.0376e-03, -5.6076e-04,  8.2397e-03, -8.8379e-02, -1.8433e-02, -6.6406e-02, -9.0332e-03,
             -9.4604e-03,  5.6267e-05,  7.7148e-02,  8.8501e-03, -9.0820e-02, -1.2390e-02,  9.1309e-02,  5.2979e-02,
             -1.0791e-01,  5.2490e-02, -1.2305e-01, -4.5166e-02, -3.1738e-02,  4.4678e-02,  1.0192e-05, -8.4961e-02,
             -1.0059e-01, -8.0872e-04,  8.4473e-02,  1.2360e-03,  3.8574e-02, -4.5166e-03, -4.6289e-01,  4.1260e-02,
              4.8447e-04, -5.1270e-02, -8.6914e-02,  3.6812e-04, -3.0136e-04, -3.4485e-03,  6.7383e-02, -2.0020e-01,
              4.0039e-02, -8.9355e-02, -2.8229e-03,  8.7402e-02, -3.2715e-02, -6.2500e-02, -1.8406e-04, -7.1777e-02,
             -4.7913e-03, -6.6757e-04,  5.1514e-02, -5.4550e-04,  3.7598e-02, -6.2256e-02, -6.7383e-02, -1.0547e-01,
             -1.0840e-01, -4.3213e-02,  2.3071e-02,  3.6621e-02,  2.9907e-02,  4.7852e-02,  8.6426e-02,  9.1553e-04,
              2.1777e-01,  7.1335e-04,  2.8564e-02,  1.3123e-02, -5.3955e-02, -6.8665e-03,  5.4688e-02,  3.3691e-02,
              9.5703e-02,  1.2988e-01,  6.2012e-02, -3.5248e-03, -1.7700e-03, -6.5918e-03, -1.5747e-02,  1.8120e-04,
              3.5156e-02,  3.3936e-02,  1.1768e-01, -1.8652e-01, -2.1606e-02, -3.7695e-01, -1.8848e-01, -3.6865e-02,
              3.0756e-05,  3.0396e-02, -1.8234e-03, -2.9755e-03, -3.7956e-04,  1.2634e-02,  2.5195e-01, -1.8281e+00,
              2.1387e-01, -4.9805e-02, -6.5918e-02, -1.2988e-01,  2.5391e-02, -7.6294e-04,  3.2471e-02, -3.4180e-02,
             -3.5248e-03, -5.6458e-03, -7.8125e-02, -1.4258e-01,  3.7598e-02, -8.8501e-03, -9.4238e-02, -5.7617e-02,
             -2.7588e-02, -2.6001e-02, -3.1128e-02,  1.3916e-02, -4.6921e-04,  2.3438e-02, -4.6387e-03, -3.9482e-04,
              9.4604e-03,  9.1309e-02, -1.1780e-02, -2.9541e-02, -5.5664e-02,  5.1514e-02,  9.3262e-02,  1.4282e-02,
              1.1169e-02,  2.3842e-04,  2.5635e-02, -1.0059e-01, -1.7014e-03, -8.6426e-02,  2.4080e-05, -1.7285e-01,
             -5.9814e-02,  3.4424e-02, -1.2934e-05, -1.2207e-01,  2.1484e-02, -1.3351e-04, -2.0142e-03,  8.3496e-02,
              4.4434e-02,  3.0975e-03, -1.7578e-02,  3.9101e-04, -1.3885e-03,  4.1992e-02,  9.7046e-03,  1.2939e-02,
              5.9082e-02, -2.0996e-02, -4.0436e-04, -1.0010e-02,  2.2217e-02, -1.1182e-01, -1.1969e-04,  3.2471e-02,
             -5.2261e-04,  6.4453e-02, -1.6093e-05,  2.9062e+00, -1.1063e-03, -1.2939e-02, -4.0283e-02,  1.6016e-01,
             -2.8992e-04, -6.2256e-02,  1.8188e-02,  2.0447e-03,  1.4420e-03, -1.7700e-02, -1.1621e-01, -5.8746e-04,
             -9.2773e-02,  1.5430e-01, -7.9590e-02,  7.2754e-02, -1.0803e-02, -9.4727e-02, -7.6172e-02,  2.0599e-03,
             -4.4922e-02,  5.3711e-03,  1.8555e-02, -1.4355e-01,  8.3984e-02, -4.8256e-04,  1.8164e-01,  7.3242e-04,
             -2.5988e-05,  8.2779e-04,  2.8809e-02,  4.8096e-02,  7.6172e-02, -1.3379e-01,  3.1090e-04,  3.0029e-02,
             -7.0572e-04, -2.2278e-03,  1.3965e-01, -1.1035e-01,  2.2411e-04, -2.1118e-02, -5.8594e-02, -1.8552e-06,
             -6.2988e-02, -1.4186e-05, -4.0771e-02, -1.9379e-03, -6.7139e-03, -4.2114e-03, -1.5497e-05,  5.5420e-02,
              5.4932e-04, -3.3447e-02, -1.5527e-01, -5.2185e-03,  2.2888e-03,  3.0060e-03,  8.5449e-04, -1.6406e-01,
             -7.0496e-03,  7.1335e-04, -3.6133e-01,  8.8501e-03, -2.4796e-04,  2.0218e-04,  1.9336e-01, -1.9775e-02,
             -1.9165e-02,  3.0060e-03, -2.1240e-02,  2.2070e-01,  3.0762e-02, -4.8065e-04, -5.5908e-02,  1.2817e-03,
              4.1260e-02,  2.6131e-04,  4.4434e-02,  2.5000e-01, -3.5938e-01, -8.3618e-03, -3.0396e-02, -2.2984e-04,
             -8.4229e-03,  5.2643e-04, -1.6689e-05, -9.2316e-04, -3.9673e-03,  1.1414e-02, -1.7456e-02, -7.4387e-05,
              7.7637e-02, -3.6719e-01,  7.3910e-05,  1.1673e-03,  5.9570e-02,  1.1414e-02, -5.1953e-01, -5.0735e-04,
             -3.2422e-01,  9.1934e-04,  1.7212e-02,  6.1523e-02, -1.0010e-02,  4.5898e-02,  5.9814e-03,  8.9264e-04,
             -3.0762e-02,  7.4707e-02, -4.0771e-02, -4.5395e-04,  5.4169e-04, -6.7871e-02,  4.3945e-02,  6.1768e-02,
             -7.9956e-03,  8.7261e-05,  3.2959e-02,  1.4453e-01, -5.8899e-03,  9.4604e-04, -2.5269e-02, -5.0781e-02,
             -1.5076e-02, -1.5076e-02, -2.2070e-01,  2.4170e-02,  1.6479e-02, -3.8574e-02, -4.7607e-02, -3.0469e-01,
              6.9885e-03, -7.2861e-04, -2.8839e-03,  7.0801e-02, -9.9182e-04,  3.7842e-03,  0.0000e+00, -3.9258e-01,
              1.3086e-01,  1.8359e-01,  6.4941e-02,  1.2894e-03,  1.1035e-01,  7.6172e-02, -5.7678e-03, -5.5420e-02,
              3.2234e-04, -3.6328e-01, -8.6975e-04,  1.1780e-02, -2.4796e-04,  1.0986e-01, -8.7261e-05, -1.7624e-03,
             -4.4922e-02, -4.6387e-02, -4.6997e-03, -1.9629e-01,  8.3008e-03,  1.5723e-01, -8.1055e-02,  1.6113e-02,
              8.9844e-02, -4.9133e-03,  4.0245e-04,  1.6093e-05,  5.2490e-02, -1.0864e-02, -2.7539e-01,  1.8835e-05,
             -1.9989e-03,  3.0029e-02, -1.5332e-01,  2.8419e-04, -2.5787e-03, -2.5513e-02,  2.4986e-04, -2.5391e-02,
             -6.8848e-02,  6.5430e-02,  5.7861e-02, -1.3611e-02,  2.0020e-01,  1.2634e-02, -1.8555e-01,  7.0312e-02,
              4.0039e-01, -3.5645e-02,  2.7084e-04,  5.5420e-02, -9.8877e-03, -4.4250e-03, -1.0620e-02, -5.0781e-02,
              6.5231e-04,  5.6458e-03,  3.7598e-02,  1.1230e-01,  2.9175e-02, -5.3711e-02, -3.3936e-02, -3.8338e-04,
             -1.1349e-04, -2.1118e-02, -5.8350e-02,  1.5234e-01,  3.0708e-04, -1.5991e-02,  6.1035e-02, -1.7188e-01,
             -7.8964e-04,  2.8839e-03,  1.3611e-02, -4.1504e-02,  4.8828e-02,  9.7656e-02,  5.3711e-02,  3.9551e-02,
              9.3262e-02, -6.2561e-03,  2.0386e-02,  1.6403e-04,  2.3560e-02,  4.3701e-02, -2.6855e-02, -1.1444e-03,
              6.9275e-03,  8.4305e-04,  1.6479e-02, -2.4902e-02, -8.1055e-02, -4.6143e-02, -6.5918e-03, -3.4668e-02,
             -7.8125e-02, -9.1171e-04, -4.3457e-02,  2.0504e-04,  7.4707e-02,  4.6387e-02,  1.7700e-02,  1.5723e-01,
              5.9509e-03, -9.7656e-02,  4.0283e-02,  1.9360e-04, -9.8145e-02,  8.4639e-06,  6.5918e-02, -2.0752e-02,
             -6.5613e-03,  1.2305e-01,  6.5918e-02, -2.5757e-02, -1.7738e-04, -2.4902e-01,  1.6880e-04,  3.1281e-04,
             -2.3047e-01, -1.4099e-02, -2.5024e-02, -4.4861e-03, -1.0071e-03,  3.2959e-03, -4.3457e-02,  7.3730e-02,
             -1.8692e-04, -2.4223e-04, -7.1289e-02, -4.9174e-07,  3.7193e-05,  6.7871e-02,  1.1520e-03, -5.5664e-02,
              1.0824e-04, -1.8120e-04,  8.0078e-02, -1.2451e-02,  7.7148e-02,  3.0273e-02, -7.7438e-04, -2.1057e-03,
             -7.0190e-04,  6.0730e-03,  2.2852e-01,  4.1260e-02,  1.9653e-02, -2.5940e-03, -5.1880e-03, -1.1621e-01,
             -4.4141e-01,  3.1738e-02, -9.3994e-03,  1.9043e-02,  1.2741e-03,  1.2988e-01, -9.8267e-03, -6.4850e-05,
              3.8330e-02,  6.8054e-03, -3.0975e-03,  5.2979e-02,  3.5645e-02, -1.4771e-02,  1.7452e-04, -8.2397e-03,
             -1.6602e-01,  2.0385e-05,  2.5177e-04, -5.4598e-05, -1.8750e-01, -3.6621e-02,  2.7832e-02, -4.8065e-04,
              6.1989e-05,  4.1809e-03, -3.2227e-01,  5.6885e-02, -4.3030e-03, -6.2943e-05,  1.4746e-01, -2.1191e-01,
             -1.5717e-03, -1.7776e-03,  1.1426e-01,  9.9487e-03,  3.7109e-02,  3.6328e-01,  4.7461e-01,  7.0572e-05,
              2.7710e-02,  7.4768e-04,  2.7710e-02, -5.2002e-02,  4.1809e-03, -5.8838e-02,  1.6174e-03, -8.9264e-04,
              2.4292e-02, -5.0354e-04,  4.2480e-02,  1.4191e-03,  1.6235e-02, -2.9297e-02,  2.6733e-02,  2.7954e-02,
             -6.1951e-03, -9.1553e-03,  1.2024e-02,  3.9551e-02, -2.7710e-02,  5.2490e-02, -7.7820e-04, -3.0060e-03,
             -1.0010e-02, -5.9128e-04, -1.3379e-01, -3.7956e-04,  1.5869e-02,  2.8320e-02, -3.2349e-03,  4.2534e-04,
              1.0312e+00, -2.2656e-01, -6.2988e-02, -5.6152e-02,  1.3281e-01,  7.9346e-03,  2.4780e-02,  9.7156e-06,
              2.0996e-01, -2.9688e-01,  6.5430e-02,  2.9182e-04, -2.0215e-01,  3.7354e-02, -1.0596e-01,  8.2520e-02,
             -8.7261e-05, -2.3315e-02, -7.4219e-02,  8.7261e-05, -5.3644e-06, -1.8120e-04,  2.1240e-02,  1.2207e-02,
              0.0000e+00,  6.3477e-02, -6.4453e-02, -5.3467e-02, -1.5564e-03, -4.1504e-02,  2.5391e-02,  2.5177e-03,
             -2.8372e-05,  2.1118e-02,  1.5820e-01, -1.3428e-02,  1.5198e-02,  1.0109e-04, -3.8330e-02, -1.0681e-03,
             -2.6953e-01,  2.9297e-02,  6.5918e-03, -6.4453e-02, -2.9492e-01,  4.7112e-04, -4.1504e-03, -4.9805e-02,
             -2.7269e-06,  8.3496e-02, -2.5635e-03, -6.1279e-02,  3.0469e-01,  1.3672e-02, -6.9427e-04,  3.8086e-02,
              3.3875e-03, -2.9057e-06,  1.7776e-03,  2.6562e-01, -1.7262e-04,  3.7689e-03, -1.7188e-01,  8.0566e-03,
              8.7357e-04, -1.3542e-04, -5.5542e-03,  4.2725e-02, -1.1670e-01,  6.2466e-05, -6.3782e-03,  5.9082e-02,
             -2.5928e-06,  9.9182e-04,  3.1055e-01, -1.2256e-01,  2.2888e-03,  5.1514e-02,  4.0039e-02, -3.0884e-02,
             -2.9492e-01,  2.7618e-03,  1.3672e-01,  3.5095e-04, -1.5625e-02,  8.3542e-04,  1.3924e-04,  1.0938e-01,
              5.6885e-02,  7.8125e-03,  4.1199e-03,  0.0000e+00,  3.3140e-05, -5.6152e-03, -4.1992e-02, -3.7598e-02,
              9.1797e-02, -4.3869e-04,  1.2451e-01,  4.5471e-03, -1.9238e-01, -7.1716e-04,  1.7871e-01,  4.3750e-01,
              2.9907e-02,  2.8516e-01, -7.4387e-04,  3.7891e-01, -1.3733e-04,  1.1536e-02,  1.0864e-02,  1.0693e-01,
              5.2795e-03, -1.3000e-02, -6.6406e-02, -1.5106e-03, -7.8583e-04, -7.7148e-02,  4.1748e-02,  4.6631e-02,
             -1.5259e-04,  6.5430e-02, -5.6839e-04,  1.0254e-02,  1.8066e-01,  1.4282e-02,  3.8281e-01, -5.3787e-04,
              4.7684e-05, -1.0449e-01, -1.1865e-01, -6.6223e-03,  1.6479e-02,  4.8637e-04,  7.1411e-03,  2.2339e-02,
              8.4473e-02, -3.7354e-02,  1.6113e-02, -2.8372e-05, -6.7711e-05,  5.7861e-02,  1.5199e-06, -1.3367e-02,
             -1.0693e-01,  1.2695e-01,  9.2578e-01, -2.8839e-03, -7.5684e-02, -7.8964e-04,  6.1512e-05, -1.3965e-01,
              1.5723e-01, -6.7871e-02,  5.4016e-03, -9.7046e-03, -3.2616e-04, -8.3496e-02,  1.3199e-03,  3.2471e-02,
             -7.9727e-04, -1.0400e-01,  1.1169e-02,  2.9175e-02,  6.0730e-03,  1.7853e-03,  4.0054e-04,  8.0585e-05,
             -1.4687e-04,  9.4604e-03, -4.4434e-02, -1.8234e-03, -5.5664e-02,  2.9663e-02, -2.7222e-02, -1.7548e-03,
             -1.1110e-04, -1.6602e-01, -1.5723e-01,  3.7384e-04,  3.8574e-02, -7.2754e-02,  9.3079e-04, -1.2146e-02,
              1.4746e-01, -1.9684e-03, -1.1621e-01, -7.5684e-02, -2.8516e-01,  1.7676e-01, -4.1504e-02, -8.2520e-02,
             -4.5703e-01, -1.1749e-03,  2.8198e-02, -1.2891e-01,  5.5695e-04,  1.2451e-02, -7.4768e-04, -1.4648e-01,
              1.7969e-01,  2.5940e-03,  1.3123e-03, -1.5869e-03,  1.3638e-04, -2.0703e-01, -1.3965e-01,  7.3730e-02,
              9.7168e-02, -9.4238e-02,  1.4210e-04, -1.5527e-01,  0.0000e+00,  1.0498e-02,  4.2915e-04,  4.4189e-02,
             -8.3008e-02, -3.2425e-04,  1.0156e-01,  2.0312e-01,  1.3256e-04,  1.1368e-03,  1.9360e-04, -5.4443e-02,
             -3.9577e-05,  8.3447e-05, -7.4387e-05, -2.6550e-03, -4.5166e-02,  1.3924e-04,  8.0566e-03,  1.0437e-02,
             -8.8379e-02,  5.6396e-02,  3.7842e-02,  1.4355e-01,  2.3193e-03,  1.0938e-01, -3.5645e-02, -1.4648e-01,
              1.4941e-01,  1.2207e-04, -1.5430e-01, -3.7994e-03,  5.8838e-02,  7.9346e-03, -1.0742e-01, -1.2666e-06,
              3.8910e-03,  4.1246e-05,  1.3184e-01,  9.8267e-03,  2.2125e-03, -5.8413e-05,  1.2695e-01, -3.5645e-02,
             -1.3855e-02, -6.4087e-03, -1.0803e-02, -5.7617e-02,  7.5073e-03,  2.5024e-02,  1.2402e-01,  1.7676e-01,
              2.0630e-02, -1.0864e-02,  4.2725e-03, -4.6875e-02, -1.8311e-02, -2.9373e-04, -3.0396e-02, -1.9653e-02],
            [-1.7700e-02,  8.9722e-03,  4.8096e-02, -7.8613e-02,  6.0059e-02,  4.5898e-01, -6.7139e-04, -6.7871e-02,
             -1.0547e-01, -1.4551e-01, -1.4587e-02, -1.0645e-01,  1.4746e-01,  1.0300e-03, -5.3406e-03, -5.5420e-02,
             -1.2598e-01, -1.3281e-01, -3.5095e-03, -1.0938e-01, -5.0537e-02,  4.4060e-04, -5.6458e-04, -9.0942e-03,
              8.6670e-03, -1.1169e-02, -6.7139e-04,  2.5177e-04, -7.7637e-02, -5.2261e-04,  1.3867e-01,  4.6387e-02,
              1.4709e-02, -1.2256e-01, -9.6680e-02,  1.5820e-01,  2.0752e-02, -3.0396e-02,  1.7700e-02, -1.2756e-02,
             -5.9814e-03,  1.2207e-04, -1.9287e-02, -1.2268e-02, -8.2779e-04, -1.5332e-01,  4.0894e-03, -2.3071e-02,
              6.3705e-04,  1.9653e-02,  1.8433e-02, -8.8882e-04, -1.3062e-02,  5.6152e-03, -1.3447e-04, -8.6426e-02,
             -3.6430e-04, -5.1025e-02,  5.1498e-04, -2.0996e-02,  1.6556e-03,  3.9864e-04,  3.6133e-02,  1.1841e-02,
             -3.3722e-03, -3.5706e-03,  6.0303e-02,  7.4707e-02,  1.1133e-01, -3.9673e-04, -5.7220e-04, -9.0027e-04,
              1.6357e-02,  7.7438e-04, -6.5231e-04,  4.0894e-03,  3.2227e-02, -7.8125e-02,  5.5176e-02, -1.4099e-02,
             -8.9264e-04,  7.2754e-02, -2.0020e-02,  1.0315e-02, -1.7524e-05,  4.7493e-04, -8.1787e-03, -1.9775e-02,
              5.0781e-02,  9.3384e-03,  4.1260e-02, -4.8340e-02, -4.1992e-02,  1.1572e-01,  1.5527e-01, -1.2354e-01,
             -8.8501e-03, -1.5320e-02, -3.4943e-03,  1.1063e-03, -3.2715e-02, -1.6992e-01, -1.7090e-02, -4.2725e-02,
              9.4223e-04, -4.3945e-02, -2.0215e-01, -3.1055e-01, -9.0942e-03,  2.4658e-02, -1.1536e-02,  5.9128e-04,
              8.4473e-02, -1.1780e-02, -1.2695e-01, -7.1716e-03,  5.0781e-02, -7.0312e-02, -2.1606e-02,  1.8311e-02,
             -3.2227e-02, -5.4932e-02,  5.8105e-02,  5.8594e-02, -1.6797e-01, -3.6774e-03, -3.9307e-02,  7.7637e-02,
              8.9355e-02,  8.0566e-02, -3.4180e-02,  1.8188e-02,  4.0293e-05,  9.0332e-02,  8.6426e-02,  2.0386e-02,
             -3.4912e-02,  3.7842e-02, -2.8809e-02,  1.8692e-03, -3.5858e-04,  1.1597e-02,  1.2493e-04, -4.2343e-04,
             -2.0996e-01,  4.6082e-03, -1.4465e-02, -1.1328e-01, -5.7220e-04, -2.7222e-02, -4.2188e-01, -3.4809e-05,
             -1.5182e-03, -2.5879e-02,  5.5908e-02,  4.6015e-05, -1.9264e-04, -2.6001e-02,  2.8534e-03,  7.6172e-02,
             -3.6621e-02, -9.8633e-02,  5.0659e-03, -7.5684e-02,  3.7109e-02, -4.6484e-01,  3.7842e-02,  3.1494e-02,
              1.0071e-03,  3.6133e-02,  1.9238e-01, -3.3140e-05,  2.7466e-02, -8.7280e-03, -1.0315e-02,  5.1758e-02,
              1.3770e-01, -5.3467e-02, -2.3193e-02, -1.6098e-03,  4.0527e-02, -1.1063e-04,  3.4668e-02,  1.0840e-01,
             -7.2266e-02, -1.4877e-03, -1.2988e-01, -1.2695e-01,  6.0730e-03,  4.5262e-07, -6.5430e-02, -5.1758e-02,
              1.7480e-01, -1.3477e-01, -4.2480e-02, -2.4414e-02, -1.2207e-01,  4.4434e-02,  2.5513e-02,  3.3984e-01,
             -5.7602e-04,  8.3008e-03, -8.4877e-05, -8.0078e-02,  1.2158e-01,  2.6611e-02,  7.9956e-03,  8.2779e-04,
             -2.1973e-02, -8.5068e-04, -3.9062e-02, -1.7047e-05,  8.6975e-04,  1.5736e-04, -8.7891e-02,  2.4414e-03,
             -1.9169e-04, -3.6240e-04,  8.3496e-02, -2.9373e-04,  1.3275e-03, -3.4180e-03,  1.7762e-05,  2.9602e-03,
             -1.7334e-02, -2.7924e-03,  2.7588e-02,  1.5747e-02, -1.5106e-03,  2.1851e-02,  1.2451e-02,  8.7891e-02,
              1.1523e-01, -9.0332e-02,  1.1035e-01, -3.4637e-03,  1.1084e-01,  3.3569e-03, -1.9141e-01,  5.4550e-04,
              9.2773e-03,  2.0996e-02, -1.5831e-04,  3.4570e-01,  3.1738e-03,  1.4038e-02,  1.1444e-03,  1.1902e-03,
             -9.7168e-02, -1.0791e-01,  1.8463e-03,  2.4605e-04, -1.8457e-01,  2.5330e-03,  8.4473e-02, -4.4922e-02,
              1.7700e-02, -8.8120e-04, -7.9346e-04, -1.1328e-01,  2.6367e-02,  1.4587e-02,  3.5553e-03,  4.4434e-02,
              3.2617e-01,  2.4414e-03,  1.4709e-02,  2.3560e-02,  1.6724e-02,  8.1787e-03,  1.2988e-01,  4.7684e-04,
             -1.0938e-01, -9.6680e-02,  1.2793e-01,  7.3828e-01, -7.9590e-02,  1.0791e-01, -6.0272e-04,  1.9455e-03,
             -3.1128e-03,  3.7305e-01, -1.6602e-02,  1.0193e-02,  5.4932e-03,  6.5918e-02,  5.6076e-04,  4.3701e-02,
              1.0147e-03,  1.0547e-01,  1.2988e-01,  6.8848e-02, -1.0498e-01,  8.0109e-05, -1.4258e-01, -4.6631e-02,
              4.6484e-01, -5.7617e-02, -2.1484e-02, -3.4180e-02, -9.3937e-05,  1.8799e-02,  7.2266e-02, -2.0599e-03,
             -2.8381e-03,  2.3804e-03,  3.0398e-05, -1.3477e-01, -7.5195e-02,  3.3722e-03, -3.1250e-02, -3.4375e-01,
              2.5269e-02,  1.2598e-01,  1.6406e-01, -3.0273e-02, -9.6436e-03,  1.6357e-02,  1.3351e-04,  9.2163e-03,
             -3.2227e-01, -2.0447e-03,  7.8613e-02,  8.3984e-02,  5.0049e-02, -1.6098e-03, -7.4768e-03, -2.4536e-02,
             -9.0332e-03, -2.8320e-01, -4.7607e-03,  7.2266e-02, -1.9431e-05,  2.5146e-02,  5.1953e-01, -9.3750e-02,
             -5.5664e-02, -6.9336e-02, -5.0964e-03, -5.7861e-02, -9.2773e-02,  3.2959e-02, -1.2793e-01,  1.2131e-03,
             -3.7891e-01,  8.2520e-02, -9.5215e-02, -1.0223e-03,  1.0315e-02,  1.1865e-01, -4.1016e-02,  8.0078e-02,
             -1.0986e-01,  2.1362e-03,  6.6757e-04,  3.3447e-02, -8.3008e-02,  1.3638e-04, -1.5723e-01,  1.0920e-04,
             -8.9355e-02, -1.5442e-02, -1.3550e-02,  1.1328e-01, -1.2219e-05,  6.5918e-03, -1.0620e-02, -1.1169e-02,
             -2.0385e-05,  9.0790e-04, -9.7046e-03, -5.9204e-03, -4.1260e-02,  2.7100e-02, -4.1992e-02,  1.0376e-02,
             -2.9053e-02, -5.9891e-04, -9.3079e-04, -8.9111e-03, -1.0147e-03,  1.9165e-02, -2.7734e-01,  3.9864e-04,
             -5.7936e-05,  1.7776e-03, -4.5166e-03,  1.2061e-01,  1.2436e-03,  2.0885e-04, -1.4038e-03, -1.3367e-02,
             -1.8158e-03, -9.4727e-02, -1.0986e-01, -1.6211e-01,  1.3770e-01, -5.0049e-03, -3.6865e-02, -9.6680e-02,
             -1.1292e-02,  2.7771e-03, -3.8330e-02, -4.7493e-04, -6.9922e-01,  6.3965e-02, -4.7461e-01,  1.2054e-03,
              5.3955e-02,  1.2146e-02, -1.3794e-02,  6.3477e-02,  7.5684e-02, -5.7068e-03, -1.3123e-02,  1.3855e-02,
              8.8215e-06, -5.3711e-03, -7.6660e-02,  4.1992e-02, -1.2598e-01,  2.6001e-02,  4.8340e-02,  3.3691e-02,
             -1.3638e-04, -5.6396e-02, -2.8931e-02,  6.4453e-02, -2.4915e-05,  2.0447e-03,  3.4912e-02,  1.6327e-03,
              8.9844e-02, -2.8809e-02,  2.9175e-02, -4.9072e-02,  2.4438e-05, -2.2461e-01,  2.9564e-04,  1.3770e-01,
              3.3447e-02, -2.1973e-03, -8.4473e-02,  2.9297e-03, -4.1504e-02,  2.3193e-03,  2.5635e-02, -5.7068e-03,
              9.5703e-02, -9.4986e-04, -2.3438e-02, -1.3123e-03, -1.2939e-02, -1.2305e-01, -5.0049e-03,  3.8147e-06,
             -1.9238e-01,  1.1902e-03, -1.2268e-02, -2.0218e-04,  3.9673e-03, -7.2021e-03, -6.9824e-02,  3.4424e-02,
             -7.1716e-03, -6.3171e-03, -1.3428e-02,  6.5918e-02,  2.2852e-01, -6.7383e-02,  4.2969e-02, -4.1260e-02,
             -5.3711e-02,  1.3733e-03, -1.3867e-01, -3.2806e-04, -1.3916e-02, -6.7871e-02, -6.3477e-03,  9.4727e-02,
              2.3682e-02,  2.2461e-02,  2.0313e-04,  1.6846e-02, -2.3682e-02, -4.7852e-02,  1.3733e-03, -2.4536e-02,
             -1.2012e-01,  2.2266e-01,  5.1758e-02, -6.8848e-02,  7.9956e-03,  1.4465e-02, -5.7373e-02,  6.5430e-02,
              4.7070e-01, -3.1738e-02,  1.4648e-02,  0.0000e+00,  9.4238e-02, -3.9673e-04, -4.3945e-02,  5.9307e-06,
             -9.0820e-02, -9.3994e-03, -1.6699e-01, -5.5176e-02, -8.8501e-04,  8.7891e-02, -1.9653e-02, -3.8910e-03,
             -1.3281e-01,  8.3008e-02, -4.7684e-05,  3.3398e-01, -1.8799e-02,  2.2461e-02,  1.7480e-01,  2.0142e-03,
             -2.3804e-02, -1.2146e-02,  1.2793e-01,  1.2512e-02,  1.0529e-03,  1.0681e-02, -1.8555e-02,  1.3574e-01,
              1.3672e-01, -4.2419e-03,  1.6689e-04,  2.4658e-02,  1.4572e-03, -6.7139e-03, -4.7119e-02,  3.0365e-03,
              2.5368e-04,  1.7773e-01,  5.8350e-02, -2.1387e-01, -1.3770e-01, -2.2461e-02, -1.2054e-03, -1.1230e-01,
              5.6885e-02, -1.6699e-01,  4.0233e-06,  2.2583e-03,  2.0742e-05,  3.9307e-02, -6.1035e-02,  6.1719e-01,
              1.2131e-03,  2.2559e-01,  2.2095e-02, -6.3705e-04,  1.7319e-03,  9.0408e-04, -4.3213e-02, -2.5940e-04,
              1.0547e-01, -2.8516e-01, -8.0872e-04,  2.0312e-01,  2.8687e-02, -1.1597e-02,  1.1230e-01, -9.3460e-04,
              3.5524e-05,  1.6937e-03,  1.4526e-02, -1.2684e-04, -7.7820e-03,  3.8330e-02,  6.8359e-03,  1.4648e-02,
              2.2559e-01, -2.2705e-02, -6.2500e-02, -3.0136e-04, -9.7656e-04, -2.5195e-01, -4.9973e-04, -1.8799e-02,
              1.4453e-01, -8.1543e-02,  5.4932e-02, -7.1289e-02, -4.6082e-03, -9.2316e-04,  2.1744e-04,  5.5847e-03,
             -1.7212e-02,  2.1362e-03,  2.1851e-02,  1.5723e-01, -4.6539e-04,  7.4219e-02,  4.8828e-03, -1.0742e-01,
             -2.7771e-03,  3.2806e-04, -3.5667e-04, -1.4221e-02, -7.8125e-03,  4.2969e-02, -1.3046e-03, -5.5847e-03,
              3.8338e-04,  1.4551e-01,  3.8147e-04,  6.5625e-01,  7.9590e-02,  9.1309e-02,  5.0049e-03,  9.2163e-03,
             -1.1169e-02,  1.6357e-02, -2.0142e-02, -6.6895e-02,  3.6621e-02,  1.1536e-02,  4.0054e-04,  4.6997e-03,
              2.4536e-02, -1.9897e-02,  1.5625e-02, -1.2451e-02, -1.3733e-03,  1.7188e-01,  1.7853e-03, -6.5002e-03,
              4.9829e-05, -1.6602e-02, -9.3750e-02,  3.3789e-01, -7.5531e-04, -8.3923e-05, -2.6367e-02, -1.9922e-01,
             -5.0537e-02, -6.3324e-04, -1.9531e-03, -3.8086e-02,  4.2236e-02, -1.7090e-02,  4.2725e-04,  1.4587e-02,
              5.6152e-03, -1.9653e-02, -1.5378e-05, -2.1667e-03, -1.8501e-04,  5.9082e-02, -2.8992e-04, -1.7395e-03,
             -2.2949e-01, -1.4221e-02, -7.0801e-02,  1.0547e-01,  3.5156e-02,  3.7193e-05,  7.8125e-02, -7.7637e-02,
              1.0864e-02,  2.6367e-01,  1.0303e-01,  5.6744e-05, -1.9150e-03, -3.4904e-04,  3.6621e-02, -1.3924e-04,
             -2.2827e-02,  5.0000e-01, -3.3875e-03, -2.7657e-04, -1.0156e-01, -1.7578e-02, -1.8215e-04, -6.8359e-01,
             -1.4160e-02, -4.3701e-02,  8.7891e-02,  1.5625e-02,  1.2146e-02,  9.7656e-02, -4.3213e-02, -8.3160e-04,
             -4.5395e-04, -8.0078e-02, -5.1117e-04,  1.7676e-01, -2.3682e-02, -1.3281e-01, -2.0142e-02,  3.9291e-04,
              8.6594e-04,  4.3297e-04, -1.5320e-02,  1.4221e-02,  1.4496e-03,  7.5195e-02,  4.7266e-01, -9.7656e-03,
             -1.4038e-02,  3.3379e-04, -1.0498e-01, -6.2866e-03,  3.1891e-03,  1.9932e-04,  2.8320e-02, -4.3213e-02,
              2.7466e-02, -2.8076e-02, -1.4496e-03, -1.2988e-01, -5.1260e-05,  2.6758e-01, -1.1133e-01, -9.8267e-03,
              1.2793e-01, -9.4727e-02,  4.0039e-02, -8.2520e-02,  4.6143e-02,  1.5869e-02,  6.9275e-03, -3.3691e-02,
              7.7344e-01,  3.9648e-01,  1.2817e-03, -1.1368e-03,  1.4400e-04,  1.1978e-03,  2.7734e-01,  8.7402e-02,
              1.1673e-03, -5.3467e-02,  2.0020e-01,  5.5078e-01,  6.9275e-03,  7.3242e-02,  1.9287e-02, -2.9053e-02,
              1.6235e-02, -6.5613e-04,  5.2246e-02,  4.1008e-04,  2.1362e-02,  3.7231e-03, -5.0354e-04, -8.1177e-03,
              1.1169e-02, -1.0791e-01,  1.5974e-05, -1.8750e-01,  1.6327e-03, -9.7656e-03, -9.4238e-02, -3.2715e-02,
              1.2268e-02,  7.4463e-03, -3.9101e-04,  6.3965e-02, -3.9978e-03, -7.0801e-02, -9.5215e-02, -4.9316e-02,
              4.1748e-02, -3.9673e-04, -1.3794e-02,  7.2266e-02,  7.7148e-02,  2.7148e-01, -4.0817e-04,  1.6113e-02,
             -6.0059e-02,  2.7710e-02,  9.2773e-03, -9.0332e-02,  7.0190e-04, -3.7766e-04, -1.5234e-01, -2.3047e-01,
              1.5080e-05,  4.9414e-01, -5.0293e-02,  2.4536e-02,  3.3379e-04, -1.1133e-01,  2.7832e-02, -6.1768e-02,
              3.1853e-04,  1.8215e-04,  1.9169e-04,  3.3379e-04,  3.6812e-04, -5.3406e-04, -4.1016e-02, -1.9531e-02,
             -3.3760e-04, -1.1279e-01,  5.6458e-03, -2.2736e-03,  4.8447e-04,  3.9339e-05, -7.6660e-02,  3.2959e-02,
             -9.6680e-02,  1.9336e-01,  2.6001e-02, -4.8340e-02,  6.5918e-02,  6.1523e-02, -4.8584e-02, -1.7700e-02,
             -1.9897e-02,  7.3624e-04,  4.4922e-02,  4.1748e-02, -1.1780e-02,  1.5335e-03, -1.8555e-02,  1.4941e-01,
              5.0354e-04, -3.2471e-02, -2.2736e-03, -2.1935e-04,  2.7618e-03,  2.7344e-01,  1.7578e-02,  1.3867e-01,
             -4.0054e-04, -9.0332e-02, -2.3145e-01,  1.3924e-04, -4.2152e-04, -2.0752e-03, -2.6733e-02,  6.9809e-04,
             -4.8828e-02, -7.6172e-02, -5.5664e-02,  2.4609e-01,  7.6172e-02,  3.3855e-05,  1.5442e-02, -7.9346e-03,
             -2.4414e-02, -2.6367e-01,  1.1182e-01, -5.3644e-05, -1.0254e-01, -3.3140e-05, -1.1475e-02, -3.3447e-02,
             -3.9551e-02,  5.7861e-02, -4.6387e-02, -6.4087e-03,  2.0996e-02,  5.3467e-02,  3.6621e-04, -4.1406e-01,
             -6.8359e-02,  1.3184e-02, -2.2095e-02, -1.3000e-02, -3.3984e-01,  1.6975e-04,  1.7357e-04, -6.7139e-04,
              1.5503e-02,  9.1309e-02,  3.6621e-04, -2.0630e-02,  4.8065e-04,  1.2695e-01, -1.5430e-01,  1.8188e-02,
             -1.7166e-04, -9.4175e-06,  2.3651e-03,  1.0157e-04, -8.3447e-06, -1.0791e-01,  1.6098e-03,  1.0693e-01,
             -2.9785e-02,  7.7148e-02,  1.8768e-03, -1.1719e-01,  1.9409e-02,  5.0781e-02,  6.2988e-02,  2.6398e-03,
             -1.3245e-02, -1.7853e-03, -8.4473e-02, -3.9864e-04,  1.1230e-01, -3.5645e-02, -2.6398e-03,  6.1768e-02,
              1.3428e-02, -8.0078e-02, -1.1865e-01,  6.3965e-02, -1.0889e-01, -6.2988e-02,  9.9121e-02,  8.4229e-03,
              1.6211e-01, -5.8838e-02,  3.5156e-01, -7.1777e-02, -7.1875e-01, -5.1117e-04,  2.8992e-03,  2.5513e-02,
             -1.8978e-04, -3.3203e-01,  3.9673e-03, -9.5825e-03,  6.6528e-03,  5.7983e-04,  3.0884e-02, -8.5449e-02,
              7.2266e-02, -4.8523e-03,  5.1025e-02,  2.8320e-01,  6.9336e-02, -1.9684e-03, -2.8711e-01,  1.0681e-02,
              2.9907e-03, -2.5177e-03, -1.0889e-01, -1.2934e-05, -2.1076e-04,  1.2512e-03,  2.5269e-02, -6.4941e-02,
              1.9897e-02, -3.7842e-02, -1.8082e-03, -4.1016e-02,  5.5420e-02, -1.9043e-01, -8.0078e-02,  1.7471e-03,
             -1.3367e-02,  5.3223e-02,  3.8867e-01, -1.7188e-01,  5.0049e-02,  3.2959e-02,  1.3062e-02,  3.2959e-02,
              6.5994e-04,  2.3730e-01, -6.0303e-02,  4.7607e-02, -1.5137e-01, -7.4707e-02, -6.4453e-02, -2.7657e-04,
              1.9165e-02, -1.3885e-03,  3.0029e-02,  5.5908e-02, -4.8828e-04, -1.6504e-01, -3.1982e-02,  6.8188e-05,
              1.4551e-01,  3.5400e-02,  1.2891e-01,  7.2479e-04,  1.3379e-01,  3.4668e-02,  3.4668e-02,  2.3804e-02,
              8.9169e-05, -8.0872e-04,  1.0300e-03, -1.6211e-01, -2.7466e-02, -4.9805e-02,  2.5195e-01, -8.3496e-02,
              7.4158e-03, -2.0312e-01, -2.2754e-01, -3.1006e-02, -6.1523e-02,  1.5039e-01, -2.1973e-03, -2.0630e-02,
             -3.4714e-04,  2.7344e-02,  3.1982e-02,  3.1494e-02, -3.9062e-02, -2.2461e-02, -1.2988e-01,  9.9121e-02,
             -2.4048e-02, -7.3547e-03,  1.0791e-01, -3.6377e-02, -6.3477e-03, -1.4877e-03,  5.5847e-03,  5.5695e-04,
              2.0508e-02, -8.6670e-03, -2.1839e-04, -2.6123e-02,  1.0376e-02, -1.5926e-04, -1.1426e-01, -1.1444e-03,
             -2.2339e-02, -1.3672e-01, -6.9336e-02,  1.4160e-01,  4.2236e-02, -1.9531e-03, -1.8539e-03, -2.5879e-02,
             -3.5400e-02, -5.3406e-03,  1.5945e-03,  1.7738e-04, -7.9870e-06,  3.5858e-03,  1.3649e-05,  4.4107e-06,
              3.9101e-04,  1.2598e-01, -1.4258e-01, -1.4282e-02, -4.1016e-02,  2.1458e-06,  4.8828e-02,  3.7575e-04,
              5.8105e-02,  5.9509e-04,  2.1240e-02,  5.5313e-04, -1.7773e-01,  4.4678e-02, -3.7598e-02,  5.4550e-04,
              3.1738e-03, -6.2500e-02,  2.9663e-02,  2.2125e-03,  2.2221e-04, -4.3213e-02,  2.3438e-01,  9.2163e-03,
              2.2461e-01,  5.3406e-03,  4.3030e-03,  2.0508e-02, -1.9836e-03,  1.7166e-04,  7.4387e-04, -1.2512e-02,
             -5.7678e-03, -1.5869e-02,  1.7188e-01,  1.6357e-02, -3.9978e-03, -1.0205e-01, -4.0039e-01, -6.2256e-02,
             -1.2598e-01,  3.4668e-02,  7.5378e-03,  1.9836e-03,  2.7061e-05, -6.3896e-05,  1.3351e-03,  4.1992e-02,
             -6.5918e-02,  1.1215e-03, -1.1035e-01,  2.4128e-04,  3.3008e-01,  4.3213e-02, -7.8125e-02,  1.7480e-01,
              1.6785e-04, -3.8281e-01, -8.4229e-03,  6.8665e-04, -1.8921e-02, -6.5430e-02,  1.7871e-01,  1.3477e-01,
             -9.4727e-02, -1.9043e-01, -2.8809e-02, -1.1587e-04,  1.2756e-02, -1.4343e-03,  1.9043e-02, -5.7617e-02,
             -9.9121e-02,  6.6895e-02,  7.4219e-02, -2.7832e-02, -1.7357e-04, -5.3223e-02,  5.7617e-02,  3.3379e-04,
             -3.7537e-03,  1.2512e-03, -4.8447e-04,  2.7710e-02, -1.3770e-01,  3.6621e-02, -1.1035e-01,  1.7452e-04,
             -1.3962e-03, -8.1055e-02,  1.2207e-02, -3.0518e-03,  1.0681e-02,  8.8215e-05,  1.0303e-01, -5.6396e-02,
              2.2095e-02, -1.6797e-01, -1.3161e-04, -3.1128e-03, -4.1504e-02, -1.2988e-01,  1.8158e-03, -2.5940e-03,
              3.4668e-02, -1.7456e-02,  3.4637e-03, -9.4604e-03,  1.0645e-01, -7.1777e-02, -8.0859e-01,  2.9373e-04,
              2.0605e-01, -3.1128e-02,  3.7231e-03,  1.7929e-03, -2.2125e-03,  4.1992e-02,  1.0693e-01, -8.0078e-02,
              5.3467e-02, -7.2479e-05, -3.2227e-02, -4.1016e-02,  5.5908e-02,  2.8198e-02, -9.7656e-02,  7.0496e-03,
              2.6978e-02,  7.3242e-04, -9.1309e-02, -6.3477e-02,  1.5039e-01, -9.9121e-02, -1.9455e-04,  9.9182e-04,
             -2.8076e-03,  2.5195e-01, -2.4170e-02,  2.0703e-01, -5.3787e-04, -1.3574e-01, -8.6426e-02, -6.7871e-02,
              7.3242e-02,  3.5706e-03,  6.9885e-03,  3.4637e-03, -1.0938e-01,  8.6060e-03, -3.9062e-02,  5.1758e-02,
              4.5166e-02, -3.5547e-01,  1.9836e-03, -9.0820e-02, -8.0078e-02, -2.3346e-03, -7.0572e-04,  2.8534e-03,
             -2.0862e-06, -4.3701e-02, -3.8300e-03, -2.0142e-02, -1.1133e-01, -9.7656e-03,  5.0537e-02,  5.7373e-02,
              5.2490e-02,  2.7832e-02,  9.8145e-02, -2.2705e-02,  1.2875e-04, -9.0820e-02,  4.9114e-05, -9.2773e-03,
             -1.6895e-01,  6.5231e-04, -3.6865e-02, -1.0010e-01, -8.8379e-02, -4.1260e-02,  9.9121e-02,  7.7057e-04,
             -7.9956e-03, -1.6309e-01, -4.3213e-02, -8.8867e-02,  1.2256e-01,  6.7871e-02,  4.0527e-02, -8.6426e-02,
             -2.0752e-03, -1.7676e-01, -3.6865e-02, -2.8610e-04,  3.5889e-02, -6.4087e-04, -4.2725e-03,  1.4355e-01,
              5.5313e-04, -1.6327e-03, -3.8818e-02,  3.1853e-04, -3.1055e-01,  1.5234e-01, -7.7209e-03,  3.4094e-05,
             -1.3489e-02, -3.3691e-02,  6.4697e-03, -2.7954e-02, -8.7891e-03,  5.7068e-03, -1.6098e-03,  1.0498e-02,
              1.7090e-01,  4.6997e-03, -1.2793e-01, -3.3569e-04,  4.7922e-05,  3.0708e-04, -2.3651e-03, -1.8433e-02,
              8.0566e-03, -1.7456e-02,  3.0518e-03, -2.5635e-02,  2.9102e-01, -1.7578e-01, -1.2756e-02, -8.0566e-03,
              1.8234e-03,  8.5938e-02,  9.2316e-04,  1.1719e-02,  1.5497e-05, -6.7749e-03, -3.5477e-04, -6.6223e-03,
              2.7588e-02,  1.0864e-02,  1.5163e-04, -1.6992e-01,  4.4556e-03,  2.6321e-04, -2.2278e-03,  1.2665e-03,
              2.6367e-02,  1.1292e-03, -4.9744e-03, -1.4746e-01,  1.3809e-03, -2.2266e-01,  3.6621e-02, -5.2002e-02,
             -5.9509e-04, -3.1738e-02,  1.8555e-02,  1.4771e-02, -3.9291e-04, -4.2236e-02, -4.6387e-03, -1.2436e-03,
             -9.9487e-03,  1.8359e-01, -1.2695e-01, -1.8215e-04, -6.4453e-02,  1.4400e-04, -8.3496e-02, -3.8574e-02,
             -8.5938e-02, -4.4189e-02,  2.2217e-02, -1.2402e-01,  2.1458e-04, -7.2021e-03,  8.9722e-03, -2.2949e-02,
              3.1494e-02, -3.1445e-01,  1.1139e-03,  1.9989e-03,  7.0190e-04, -2.2266e-01, -2.3956e-03,  5.5176e-02,
              4.5410e-02, -6.2500e-01,  4.6349e-04, -5.5695e-04,  1.1169e-02,  2.8534e-03,  1.1670e-01,  1.0864e-02,
             -1.5015e-02, -1.7212e-02,  1.5717e-03, -2.1515e-03,  1.2329e-02, -2.8992e-03,  2.4805e-01, -1.6992e-01,
              1.0925e-02, -3.9551e-02, -9.9609e-02,  8.5938e-02,  5.3955e-02, -1.3275e-03,  2.5940e-04, -3.4714e-04,
              2.4109e-03, -4.0771e-02, -2.2583e-02, -5.9082e-02,  2.0996e-02, -5.9128e-04,  1.4551e-01,  5.7983e-03,
              2.5195e-01, -3.2997e-04,  2.1515e-03, -5.7602e-04,  5.0293e-02,  1.1035e-01, -1.4221e-02, -4.1504e-02,
              1.0223e-03, -1.8597e-05,  1.8066e-01, -1.4893e-02, -5.9509e-04, -1.6846e-02, -1.0452e-03, -2.7100e-02,
             -1.9434e-01,  9.4604e-03,  5.6885e-02,  5.1575e-03,  3.0060e-03,  1.2875e-04,  7.1049e-05, -2.3193e-02,
             -4.1809e-03, -1.0452e-03,  2.0264e-02,  6.1523e-02, -2.7275e-04,  4.4556e-03,  6.6895e-02,  8.4473e-02,
             -7.6675e-04,  1.2756e-02, -4.7607e-03,  1.0498e-02, -1.7871e-01, -1.4648e-03,  5.2734e-02,  7.1526e-05,
              5.2490e-02, -6.7871e-02,  5.2734e-02,  1.5137e-01,  1.0498e-01, -4.0039e-01, -1.8677e-02, -8.0872e-04,
              8.3008e-03, -1.4221e-02, -1.5137e-01,  9.1934e-04,  9.4604e-03,  8.0872e-04, -2.6321e-04,  2.5195e-01,
              1.9550e-04,  3.8338e-04,  9.3842e-04,  5.4199e-02, -1.1963e-02,  1.5259e-02, -1.7773e-01,  8.0566e-03,
             -1.0300e-04, -4.7119e-02,  2.3828e-01, -9.5703e-02, -4.4556e-03,  2.8076e-03, -2.9785e-02,  2.6367e-02,
              2.8038e-04,  3.9795e-02,  1.2894e-03,  1.1635e-04, -8.7891e-02,  1.1368e-03, -5.8365e-04,  5.0537e-02,
              8.4473e-02,  2.3315e-02, -1.1444e-03, -8.7402e-02, -1.4465e-02, -1.0303e-01, -2.0215e-01,  5.2246e-02,
             -2.3651e-04,  8.6308e-05,  1.3046e-03, -2.8809e-02, -1.9264e-04, -2.3193e-03, -4.9072e-02, -1.3504e-03,
             -2.2125e-03,  1.1063e-03,  1.2988e-01, -7.7209e-03, -1.1719e-01, -5.6839e-04, -3.7003e-04, -1.2634e-02,
              1.7334e-02, -7.6660e-02,  8.2779e-04, -3.8086e-02, -4.9805e-02,  1.4453e-01,  1.4355e-01, -4.6692e-03,
             -8.6914e-02,  3.7354e-02,  2.4796e-04, -3.8528e-04,  1.6689e-04,  4.0527e-02,  1.8799e-02, -8.5449e-02,
              4.8828e-02, -6.8665e-04,  2.6001e-02,  1.4725e-03,  8.5449e-03, -5.9082e-02,  1.7383e-01,  6.1951e-03,
             -3.2227e-02, -9.2773e-02, -2.3315e-02, -2.5024e-03, -2.7222e-02, -5.1514e-02,  2.5024e-02,  2.3315e-02,
             -2.4033e-04,  3.5645e-02, -7.3242e-02,  7.8678e-06,  2.6367e-02,  2.5977e-01,  1.1084e-01, -8.2779e-04,
              8.3160e-04, -1.6327e-03,  9.2163e-03, -8.9645e-04, -7.9102e-02,  1.1353e-02,  6.3965e-02, -4.4434e-02,
             -1.1623e-06,  9.8267e-03,  2.5940e-03, -3.6377e-02, -4.4189e-02,  8.4961e-02, -9.9609e-02, -9.3262e-02,
             -1.1902e-02,  5.3711e-02, -3.8330e-02,  2.5586e-01,  2.3438e-01, -8.2520e-02,  4.5166e-03,  4.6997e-03,
              2.0294e-03, -4.5471e-03, -4.1580e-04,  2.5391e-02, -1.6113e-02, -9.2773e-03,  4.1962e-05, -1.1963e-01,
              1.8692e-03,  1.2131e-03,  5.3223e-02,  4.3640e-03, -1.5039e-01,  1.7700e-02, -8.8379e-02, -1.0107e-01,
             -8.0078e-02,  1.0586e-04,  3.8719e-04, -1.8457e-01,  6.0791e-02,  8.1177e-03, -5.0293e-02,  6.5994e-04,
             -3.5156e-02, -7.4768e-04, -2.5269e-02,  7.6675e-04,  1.2512e-02, -1.3367e-02, -7.7209e-03, -7.1411e-03,
              9.5215e-02, -5.9843e-05,  5.8594e-02,  5.2734e-02,  5.4550e-04,  1.7262e-04,  3.3760e-04, -7.4463e-03,
              5.2246e-02,  4.6997e-03, -1.8677e-02,  1.8311e-03, -8.8501e-04,  3.8086e-02, -7.9956e-03, -1.5320e-02,
              4.3945e-02, -2.1057e-03, -2.9663e-02, -5.4688e-02, -1.8768e-03, -7.1289e-02, -1.5259e-03, -4.2725e-04,
             -1.3256e-04, -4.4678e-02,  9.5215e-02, -4.7363e-02,  2.9755e-04,  7.2937e-03,  2.8229e-04,  2.1729e-02,
             -2.4414e-04,  5.5664e-02, -4.6875e-02,  2.9419e-02,  1.2109e-01, -1.2988e-01,  6.6406e-02,  1.5182e-03,
              1.2817e-02,  5.2490e-03, -1.2146e-02, -8.0566e-03,  2.7954e-02, -3.3569e-03,  1.1215e-03, -9.6191e-02,
              1.4355e-01,  1.0605e-03, -1.2793e-01, -2.9755e-03,  6.5918e-02,  1.1133e-01,  8.3008e-03, -1.1169e-02,
             -3.9307e-02, -3.0327e-04,  1.1084e-01,  8.2397e-03,  2.1606e-02,  5.1025e-02,  1.1841e-02, -4.4823e-05,
             -3.4424e-02, -2.2949e-02, -1.7853e-03,  3.4180e-01, -5.1117e-04, -3.6377e-02, -2.1935e-04,  1.0395e-04,
              5.0537e-02, -1.6016e-01, -6.2561e-04,  1.1963e-01, -1.3000e-02,  9.6680e-02, -4.8584e-02, -2.0142e-02,
             -1.3184e-02,  1.6403e-04, -2.9907e-02,  5.8350e-02,  8.1543e-02, -8.1062e-05, -5.1514e-02,  2.8320e-01,
              5.6458e-03,  1.0315e-02,  1.8750e-01,  4.2439e-05,  1.2512e-03,  1.0300e-03, -2.0508e-02,  1.6022e-03,
             -6.1035e-02,  1.2634e-02, -8.0872e-04, -1.7456e-02,  1.4453e-01,  2.8076e-03,  1.3580e-03, -4.0054e-04,
              2.8198e-02, -5.6252e-07, -9.2773e-02, -6.1417e-04, -5.3223e-02, -3.2043e-04,  5.1025e-02,  1.0156e+00,
             -3.9673e-03,  2.3926e-02,  6.7520e-04, -3.6377e-02, -5.7983e-03,  2.6758e-01, -3.1982e-02,  5.0049e-03,
             -7.7057e-04, -1.7471e-03,  3.6049e-04,  1.0449e-01, -9.7046e-03, -3.9673e-03,  2.4902e-02,  2.1973e-03,
              2.0409e-04, -6.5308e-03, -1.7090e-01, -4.7656e-01, -3.1445e-01,  6.7383e-02,  1.9684e-03, -1.1523e-01,
              9.9487e-03,  6.5918e-02,  1.3428e-02,  1.5991e-02,  4.3945e-02,  3.6914e-01, -6.5430e-02,  2.4796e-04,
             -3.1836e-01,  2.2430e-03,  3.5477e-04, -7.4768e-04,  5.7373e-03, -2.0218e-04, -4.9072e-02, -8.7280e-03,
             -1.8433e-02, -2.9883e-01, -2.6562e-01, -1.2684e-04,  2.2070e-01, -3.0518e-02,  6.5430e-02,  1.2064e-04,
             -1.9150e-03, -1.6602e-01, -1.5381e-02, -1.0910e-03, -1.8457e-01,  1.8239e-05, -1.0193e-02,  7.3730e-02,
              3.3936e-02, -1.1035e-01,  1.7480e-01,  8.6914e-02,  6.3705e-04, -6.2988e-02, -1.3672e-01, -2.0266e-05,
             -2.9663e-02,  1.2085e-02, -2.2095e-02, -2.5749e-04, -1.2012e-01,  9.3505e-07,  2.0117e-01,  4.9072e-02,
              1.0633e-04, -1.9550e-04, -1.4844e-01,  4.5166e-02, -8.7891e-02, -4.5586e-04, -1.7383e-01,  2.1484e-02,
              3.6774e-03,  4.3945e-02, -2.2339e-02, -1.9360e-04,  4.4922e-02,  3.2425e-04, -1.7242e-03, -3.5352e-01,
              7.8678e-05, -7.6172e-02, -8.2031e-02,  4.4434e-02, -2.6245e-03,  4.5166e-02,  6.4453e-02,  1.1670e-01,
              4.2114e-03,  2.4414e-02, -3.1494e-02, -5.7129e-02,  1.7578e-02, -4.1504e-02, -6.9580e-03,  5.6076e-04,
              1.1597e-02, -2.9449e-03,  4.8218e-03, -5.1562e-01, -2.0996e-01, -7.5684e-02,  4.6997e-03,  3.5400e-02,
             -2.2583e-02,  6.1768e-02, -6.2988e-02, -9.0790e-04, -4.3701e-02, -1.2024e-02, -8.3984e-02, -3.3594e-01,
              1.9169e-04, -2.8687e-02,  2.0703e-01, -9.4223e-04,  2.0386e-02, -3.1128e-02, -1.0889e-01,  4.0588e-03,
             -8.5831e-04, -7.6172e-02, -7.9346e-03,  3.4424e-02,  8.6060e-03,  4.5410e-02, -5.7861e-02,  4.7607e-02,
              2.9785e-02,  3.0396e-02,  1.3489e-02, -1.6235e-02,  7.7637e-02, -1.5488e-03, -1.2512e-03,  2.1648e-04,
              9.4604e-03, -4.7302e-03, -8.4686e-04, -2.0898e-01, -2.0312e-01, -7.1289e-02, -7.5684e-02, -1.8433e-02,
             -3.2654e-03, -3.5286e-04, -6.2988e-02, -1.5335e-03,  2.1172e-04,  7.6660e-02, -9.4238e-02,  6.7444e-03,
              1.1621e-01, -6.9046e-04,  1.6724e-02,  6.4373e-05, -1.2159e-04,  3.2997e-04, -1.0132e-02, -6.0425e-03,
              8.8882e-04,  1.7456e-02,  4.1809e-03, -8.5938e-02, -1.0498e-02, -1.2207e-03,  3.2806e-03,  8.0109e-05,
              2.3438e-02,  3.4668e-02, -2.9449e-03,  8.5449e-02, -5.5664e-02,  2.4902e-01,  2.1851e-02,  9.0332e-02,
              9.3079e-04, -1.9302e-03,  2.2827e-02,  4.4678e-02,  1.3574e-01, -8.6670e-03, -1.3281e-01, -1.5259e-03,
             -4.1504e-02,  1.3855e-02, -1.2402e-01, -1.5625e-01, -9.7656e-02,  1.0449e-01,  1.6992e-01, -4.1809e-03,
             -1.0347e-04,  3.4180e-02,  1.4038e-03, -6.5308e-03, -2.8320e-02, -1.1914e-01,  1.0864e-02,  2.6107e-05,
              1.0010e-01, -2.5177e-03, -1.6406e-01, -3.8757e-03, -6.3477e-02, -7.7820e-03, -4.0531e-05, -5.1880e-04,
              1.2939e-02,  1.0205e-01, -9.5825e-03,  3.2715e-02,  6.1523e-02,  1.2894e-03, -2.9419e-02, -2.1240e-02,
              1.2634e-02, -3.4668e-02, -4.6387e-02, -3.2471e-02,  9.7275e-04,  5.4550e-04,  1.4099e-02, -6.1989e-05,
              2.0264e-02,  1.0559e-02, -4.4922e-02,  6.9824e-02, -7.2861e-04,  2.1973e-02,  1.6602e-01, -1.9897e-02,
              1.2891e-01,  2.0630e-02,  1.6937e-03,  4.6875e-02,  3.3826e-06,  3.7384e-04, -6.9141e-05, -4.9591e-04,
              7.2754e-02, -8.9645e-05,  5.4932e-04, -9.8267e-03,  9.3262e-02,  4.2725e-02,  1.7262e-04, -7.9102e-02,
             -2.3926e-02, -9.4238e-02, -4.5703e-01, -6.0120e-03, -4.3701e-02,  1.4648e-01,  9.5703e-02,  1.5869e-02,
             -1.5747e-02,  5.2185e-03,  3.1006e-02,  4.7302e-03, -4.0770e-05, -7.9102e-02, -3.1982e-02, -1.7456e-02,
             -2.5513e-02, -9.1553e-03,  9.5215e-02, -3.6865e-02,  1.6975e-04,  5.8838e-02, -6.5231e-04, -6.4453e-02,
              4.3164e-01, -4.4441e-04, -3.8300e-03, -2.8198e-02, -2.4796e-04, -3.0060e-03, -8.4229e-03,  2.3071e-02,
             -5.9509e-03, -3.4912e-02,  7.7820e-04, -1.6861e-03, -3.6621e-02, -1.1963e-01, -2.4414e-02, -1.4453e-01,
              6.6406e-02, -3.8385e-05,  1.0010e-01, -8.0566e-03, -4.4632e-04,  7.6660e-02, -7.4463e-03, -8.2520e-02,
             -2.5988e-05, -8.8379e-02,  1.1768e-01, -7.2266e-02,  1.0156e-01, -9.3262e-02, -5.2002e-02, -6.2500e-02,
              4.6143e-02, -1.8555e-01, -1.4709e-02,  1.5640e-03, -2.9541e-02, -1.3351e-04,  2.3438e-01, -6.6757e-05,
              6.5613e-03, -3.7193e-04, -9.1309e-02,  2.5330e-03, -1.7395e-03,  2.4707e-01, -1.0986e-02, -1.4404e-02,
             -2.9663e-02,  5.2261e-04,  8.6060e-03,  2.8809e-02, -3.0060e-03,  7.4707e-02,  3.4180e-03, -1.8616e-03,
             -9.8877e-03,  7.2861e-04,  2.9449e-03, -1.0254e-02, -7.1716e-04,  5.5542e-03, -3.7384e-04, -5.8365e-04,
              4.0894e-03, -7.2861e-04,  1.2158e-01, -3.6621e-02,  7.7637e-02,  9.3750e-02, -6.7520e-04, -7.9956e-03,
              3.6133e-01, -3.9368e-03,  5.9814e-02,  3.2471e-02,  1.3184e-01,  7.5684e-02,  2.0264e-02, -1.3123e-02,
              2.0599e-03,  1.1683e-04, -1.4877e-04, -9.7656e-03,  1.1444e-03, -2.0294e-03,  7.3730e-02,  3.1586e-03,
             -2.4536e-02,  3.6430e-04,  1.6895e-01,  1.8433e-02, -4.2969e-02,  4.1748e-02, -5.9570e-02, -8.0566e-03,
             -4.0283e-02,  4.3945e-02, -1.3885e-03,  1.0186e-10,  4.5410e-02, -8.7023e-06,  2.5391e-02, -3.9307e-02,
              1.5076e-02,  2.6703e-04, -3.3951e-04, -2.4023e-01, -1.5869e-02, -5.3711e-02,  8.7891e-03,  1.5430e-01,
              5.0781e-02,  7.2266e-02, -9.7656e-03,  7.8735e-03,  4.3488e-04, -2.7588e-02, -1.3184e-01,  5.8838e-02,
              4.2725e-03, -3.6926e-03,  2.9602e-03, -2.0117e-01, -9.4604e-03,  6.2180e-04,  8.7357e-04,  1.2634e-02,
              2.4707e-01, -1.5259e-03,  1.5820e-01, -2.4658e-02,  3.8574e-02,  2.6123e-02, -1.0910e-03,  1.7776e-03,
              1.5137e-01,  2.5368e-04,  2.1362e-02, -5.9204e-03,  3.2806e-04, -1.4160e-02, -2.9755e-04,  7.2754e-02,
              7.2937e-03, -3.4180e-02,  1.1902e-02,  1.5820e-01,  3.7695e-01,  3.4180e-01,  5.1270e-02,  1.1673e-03,
              9.2285e-02, -5.7373e-03, -1.7471e-03, -3.2227e-02,  6.2256e-03,  2.1387e-01, -6.3705e-04, -1.3351e-05,
              7.7515e-03, -1.1230e-02,  2.2070e-01,  2.6172e-01, -1.4258e-01,  1.6880e-04, -4.6387e-03, -6.1279e-02,
              5.4443e-02,  5.8289e-03,  6.0730e-03, -3.0884e-02,  9.1797e-02, -1.2305e-01,  1.6785e-03, -2.3346e-03,
             -1.5442e-02,  1.9727e-01,  9.8633e-02, -4.5898e-02, -6.2988e-02, -7.1777e-02,  1.7212e-02,  6.6895e-02,
              6.4850e-04,  1.1963e-01, -5.8746e-04, -3.5553e-03, -3.3203e-02,  4.5166e-03, -1.4453e-01, -2.0447e-03,
              9.8419e-04, -1.3184e-02, -3.1738e-02,  9.9182e-04,  4.6875e-01, -7.6599e-03,  9.2983e-06,  1.4355e-01,
             -2.1851e-02,  5.1562e-01,  3.3203e-02,  5.7373e-02,  1.6406e-01,  1.7456e-02, -1.3065e-04,  3.3447e-02,
             -4.8584e-02,  6.3477e-02,  2.4796e-04, -1.1253e-04, -1.4221e-02, -1.0071e-03,  8.3618e-03,  8.4961e-02,
             -1.0840e-01,  8.6060e-03,  1.0742e-01, -5.7983e-04, -3.1128e-02, -9.2285e-02,  5.7861e-02,  2.1648e-04,
             -1.2598e-01, -5.2734e-02,  1.5625e-02,  4.2725e-02,  2.5749e-04,  4.1199e-03, -8.6212e-04, -1.3379e-01,
              1.9379e-03, -4.2969e-01,  1.0376e-02, -1.2360e-03,  2.2705e-02,  9.4727e-02,  1.8750e-01,  1.7456e-02,
              2.0215e-01, -9.7656e-03,  3.1586e-03, -1.2891e-01,  4.9072e-02, -3.2043e-04,  2.4048e-02,  2.3499e-03,
              1.9531e-02, -1.6479e-02,  1.4420e-03, -4.2969e-02,  8.7280e-03, -2.2095e-02,  4.1504e-02,  3.0640e-02,
             -8.1543e-02,  4.8584e-02, -6.9824e-02,  3.7909e-05,  1.5381e-02,  3.3417e-03, -3.7598e-02, -6.0791e-02,
              1.0757e-03, -5.4297e-01, -8.3496e-02, -4.9744e-03,  5.4443e-02, -6.6895e-02,  5.2734e-02, -6.3705e-04,
              2.2984e-04, -5.7373e-02, -8.0078e-02, -1.4343e-03,  6.2943e-05, -2.5757e-02,  7.5989e-03, -7.6294e-03,
              1.0312e+00, -3.5858e-04,  3.0365e-03,  6.7871e-02, -1.7578e-01,  3.4180e-02, -8.9844e-02,  2.5757e-02,
              9.2163e-03, -3.2959e-03, -7.0953e-04,  1.3379e-01,  1.4282e-02,  7.6675e-04,  2.0599e-04, -5.5432e-06,
             -4.2725e-03,  4.0527e-02, -9.2773e-02, -8.7280e-03,  1.1063e-03, -1.2695e-02,  3.3447e-02,  1.6211e-01,
             -3.8818e-02, -1.5503e-02, -3.8330e-02,  1.3199e-03,  2.0264e-02,  3.0029e-02, -3.9307e-02,  3.0899e-04,
             -1.2451e-02, -9.7752e-06,  2.0905e-03, -2.5391e-02, -1.2878e-02,  2.1387e-01,  1.0864e-02,  3.3984e-01,
              7.8613e-02, -2.0020e-01, -2.0142e-03,  1.5793e-03,  5.0537e-02, -3.0664e-01,  2.3682e-02, -3.5553e-03,
              1.2695e-01, -5.0537e-02,  2.2278e-03,  6.2500e-02, -1.1108e-02, -4.0771e-02, -5.8174e-05, -1.3916e-02,
              3.6133e-02, -2.4658e-02, -4.6349e-04, -3.3722e-03, -9.6130e-04,  7.6172e-02,  1.4453e-01,  7.9346e-03,
              1.2305e-01, -6.8848e-02,  2.4536e-02, -2.9602e-03, -3.6240e-04,  7.4463e-03,  6.9275e-03,  1.1084e-01,
              4.3213e-02, -7.3730e-02, -4.3640e-03,  2.7618e-03,  1.3123e-02, -2.0996e-01, -1.7090e-02, -1.1587e-04,
             -4.1992e-02,  2.8229e-04,  8.8501e-03,  5.7373e-02,  1.0791e-01,  9.8633e-02, -9.6680e-02,  2.0264e-02,
              1.3477e-01, -3.9551e-02, -8.5449e-02, -2.5940e-04, -2.2095e-02, -2.2095e-02,  9.1797e-02,  1.3489e-02,
             -4.0527e-02,  3.2471e-02,  2.4536e-02,  0.0000e+00, -6.5918e-02,  2.6611e-02, -4.4434e-02,  1.0925e-02,
              1.0376e-03,  4.8523e-03,  7.2002e-05, -1.4355e-01, -5.6641e-02,  2.7771e-03, -1.9629e-01,  1.7319e-03,
              9.8267e-03,  5.9082e-02,  4.9744e-03,  5.4199e-02,  2.4438e-05, -1.7944e-02,  1.5430e-01, -3.6133e-02,
              9.4238e-02, -1.8850e-06, -2.1362e-02, -1.2878e-02, -2.7275e-04,  5.2929e-05, -8.8120e-04,  1.0205e-01,
             -4.4434e-02, -2.0215e-01, -3.7354e-02, -6.5613e-04, -1.7188e-01, -4.1580e-04, -3.0212e-03, -4.0771e-02,
              1.3123e-03, -2.0599e-03,  4.1809e-03,  1.5137e-01,  3.0823e-03,  2.1729e-02, -1.3086e-01,  1.7334e-02,
              1.2354e-01,  4.5166e-03, -3.6377e-02,  3.8330e-02, -1.6211e-01,  8.2970e-05,  1.3672e-01,  9.9609e-02,
             -2.6489e-02,  7.2266e-02, -7.5684e-02,  6.0320e-05, -1.7548e-04,  4.5166e-02, -8.3923e-04, -3.1891e-03,
             -3.6926e-03,  8.2031e-02,  9.8145e-02, -9.3384e-03, -6.8848e-02, -1.8848e-01, -2.4414e-01,  1.9141e-01,
              2.4796e-04, -2.1582e-01, -5.4688e-02,  2.7954e-02, -9.1076e-05, -1.1368e-03, -8.6212e-04, -2.3193e-03,
             -8.4961e-02, -3.7193e-04,  7.4387e-05,  9.9609e-02,  3.1433e-03,  7.1289e-02, -5.4932e-04,  7.2861e-04,
              2.4109e-03,  2.9541e-02,  1.2988e-01, -3.6240e-05,  1.0400e-01, -2.1057e-03, -6.8848e-02,  2.1606e-02,
              1.4258e-01,  7.6660e-02, -9.0942e-03,  3.2959e-03, -1.4771e-02, -3.3691e-02, -4.0588e-03,  8.4839e-03,
             -1.0376e-02,  3.8086e-02,  4.2439e-05,  5.4199e-02,  7.3853e-03,  3.0396e-02,  5.8746e-04,  8.5449e-04,
             -1.8677e-02,  1.4954e-02, -1.3306e-02, -8.5938e-02,  5.3955e-02, -5.9204e-03, -8.0078e-02,  3.1891e-03,
              2.3346e-03,  7.2266e-02,  1.7929e-04,  5.9605e-05,  1.8652e-01,  4.1504e-03,  6.6280e-05,  2.3560e-02,
             -3.4180e-02,  5.6076e-04,  1.7090e-03,  4.4434e-02,  1.0452e-03,  4.7852e-02,  1.7334e-02,  3.5645e-02,
              1.8164e-01,  5.3467e-02,  6.4468e-04, -4.1962e-04,  3.1662e-04, -1.6113e-02, -7.5989e-03,  1.3770e-01,
             -1.0147e-03,  6.3965e-02,  4.4250e-04, -4.9072e-02,  4.8637e-04,  8.8501e-03, -3.2959e-02,  7.4863e-05,
             -2.9419e-02, -7.7148e-02,  8.3618e-03,  5.3787e-04, -3.2471e-02, -1.9379e-03,  2.2461e-02, -5.3711e-02,
             -2.5513e-02,  1.2891e-01,  5.4121e-05,  1.3794e-02, -2.5779e-06, -3.2715e-02,  7.0190e-03, -5.9204e-03,
             -6.5918e-02,  2.6978e-02, -7.3242e-02, -2.3047e-01, -6.7871e-02,  4.9805e-02,  8.5938e-02,  4.2480e-02,
             -2.3365e-05,  6.2988e-02, -2.2430e-03, -2.1606e-02,  6.6406e-02,  4.1016e-02,  2.9175e-02,  1.1523e-01,
             -6.6406e-02, -2.9419e-02, -2.0508e-02, -1.3379e-01,  2.5368e-04,  1.3828e-04,  6.3896e-05, -2.4292e-02,
             -4.9072e-02,  6.8665e-03,  1.0156e-01,  4.3030e-03, -4.6997e-03,  1.9141e-01,  1.0559e-02, -1.1914e-01,
              4.4823e-04, -1.0920e-04, -6.4697e-03, -1.1841e-02, -3.5858e-03,  1.1047e-02,  3.8300e-03, -4.9973e-04,
              2.2125e-04, -1.0147e-03,  1.7242e-03, -6.8359e-02, -9.2773e-03,  9.0942e-03,  1.3306e-02, -7.9956e-03,
              1.2512e-03, -4.5204e-04, -1.7773e-01,  5.1514e-02,  1.2500e-01, -1.7071e-04,  3.7305e-01, -1.7090e-02,
              8.1177e-03,  1.7090e-01,  3.4912e-02,  1.3477e-01, -7.4387e-04, -1.2305e-01,  2.2168e-01, -5.3406e-04,
             -2.3041e-03,  3.4142e-04, -2.8125e-01,  4.8340e-02, -6.2561e-03,  3.3722e-03, -8.9844e-02,  3.5645e-02,
             -5.8838e-02,  7.8201e-05,  1.3855e-02, -1.0315e-02,  2.4536e-02,  7.6660e-02, -8.6426e-02, -5.8594e-02,
             -1.5076e-02,  9.4414e-05, -3.4904e-04, -3.9307e-02,  3.1738e-02,  8.1177e-03,  3.4766e-01, -3.1250e-02,
              2.8906e-01, -6.4850e-04, -2.4223e-04,  5.1270e-03,  1.0742e-01,  9.7168e-02, -8.8215e-05, -1.0254e-01,
             -4.5410e-02,  3.4943e-03,  6.5918e-02, -1.2684e-04,  9.3579e-06,  2.0996e-02, -4.1602e-01,  3.0664e-01,
              7.4506e-07, -2.0020e-02,  4.6349e-04, -3.2715e-02,  1.5503e-02, -4.1580e-04, -1.9897e-02,  4.0894e-03,
              7.4219e-02,  1.4648e-02, -2.8320e-02, -1.4062e-01, -8.7738e-05,  1.4282e-02,  2.9297e-02,  7.4219e-01,
              2.4414e-02, -3.3203e-02,  9.9659e-05,  1.5723e-01, -1.0681e-02,  4.3213e-02, -6.6406e-02, -2.2507e-04,
             -4.0771e-02,  1.0889e-01, -7.0095e-05, -5.7602e-04,  1.2305e-01,  3.4668e-02,  1.6406e-01,  1.0681e-03,
              1.3867e-01,  6.9427e-04, -1.3428e-02, -7.5684e-02, -1.1865e-01, -7.2479e-04, -1.1523e-01, -5.0049e-02,
              3.2715e-02, -3.9307e-02,  1.1182e-01,  2.1935e-05, -8.0490e-04, -2.8076e-02, -2.6398e-03,  1.4771e-02,
             -3.8528e-04,  4.7302e-03,  9.2316e-04, -4.9973e-04, -8.9355e-02, -4.9438e-03, -2.6733e-02,  7.4768e-03,
              1.1826e-03,  1.0596e-01, -4.4922e-02, -2.3842e-04, -2.0752e-03, -1.3965e-01,  9.0027e-04, -3.1853e-04,
             -7.0312e-02, -2.9175e-02, -1.7212e-02,  2.1606e-02,  1.9043e-01, -3.0640e-02, -1.1749e-03, -1.3306e-02,
             -4.9973e-04,  3.2501e-03, -6.5625e-01, -3.1494e-02,  7.5989e-03, -1.1841e-02, -5.9891e-04, -5.0537e-02,
              1.0693e-01,  2.9297e-02,  3.4424e-02, -4.1504e-03, -3.1006e-02, -1.2188e+00, -1.3574e-01,  2.3560e-02,
             -7.0801e-03,  6.7139e-03,  3.6328e-01, -3.8528e-04, -6.4941e-02,  3.4912e-02, -1.1670e-01, -7.5781e-01,
             -1.3794e-02, -5.8174e-05,  5.4297e-01,  3.3447e-02,  9.4223e-04, -8.2397e-03, -1.8066e-02,  2.0905e-03,
             -5.7861e-02,  1.3351e-03, -9.1797e-02, -7.1777e-02, -4.5654e-02,  1.5717e-03, -5.0354e-04,  2.6245e-03,
             -8.9355e-02, -4.8637e-05,  7.1335e-04, -1.2112e-04,  1.9550e-05, -8.0490e-04,  2.3270e-04,  1.2817e-02,
              1.9824e-01, -2.2583e-03, -4.5776e-03, -1.6504e-01,  7.4387e-05,  6.3965e-02,  1.5527e-01,  7.6660e-02,
             -9.1797e-02, -7.9956e-03,  3.8330e-02, -1.4572e-03, -4.0054e-04, -4.2969e-02,  5.3711e-02,  1.4551e-01,
             -1.1587e-04, -2.7100e-02, -1.3428e-02,  2.8076e-03,  2.5635e-02, -2.0504e-05,  5.0049e-02,  1.9922e-01,
             -2.4658e-02, -3.3569e-03, -5.8350e-02, -3.8281e-01, -2.6512e-04,  9.8145e-02, -9.2773e-02,  6.6895e-02,
             -2.1484e-01,  1.4343e-02,  7.0801e-03,  1.9226e-03, -4.9973e-04,  2.8931e-02,  4.6968e-05,  7.0312e-02,
              6.4850e-05, -5.7617e-02,  1.5918e-01, -1.0010e-02,  6.2180e-04,  1.0498e-01, -1.8677e-02,  1.1133e-01,
              6.4453e-02,  1.1139e-03,  1.9653e-02, -5.4321e-03,  1.4404e-02,  4.7119e-02,  1.0742e-02,  8.1787e-03,
             -6.9141e-05,  7.5195e-02,  6.7871e-02,  1.9836e-03,  1.0529e-03,  3.3760e-04,  6.0320e-05,  1.1230e-01,
             -5.2002e-02,  4.3030e-03, -5.6641e-02,  3.5156e-01,  8.2520e-02, -7.0801e-03, -2.5558e-04, -4.4556e-03,
             -1.4019e-04, -1.4160e-02, -2.5195e-01,  9.4238e-02, -7.5195e-02, -4.7266e-01, -4.0054e-05, -2.7618e-03,
             -3.9978e-03, -1.2268e-02, -5.5075e-05,  3.8330e-02,  5.8838e-02,  6.0120e-03,  4.6730e-04, -1.6235e-02,
              1.3580e-03, -4.3335e-03, -1.4648e-02, -2.3560e-02,  2.2095e-02, -2.3937e-04, -2.8320e-02,  6.0059e-02,
             -5.9814e-02,  3.7432e-05, -2.1387e-01, -5.9843e-05, -2.8076e-02, -2.1191e-01, -1.7383e-01, -2.2278e-03,
             -1.2268e-02,  2.4658e-02, -4.0283e-03, -3.2715e-02,  1.2329e-02, -3.3760e-04,  4.2725e-02, -2.0386e-02,
             -6.4392e-03,  3.8574e-02, -2.5879e-02,  3.8086e-02, -1.5503e-02, -1.8799e-02,  9.3750e-02,  1.5020e-05,
             -1.1084e-01,  1.1963e-02, -1.3672e-01,  2.1484e-01,  3.7354e-02, -9.5367e-04,  2.3193e-02,  4.7607e-03,
              5.2734e-02, -1.5354e-04,  9.0820e-02,  2.4780e-02,  2.7618e-03, -2.9541e-02,  9.4604e-03, -1.3770e-01,
             -9.8944e-06, -1.3199e-03, -2.5635e-02, -9.0332e-02, -4.2419e-03, -2.6855e-02, -4.3640e-03,  5.8838e-02,
              6.3965e-02,  2.3804e-03, -1.1444e-03,  4.3297e-04,  9.2773e-02, -2.4048e-02, -2.6733e-02, -9.8419e-04,
              4.1211e-01, -7.7148e-02,  6.6406e-02,  3.4912e-02,  3.6621e-02, -2.2095e-02, -8.2397e-04,  1.0252e-04,
             -4.7852e-02,  1.6880e-04, -1.2684e-04,  1.1215e-03, -4.2915e-06,  9.3750e-02,  1.7319e-03,  9.5703e-02,
              4.7119e-02, -3.0994e-06,  9.6130e-04,  9.0599e-05, -1.1873e-04, -1.9141e-01, -5.0293e-02,  1.5723e-01,
             -2.7161e-03, -2.1210e-03, -8.7357e-04, -2.9541e-02, -1.2390e-02,  4.1962e-04, -2.2852e-01,  4.0770e-05,
              6.8665e-05, -8.0078e-02, -5.8594e-02, -5.2979e-02, -1.8677e-02, -1.9989e-03,  3.3691e-02,  6.7871e-02,
              8.7891e-02,  1.5259e-04, -3.5645e-02,  5.1880e-03, -4.2236e-02,  1.3611e-02,  9.2285e-02,  4.6997e-03,
             -2.8133e-05, -3.2617e-01,  9.0332e-03, -1.8883e-04,  1.9287e-02, -2.2266e-01,  1.2268e-02,  6.4453e-02,
             -6.3477e-02, -1.0498e-01,  5.7068e-03, -1.2061e-01,  9.6321e-05,  6.0654e-04,  1.4648e-01,  1.6212e-04,
             -3.4180e-02, -8.1641e-01, -2.9883e-01, -1.2158e-01, -1.4258e-01, -1.2146e-02, -2.2461e-02,  4.4678e-02,
              6.8359e-02,  2.2221e-04,  8.2397e-04, -3.3203e-02,  3.7598e-02,  5.2002e-02,  2.9945e-04, -5.5552e-05,
             -6.8848e-02, -6.0547e-02,  1.5411e-03, -1.5442e-02,  5.2734e-02,  4.6921e-04,  2.0508e-01,  8.1062e-05,
              1.4591e-04,  2.8419e-04,  2.2583e-03, -4.1016e-01,  1.6504e-01,  3.6621e-02,  5.3223e-02, -1.6504e-01,
              6.9885e-03,  3.6133e-02,  2.7100e-02, -2.8076e-02, -4.5776e-03, -1.2268e-02, -9.6893e-04, -8.6670e-03,
              3.9453e-01,  1.8652e-01,  1.8158e-03,  6.2943e-05, -5.6885e-02,  1.6098e-03,  1.0205e-01,  6.7902e-04,
              1.0376e-02,  4.6692e-03,  4.4678e-02, -1.7643e-04,  4.3488e-04,  9.3750e-02, -1.8433e-02,  3.3875e-03,
             -4.1016e-02, -3.9307e-02,  2.9541e-02,  3.6133e-02,  9.6191e-02, -6.0791e-02, -6.2012e-02, -5.3955e-02,
              2.2705e-02, -5.5237e-03,  1.0315e-02,  4.7874e-04,  7.7724e-05, -7.8613e-02, -7.1777e-02,  1.6724e-02,
              8.7280e-03, -2.1851e-02, -6.4697e-03, -6.8848e-02, -1.8750e-01,  1.6895e-01, -5.5664e-02,  2.4796e-04,
             -1.6499e-04,  2.2852e-01, -5.9509e-03,  1.0400e-01,  4.4678e-02,  1.3516e+00, -2.8906e-01,  5.3467e-02,
             -7.0801e-02, -8.0490e-04, -1.8311e-02, -3.0136e-04,  1.5640e-04, -4.7363e-02, -7.2266e-02,  3.9795e-02,
             -1.2793e-01, -3.7193e-04, -5.5695e-04, -4.2480e-02, -5.1117e-04,  6.3477e-03,  2.5513e-02,  4.7607e-02,
              4.6631e-02,  3.6430e-04, -8.7891e-02,  2.6245e-02, -7.2021e-03, -9.5215e-03,  6.1768e-02, -1.1719e-01,
             -1.2360e-03,  1.1063e-03,  6.8359e-02, -1.4973e-04,  2.5749e-05, -3.3447e-02,  3.2471e-02, -4.1016e-02,
              2.6001e-02, -6.5994e-04, -1.9264e-04, -6.2500e-02,  7.9727e-04, -3.5667e-04, -2.7954e-02,  9.3750e-02,
              2.3438e-02,  3.9307e-02,  9.2773e-03,  7.4707e-02,  8.0872e-04, -1.5015e-02, -1.9169e-04, -5.5420e-02,
             -9.7656e-02, -1.3281e-01,  2.5940e-03, -9.1309e-02,  9.8877e-03, -6.2500e-02,  5.0545e-05,  2.6131e-04,
             -1.6937e-03,  4.1748e-02,  7.8678e-05, -2.8534e-03, -1.7090e-02,  2.8906e-01,  3.0029e-02, -6.9046e-04,
             -1.4591e-04, -1.2268e-02, -2.4414e-04,  1.3550e-02, -1.0010e-01, -5.4016e-03, -5.0049e-03,  1.9043e-02,
             -7.8613e-02, -5.8594e-03,  7.3730e-02,  8.4473e-02, -2.2888e-03, -6.7871e-02,  1.4725e-03, -8.8215e-05,
             -1.7334e-02,  2.6172e-01, -1.3770e-01, -3.2959e-02,  2.0142e-02,  3.5156e-02, -2.6978e-02,  2.6367e-02,
             -3.7109e-02,  1.8164e-01, -2.9883e-01, -5.4688e-02,  4.9072e-02, -7.9155e-05, -4.3213e-02, -4.3678e-04,
             -6.2988e-02, -2.0599e-04,  4.3945e-02,  2.0752e-03, -3.0859e-01, -8.5354e-05,  1.0742e-01,  9.2773e-03,
              5.3101e-03, -2.1875e-01,  1.6479e-02, -1.5564e-02,  4.7607e-03, -2.7832e-02, -8.9844e-02, -1.3885e-03,
             -7.3242e-02, -1.4746e-01, -1.3062e-02,  1.7929e-04,  3.7689e-03, -1.3611e-02, -1.8692e-03,  1.2695e-01,
             -7.8964e-04, -1.6406e-01,  7.5912e-04, -1.1730e-04,  1.4191e-03,  2.6489e-02,  1.9336e-01,  2.6611e-02,
              3.0029e-02, -7.9590e-02,  5.3711e-02, -2.5269e-02,  1.6699e-01, -1.3965e-01, -2.6172e-01, -2.9564e-04,
              6.0059e-02,  1.3542e-04,  6.6757e-04,  1.9989e-03,  2.8906e-01, -1.3184e-01, -4.2480e-02, -9.0820e-02,
             -1.2988e-01,  6.7749e-03, -1.3733e-02,  3.5248e-03, -2.2705e-02, -3.5742e-01, -8.6426e-02,  7.2754e-02,
             -9.8944e-06,  2.3340e-01,  1.1597e-02,  8.1635e-04, -4.3945e-03,  3.5400e-02, -1.9455e-04,  1.7242e-03,
             -2.3438e-01,  1.4062e-01, -2.7148e-01,  2.7734e-01, -7.0953e-04,  1.2158e-01, -2.0264e-02, -1.5320e-02,
              3.6621e-03, -2.8320e-01, -8.9111e-03,  1.3962e-03,  6.9141e-05, -2.5940e-04,  1.9653e-02, -5.6458e-04,
             -5.9570e-02, -2.9449e-03,  3.6469e-03,  3.8574e-02,  1.4453e-01, -1.8597e-04, -4.5703e-01, -8.3447e-05,
              1.5332e-01, -1.6992e-01, -4.0039e-02,  4.9219e-01,  1.2891e-01, -8.4839e-03, -4.6349e-04, -4.3392e-05,
              2.2949e-01,  4.0817e-04, -1.1015e-04,  1.6403e-04,  4.6143e-02, -2.8534e-03, -1.3504e-03, -2.6733e-02,
              1.1658e-02, -1.4663e-05,  5.4932e-02, -6.3782e-03, -1.6846e-02,  1.4954e-02,  1.3489e-02, -5.8594e-02,
              2.8992e-04, -4.4632e-04, -8.4961e-02,  1.8066e-02, -1.3351e-04,  1.4572e-03, -1.2970e-04,  3.0029e-02,
             -1.6211e-01, -9.9609e-02,  1.1182e-01, -1.0681e-02,  1.4343e-02, -4.7913e-03,  7.2327e-03, -5.6152e-02,
              9.2163e-03, -2.5879e-02, -6.7139e-04, -3.2715e-02, -5.9814e-02,  6.7139e-04,  3.3379e-04,  4.4678e-02,
              3.9795e-02,  7.7148e-02, -3.1494e-02,  4.5898e-02, -2.4223e-04, -2.8320e-01, -1.1475e-01, -2.3438e-01,
             -8.4839e-03,  6.2500e-02,  6.3477e-03,  6.6757e-04,  4.7607e-03, -8.7891e-02, -1.5831e-04,  8.9355e-02,
             -5.1498e-04,  2.3535e-01,  1.2988e-01,  1.6174e-03, -9.0790e-04,  7.5378e-03,  4.6143e-02,  3.3936e-02,
             -1.6724e-02, -6.2256e-03, -1.3489e-02,  5.9814e-02, -9.2773e-02, -3.1738e-03,  1.0205e-01, -4.5002e-06,
             -4.8828e-03,  8.1055e-02,  1.7944e-02,  3.0151e-02,  6.8848e-02, -4.5586e-04,  1.9302e-03,  8.1543e-02,
             -3.2422e-01,  6.8188e-05, -7.6294e-04,  5.6152e-03,  3.4180e-02, -2.6367e-01, -5.3223e-02, -9.3460e-04,
              7.0190e-03,  8.5068e-04,  5.4688e-02,  3.3203e-02,  6.7383e-02,  5.3711e-03, -6.6528e-03,  5.6885e-02,
              1.2634e-02,  4.1748e-02,  1.8845e-03,  4.7852e-02,  3.9795e-02, -2.9297e-02, -1.0872e-04, -5.5908e-02,
              4.2969e-01, -1.0437e-02, -9.1553e-04, -6.8665e-04, -3.9062e-02,  1.7700e-02, -4.1016e-01, -1.6846e-02,
              4.5776e-04,  4.1992e-02, -5.5176e-02, -2.2697e-04, -3.2043e-04,  8.2779e-04, -3.5889e-02,  1.5820e-01,
              6.8359e-03, -8.3984e-02, -1.8387e-03, -3.2959e-02,  2.0703e-01, -6.5430e-02, -1.6880e-04,  1.2756e-02,
             -1.3351e-03,  1.3733e-02,  1.0254e-01, -7.8201e-04, -3.4668e-02, -4.3213e-02, -2.9297e-02, -1.1084e-01,
             -1.1328e-01, -1.1963e-01,  3.0151e-02,  1.8311e-02,  4.0771e-02,  4.8340e-02,  1.3184e-01, -2.3193e-03,
              3.7500e-01,  1.1597e-03, -4.4678e-02, -1.0681e-02,  6.0547e-02,  2.0752e-03,  8.5449e-02, -5.2490e-02,
             -1.0547e-01, -7.4219e-02, -1.9141e-01, -2.4292e-02,  1.5076e-02,  2.2339e-02,  1.9897e-02,  3.4332e-04,
             -3.1982e-02,  1.2695e-02, -2.2168e-01, -2.6367e-01, -3.4180e-02,  2.4536e-02, -6.2500e-02,  2.2339e-02,
             -2.6703e-04,  9.3750e-02,  1.3733e-02, -1.5625e-02,  1.0147e-03,  1.9409e-02, -1.0693e-01, -4.9609e-01,
              2.8564e-02, -3.6865e-02, -1.1377e-01, -6.5918e-02,  3.6133e-02, -7.0190e-04, -7.0801e-02, -4.9744e-03,
              3.1128e-02, -4.0770e-05, -2.7954e-02,  4.4861e-03,  2.3071e-02,  1.0559e-02,  8.9722e-03, -2.6245e-02,
              8.5449e-02,  1.5564e-02, -9.5215e-03,  1.7700e-02, -4.3297e-04, -2.5146e-02,  2.4048e-02, -1.0834e-03,
              7.4219e-02,  7.8125e-02,  3.0884e-02,  3.6377e-02,  9.5215e-02,  1.8311e-02,  9.7168e-02, -4.9438e-03,
             -1.1536e-02,  1.8311e-03, -1.9043e-02, -8.6426e-02, -1.0010e-02,  1.3184e-01, -3.3140e-05, -1.0449e-01,
              2.3499e-03,  3.7354e-02,  4.3869e-04, -4.2480e-02,  7.7820e-04,  1.0681e-04,  5.7983e-03,  9.5215e-02,
              4.0039e-02,  1.5234e-01,  1.8555e-01,  2.8229e-04, -9.8419e-04, -8.3496e-02,  1.0132e-02,  5.1575e-03,
              1.1133e-01,  1.8433e-02,  9.1553e-05, -4.9438e-03,  8.3496e-02,  5.3101e-03, -6.1989e-05,  2.0264e-02,
             -1.5869e-03,  1.3123e-02,  8.8811e-06,  3.2500e+00,  1.0910e-03,  9.5215e-03,  5.1514e-02,  8.4961e-02,
              4.5013e-04,  3.4424e-02,  4.4434e-02,  5.9814e-03, -5.8594e-03,  1.6357e-02,  7.7148e-02, -2.2125e-04,
              1.9653e-02,  3.1128e-02, -2.4707e-01, -8.8379e-02,  8.3496e-02, -1.2061e-01, -3.0884e-02, -2.5940e-03,
             -2.7588e-02,  1.6968e-02,  2.0703e-01, -3.1982e-02,  3.9307e-02,  8.0490e-04,  2.1484e-01,  5.7373e-03,
              1.9908e-05, -3.6955e-05, -2.6978e-02, -9.0820e-02,  6.9336e-02, -3.1738e-02,  3.9482e-04, -4.7363e-02,
             -9.3460e-04,  5.8899e-03,  8.6060e-03,  4.0283e-02, -1.2970e-04, -3.0151e-02, -2.7100e-02, -1.5020e-05,
              3.8147e-03,  1.3590e-05,  1.0303e-01, -6.5308e-03, -5.2246e-02, -1.5259e-03, -3.7998e-06,  3.8300e-03,
             -8.2493e-05, -7.3242e-02,  4.5166e-02,  1.1169e-02,  1.4725e-03, -3.1494e-02, -1.0014e-04, -3.8086e-02,
             -4.0039e-02,  5.4932e-04,  6.1328e-01, -1.0986e-01,  3.1090e-04,  6.1417e-04,  3.2227e-02, -5.2002e-02,
             -2.4536e-02,  2.6123e-02, -5.1514e-02,  8.0566e-02, -3.0518e-02,  3.6240e-04, -6.2988e-02,  2.6703e-05,
              7.1777e-02,  6.0654e-04, -7.6660e-02, -1.0449e-01, -2.5586e-01, -4.5898e-02,  1.4551e-01, -4.1962e-04,
              4.7112e-04, -7.3624e-04,  7.7057e-04, -1.0300e-03,  9.2316e-04, -6.0120e-03,  2.4658e-02,  3.7575e-04,
              8.1055e-02,  1.2158e-01,  4.5967e-04,  5.0735e-04,  6.4453e-02, -1.8677e-02, -1.2988e-01, -3.8719e-04,
              5.9375e-01, -8.6308e-05,  3.5400e-02, -3.5400e-02,  5.9766e-01, -9.4727e-02, -1.0791e-01, -2.2430e-03,
              2.4170e-02, -5.2734e-02, -3.3936e-02,  1.2589e-03, -5.0354e-04, -8.4961e-02,  9.1309e-02, -7.8613e-02,
             -2.5391e-02, -4.1008e-04, -2.4536e-02, -1.2500e-01, -6.6223e-03,  5.4550e-04, -1.0449e-01, -2.4719e-03,
              3.3691e-02, -9.6436e-03,  1.2988e-01,  1.0132e-02, -1.1475e-01, -6.8359e-02, -1.9336e-01,  8.5449e-02,
             -1.2878e-02, -2.5368e-04,  2.3071e-02,  1.6235e-02, -1.2283e-03,  1.9897e-02, -1.0938e-01, -1.2812e+00,
              4.3945e-01,  5.9570e-02, -1.4746e-01,  1.6022e-03,  2.9297e-02,  1.2598e-01, -2.1057e-03,  3.6865e-02,
              2.0885e-04, -2.6172e-01, -4.3640e-03,  2.8564e-02, -5.2929e-05, -2.0898e-01,  7.2956e-05,  1.0889e-01,
             -6.8359e-03, -1.3428e-03, -3.8086e-02,  4.2725e-02,  1.5442e-02,  9.9609e-02,  4.0527e-02,  4.0527e-02,
              1.5332e-01, -5.1758e-02,  7.2002e-05,  9.2387e-06,  7.2266e-02, -2.1118e-02, -8.9844e-02,  3.7193e-05,
             -3.5889e-02, -4.8828e-02, -2.5000e-01,  2.8076e-03, -1.3123e-03, -1.4221e-02, -5.4550e-04, -6.3477e-02,
             -9.2773e-02,  1.8066e-02, -7.8613e-02, -7.2021e-03, -1.8555e-01, -5.6458e-03,  1.2354e-01,  3.4912e-02,
             -1.1035e-01,  4.0771e-02, -2.8229e-04,  2.2339e-02,  2.3346e-03, -2.1973e-03, -1.2598e-01, -4.5654e-02,
             -1.8692e-04, -3.0060e-03,  1.3281e-01, -5.9082e-02,  2.2339e-02, -1.2500e-01,  1.1230e-02, -8.0872e-04,
             -3.7003e-04, -5.7129e-02,  1.1328e-01,  6.0791e-02,  3.3951e-04, -1.8799e-02,  5.6885e-02,  3.6328e-01,
             -2.5749e-04,  1.4267e-03,  2.0874e-02, -4.6875e-02,  1.2451e-02,  3.0664e-01,  3.9978e-03, -3.9062e-02,
              2.5513e-02,  4.1797e-01, -2.9419e-02,  1.9646e-04, -2.8687e-02,  2.6367e-01, -2.7222e-02,  1.1444e-03,
              4.5166e-02, -7.2754e-02,  1.6357e-02,  1.2305e-01, -5.8350e-02, -4.5410e-02, -5.4626e-03,  1.9409e-02,
             -5.3711e-02,  1.4782e-04,  2.0142e-02, -3.2471e-02, -8.9722e-03, -3.4180e-02, -2.5391e-02,  3.8086e-02,
              4.1748e-02,  2.0752e-02,  6.8359e-02, -1.5080e-05, -2.4707e-01,  2.7084e-04, -4.2725e-03, -3.5095e-03,
             -2.2827e-02, -6.3477e-02,  7.2754e-02, -3.6133e-02, -1.3504e-03,  5.2734e-02, -4.2480e-02,  3.3379e-06,
             -9.6680e-02,  2.6398e-03, -5.4016e-03, -1.6174e-03,  9.5749e-04, -5.0049e-03, -2.3926e-02, -1.8066e-02,
             -1.4114e-03,  1.5354e-04, -2.3560e-02, -2.7381e-07, -2.1048e-07, -5.6396e-02, -2.3499e-03,  4.7119e-02,
              3.9291e-04, -9.0027e-04, -1.6479e-02, -2.1973e-02,  2.1191e-01,  1.9897e-02,  2.1076e-04,  4.0894e-03,
             -7.3624e-04,  2.6093e-03,  1.4844e-01, -9.0820e-02, -6.9336e-02, -2.7771e-03, -1.2988e-01,  3.6377e-02,
             -7.7148e-02, -4.5410e-02,  1.5991e-02,  2.4658e-02,  2.3041e-03,  2.9419e-02, -9.8145e-02, -7.8201e-05,
             -3.1982e-02,  1.8463e-03,  1.1749e-03, -1.1597e-02, -3.1738e-02,  8.6060e-03,  4.1389e-04,  5.6763e-03,
             -1.0303e-01, -1.1325e-05,  3.6049e-04,  1.5640e-04, -4.0283e-02,  4.9316e-02, -4.6387e-02, -1.8215e-04,
              3.4809e-05,  4.3945e-03, -2.3560e-02,  9.9121e-02, -1.8311e-03, -2.4109e-03,  1.6016e-01,  2.6245e-02,
             -2.2736e-03,  2.8992e-03, -1.0938e-01, -1.3916e-02,  6.1523e-02,  3.1836e-01,  2.0215e-01,  4.1008e-04,
             -8.1055e-02, -5.5695e-04,  2.0508e-01,  6.9824e-02, -6.0730e-03, -8.8867e-02, -7.3624e-04, -4.8256e-04,
             -1.1292e-02,  4.0245e-04, -9.6893e-04,  1.1158e-04, -2.8931e-02,  5.1514e-02, -3.2227e-02,  1.0400e-01,
             -3.0212e-03, -4.7607e-03, -1.1658e-02,  1.5198e-02, -6.7749e-03, -1.4160e-01, -4.1389e-04, -8.3160e-04,
              1.1914e-01,  3.3188e-04,  1.7090e-01, -8.0490e-04,  3.9795e-02,  3.0640e-02, -1.1063e-03,  6.3705e-04,
              1.5156e+00, -4.7363e-02, -3.9795e-02, -4.6094e-01,  2.0605e-01,  4.1199e-03,  4.9072e-02, -3.8528e-04,
              3.9648e-01,  8.5938e-02,  1.0986e-01, -1.6975e-04, -9.5215e-03,  5.4443e-02, -2.5024e-02, -7.0801e-02,
              1.3447e-04,  4.2114e-03, -2.4170e-02,  3.6812e-04, -2.8992e-04, -4.5776e-04,  7.7637e-02, -5.6885e-02,
             -3.2471e-02,  1.6602e-01, -1.1133e-01, -7.6904e-03, -1.0376e-03,  2.3560e-02, -1.6235e-02, -9.0942e-03,
              2.5630e-05, -2.4292e-02, -7.8964e-04, -6.5308e-03,  1.0596e-01,  2.0027e-04,  5.8350e-02, -7.7438e-04,
              3.3398e-01, -2.5635e-02,  6.0425e-03,  8.5449e-03, -5.6885e-02,  6.7139e-04,  4.7302e-03, -6.3782e-03,
              5.5790e-05, -1.9775e-02, -1.7624e-03,  8.3008e-02,  1.8359e-01,  2.8992e-04, -6.4697e-03, -1.1963e-02,
             -1.0315e-02,  6.4373e-06,  1.8616e-03, -5.8594e-01, -7.1526e-05,  1.6846e-02, -5.5664e-02, -2.1387e-01,
              6.9427e-04, -7.5073e-03,  3.6133e-02,  1.2500e-01, -1.6992e-01, -2.0027e-05, -1.1108e-02, -4.1260e-02,
              7.6834e-08, -1.7166e-04,  6.0938e-01, -4.3701e-02, -4.2969e-02,  4.5410e-02,  6.4453e-02,  6.6406e-02,
             -2.1973e-01,  1.1520e-03,  1.3867e-01,  4.5967e-04,  1.4062e-01, -2.1973e-03, -6.7711e-05,  3.5156e-02,
              1.2695e-01, -2.6398e-03,  3.5095e-03,  7.0801e-02,  2.0981e-05, -9.4727e-02, -1.7014e-03, -7.0496e-03,
              4.4189e-02,  4.2439e-05,  4.9561e-02,  9.5215e-03,  2.0605e-01, -2.6550e-03,  1.5332e-01,  3.6133e-02,
              5.6396e-02,  4.6484e-01, -2.6512e-04,  1.7578e-01, -5.2691e-05, -1.0596e-01,  1.4832e-02, -2.8931e-02,
             -1.9165e-02,  4.4189e-02, -1.2598e-01, -8.9645e-04, -8.7738e-04, -1.7822e-02,  7.8735e-03, -3.6621e-02,
              4.1008e-04, -5.0781e-02,  3.0975e-03, -2.2705e-02,  3.3594e-01,  1.2939e-02, -1.3550e-02, -3.5095e-03,
             -1.6689e-05, -4.0039e-02,  7.6172e-02, -2.0020e-02,  1.1536e-02,  4.0436e-04,  4.6387e-02, -3.2227e-02,
              9.5215e-02, -6.4941e-02, -4.5410e-02,  1.2517e-05, -3.4523e-04, -7.1777e-02,  8.0466e-06, -7.4463e-03,
             -2.0599e-03,  1.0205e-01,  6.7188e-01, -1.7090e-02, -3.3691e-02,  5.1498e-04,  2.7275e-04, -1.0303e-01,
              2.1191e-01, -2.5757e-02,  2.3193e-02,  4.1992e-02,  4.5586e-04, -4.7607e-03, -3.1128e-03,  9.3384e-03,
             -4.4060e-04, -2.9785e-02,  2.7588e-02,  4.6082e-03, -3.1128e-02,  4.4861e-03, -5.0735e-04,  5.0354e-04,
              3.7432e-05, -5.3101e-03,  1.9531e-02, -1.4343e-03, -9.0332e-02,  1.3000e-02,  3.1738e-02, -2.7275e-04,
             -2.3651e-04, -6.5430e-02,  2.0752e-02,  2.4414e-04, -5.0049e-03, -7.0312e-02,  1.1368e-03, -7.7515e-03,
             -1.1902e-02,  1.0620e-02, -1.4062e-01,  1.2283e-03,  6.0938e-01,  1.4648e-01, -5.9082e-02, -1.8359e-01,
              3.0664e-01,  1.1780e-02, -5.4626e-03, -1.4062e-01,  3.3569e-04,  5.1025e-02,  3.3951e-04, -1.0449e-01,
              2.6367e-02, -1.4801e-03,  1.8921e-03,  1.0864e-02,  4.4060e-04, -4.6680e-01,  5.1270e-02, -7.8735e-03,
              4.9072e-02, -5.0537e-02, -1.0300e-04, -6.2988e-02,  4.2969e-01,  2.3804e-03, -8.7357e-04, -9.4238e-02,
             -5.2734e-02, -2.5272e-05,  1.4746e-01,  5.7617e-02,  1.5259e-04, -8.0872e-04, -1.6479e-03, -2.5977e-01,
              1.6403e-03, -3.1948e-05, -1.0395e-04, -8.7402e-02,  3.1738e-02, -1.1587e-04, -3.5248e-03,  5.4443e-02,
             -5.2185e-03, -1.1353e-02,  1.4282e-02,  2.7466e-02,  1.5488e-03,  1.5564e-02,  3.9062e-02,  5.2246e-02,
              4.3945e-02,  9.2697e-04, -1.0400e-01, -1.8433e-02, -6.9336e-02,  2.9602e-03, -6.7383e-02, -5.1558e-06,
              1.0910e-03,  4.1723e-05,  3.1055e-01,  6.0547e-02, -1.0147e-03, -1.0443e-04,  1.0400e-01, -6.6406e-02,
             -2.6978e-02, -1.7700e-02,  3.3569e-03, -2.0874e-02, -1.2589e-03, -3.6774e-03,  1.1475e-01,  7.6660e-02,
              6.3965e-02,  1.2109e-01,  1.8024e-04,  2.0874e-02,  4.6631e-02,  2.4033e-04,  9.3994e-03,  2.2583e-02],
            [-4.7119e-02,  5.4321e-03, -9.9487e-03,  5.7373e-02, -9.0332e-03, -1.8311e-02,  1.6212e-05,  8.4839e-03,
             -1.4465e-02, -2.2339e-02, -1.6602e-02,  4.8584e-02,  4.9072e-02,  1.6632e-03,  6.0797e-05, -4.9805e-02,
              2.7466e-02,  3.1494e-02, -1.3065e-04,  7.0801e-02, -8.6426e-02,  1.8555e-02,  1.2493e-04,  2.2125e-03,
              4.7607e-03, -3.3936e-02,  1.7071e-04, -6.4373e-05, -4.8828e-02,  7.5340e-05, -8.0566e-03, -1.0938e-01,
             -5.7373e-03, -6.6406e-02, -2.3926e-02,  1.0376e-02,  8.9722e-03, -1.6235e-02, -1.3550e-02, -4.7913e-03,
              7.2937e-03,  3.0160e-05, -1.5335e-03, -1.5564e-03,  2.9755e-03, -1.0254e-02, -4.0894e-03,  1.1353e-02,
              5.4836e-05,  6.8848e-02,  9.0332e-03, -1.5450e-04,  3.4027e-03,  2.9945e-04,  8.6427e-06,  2.7100e-02,
              7.5698e-06,  1.8997e-03,  5.5790e-05, -1.0061e-04, -5.3024e-04,  1.3733e-04,  2.1851e-02, -1.0925e-02,
             -3.3691e-02,  5.8365e-04,  4.6082e-03,  9.9121e-02,  4.8096e-02,  3.1441e-06, -1.7452e-04, -3.0327e-04,
             -3.2715e-02, -1.9150e-03,  2.5940e-04, -9.8419e-04,  1.4526e-02,  2.9053e-02,  2.8687e-02,  7.6294e-03,
              2.6321e-04, -3.1128e-02, -5.0049e-03, -8.7891e-03,  3.3379e-04,  6.3419e-05, -1.9073e-03, -3.3936e-02,
              2.7100e-02, -2.1973e-02, -9.4604e-03,  2.3193e-02, -5.6152e-02, -5.4199e-02,  4.0527e-02,  5.0781e-02,
             -5.0049e-02, -9.4604e-03,  5.1575e-03,  1.3672e-02, -2.6855e-02, -4.3164e-01,  3.1006e-02,  1.0840e-01,
              1.4282e-02,  2.1362e-02,  3.2715e-02, -3.3936e-02,  4.0588e-03,  3.2227e-02, -1.7212e-02, -3.1233e-05,
              2.2949e-02, -6.2561e-04, -1.9775e-02, -3.0151e-02,  1.1426e-01,  8.0078e-02, -2.9053e-02,  3.8818e-02,
             -2.8687e-02, -4.0771e-02, -3.8147e-03,  3.8086e-02,  6.8359e-02, -7.0572e-04, -7.7209e-03,  2.2827e-02,
              1.7700e-02,  2.1057e-03, -3.5156e-02,  1.6602e-02,  1.1742e-05, -3.3936e-02, -6.3477e-02, -5.7678e-03,
              5.1270e-02,  6.3171e-03, -1.7700e-02, -2.0027e-04, -7.5817e-05,  1.2512e-02,  7.1153e-07, -1.3053e-05,
             -1.1035e-01, -1.1063e-03,  9.4604e-03, -4.3030e-03,  3.9673e-04, -5.9509e-03,  2.8320e-01,  2.4676e-05,
              4.5967e-04,  4.1748e-02,  1.3062e-02,  9.2030e-05,  6.6833e-03,  1.2207e-02,  5.1498e-04,  8.3008e-02,
             -1.2817e-02, -1.2878e-02,  1.4496e-04, -6.3782e-03,  1.8066e-02, -1.4941e-01,  5.9509e-03,  9.8267e-03,
              5.8365e-04,  1.1768e-01,  4.1748e-02, -4.2915e-06,  1.1230e-02,  6.4453e-02, -5.9326e-02,  2.8381e-03,
             -3.5645e-02,  1.2207e-01,  1.9409e-02, -1.0910e-03,  6.5002e-03,  3.7003e-04, -2.2095e-02,  1.1230e-01,
              1.6968e-02,  2.3346e-03,  1.1182e-01, -6.1768e-02,  3.6621e-03,  3.7439e-07, -3.1738e-02,  2.1729e-02,
              2.6733e-02, -8.3618e-03,  1.6846e-02,  8.5449e-03,  6.1768e-02, -1.0254e-02,  6.5430e-02, -4.2969e-01,
             -7.3242e-04, -1.7456e-02, -4.6492e-06,  3.4180e-02,  1.9653e-02, -4.9133e-03,  1.0010e-02, -1.3924e-04,
             -6.6223e-03, -4.1992e-02,  1.8433e-02,  3.0279e-05, -1.2112e-04, -7.4387e-05,  1.1902e-02, -1.0498e-02,
              7.5817e-05, -2.4719e-03,  1.3770e-01,  1.3504e-03,  6.9809e-04,  1.1475e-02, -1.2064e-04, -4.2725e-04,
             -3.7842e-03,  7.2861e-04,  1.1536e-02,  4.7302e-03,  3.1471e-04,  5.9326e-02,  3.5400e-03,  6.6528e-03,
              3.9062e-02,  3.7354e-02,  2.0752e-02,  1.0132e-02,  4.2480e-02,  2.3193e-03,  9.8633e-02, -1.5926e-04,
             -6.4087e-03, -3.6621e-02, -9.7275e-05,  5.0293e-02,  1.9684e-03,  2.9297e-03, -8.2493e-05, -6.0272e-04,
             -7.6660e-02,  1.1902e-02, -1.4496e-03,  6.3324e-04, -3.7842e-02,  1.7624e-03,  6.1768e-02, -1.1658e-02,
              5.2185e-03,  3.2616e-04,  3.7766e-04, -4.1992e-02, -3.8147e-03, -1.4114e-03,  8.4839e-03, -6.6833e-03,
              2.4512e-01, -5.9891e-04,  2.5024e-02,  5.7068e-03,  3.6163e-03,  1.0803e-02, -4.7607e-02, -4.0436e-04,
             -1.3086e-01, -1.7212e-02,  1.0803e-02,  1.0078e+00, -7.6904e-03, -3.0151e-02,  9.0332e-03, -2.7008e-03,
             -4.2419e-03,  8.8501e-03,  1.7578e-02,  7.4463e-03,  1.5869e-03, -2.4536e-02,  2.3127e-05,  9.2773e-02,
              5.5542e-03,  5.4688e-02,  1.4160e-01,  5.9509e-04, -2.8320e-02,  9.7156e-06,  1.0889e-01, -1.7969e-01,
             -8.2031e-01, -2.2583e-02,  5.3101e-03,  4.1992e-02, -5.7459e-05, -6.9336e-02,  6.5613e-03, -1.2970e-03,
              8.5831e-04, -8.3923e-04,  5.5075e-05, -2.3315e-02,  1.9409e-02, -2.1515e-03, -6.1035e-02, -3.7354e-02,
             -2.2583e-02,  1.5259e-02, -4.4189e-02, -4.0771e-02,  1.1139e-03,  2.4414e-02,  2.8419e-04,  4.0527e-02,
             -3.1641e-01,  2.1362e-03,  2.3071e-02, -2.1362e-02,  1.2402e-01,  1.5015e-02, -3.1433e-03,  1.0803e-02,
             -1.5564e-02, -2.4536e-02, -1.5488e-03,  7.0801e-02,  9.4414e-05,  1.7456e-02, -9.2285e-02, -7.5989e-03,
             -1.9455e-03, -9.6191e-02,  7.5378e-03, -3.0762e-02,  9.5215e-02, -3.8086e-02, -2.4292e-02,  3.2806e-04,
              3.3203e-02, -5.2246e-02,  9.2773e-03, -5.6076e-04,  1.8921e-02, -5.2002e-02,  2.1973e-02, -2.7100e-02,
              1.3770e-01,  2.8229e-04, -1.4114e-04,  1.2878e-02,  4.9133e-03, -4.6968e-05, -9.3994e-03, -7.6294e-05,
             -4.2236e-02, -1.6968e-02,  1.4709e-02, -4.5166e-03,  5.7983e-04,  8.3008e-03,  9.1797e-02, -2.5024e-02,
              1.9646e-04, -2.3746e-04,  9.2773e-03,  2.7161e-03,  4.4861e-03, -1.1719e-02,  6.4087e-03, -1.0376e-02,
              3.2715e-02,  4.8065e-04, -1.6594e-04, -2.1729e-02,  1.5545e-04,  7.4387e-05, -8.2520e-02, -7.4387e-05,
              2.9602e-03, -1.8311e-04,  1.5625e-01,  3.9307e-02, -3.0273e-02,  2.5511e-05, -7.1335e-04, -4.3030e-03,
              3.3379e-04,  4.2236e-02, -8.0078e-02, -1.3962e-03,  1.0400e-01,  1.5030e-03,  1.5564e-02, -2.0142e-02,
             -9.2163e-03, -5.2490e-03,  1.3977e-02, -3.2997e-04, -4.8828e-02, -1.8997e-03, -4.9805e-01,  2.3746e-04,
             -6.1279e-02,  8.4839e-03, -1.2793e-01,  3.3203e-02,  4.1199e-03,  3.4523e-04, -1.5503e-02, -5.9204e-03,
              8.5354e-05,  2.0264e-02, -3.9307e-02,  1.6846e-02, -5.6641e-02, -4.5898e-02,  3.0518e-02, -6.2256e-03,
             -1.7834e-04,  2.5513e-02, -1.0681e-02,  3.4180e-02, -3.1471e-05,  9.5367e-05, -3.2654e-03,  1.2268e-02,
              4.8828e-02, -2.0020e-02,  8.4839e-03, -8.6060e-03, -8.6308e-05, -1.5625e-01, -4.1962e-05,  8.9844e-02,
             -5.4169e-04, -1.9989e-03,  6.0654e-04,  2.2697e-04, -4.5166e-02,  1.5068e-04,  1.0254e-02, -9.7656e-04,
              4.4922e-02, -7.7438e-04,  3.9368e-03, -8.8867e-02, -8.2397e-03, -3.9795e-02, -1.9836e-03,  1.7807e-06,
             -2.3315e-02,  2.3270e-04,  3.0975e-03,  9.0599e-05, -2.8809e-02,  3.5742e-01,  4.7607e-02,  9.5215e-03,
              1.5259e-02, -7.8125e-03, -4.4823e-04,  2.6489e-02,  1.8555e-01,  5.5664e-02, -2.3438e-02,  3.8818e-02,
             -1.8066e-02, -1.6212e-04, -1.5137e-01,  2.9755e-04,  1.5015e-02, -1.2256e-01,  1.8387e-03,  1.3281e-01,
             -3.0518e-03, -5.5313e-04, -3.7670e-05,  9.8267e-03, -1.3367e-02, -5.1575e-03,  7.3433e-05,  3.4912e-02,
             -3.9062e-02, -3.7354e-02, -8.0566e-03,  6.5430e-02,  2.1851e-02,  7.8125e-02,  1.1719e-02, -6.6406e-02,
              9.1309e-02,  8.4839e-03, -2.6489e-02, -8.3008e-02,  8.4473e-02, -9.3460e-05, -1.4062e-01,  1.2517e-06,
             -3.1982e-02,  5.4626e-03, -2.4658e-02, -9.2285e-02,  2.0146e-05,  2.6093e-03, -2.1606e-02, -4.3945e-03,
              2.1362e-02,  3.6865e-02, -3.0518e-05,  6.6016e-01, -1.9165e-02, -3.4424e-02, -4.9072e-02, -2.5635e-03,
             -5.5237e-03,  6.7871e-02, -1.4221e-02, -1.3733e-02,  3.5477e-04, -2.1973e-02, -1.3794e-02,  5.5908e-02,
              1.0645e-01,  2.2125e-03, -1.6022e-04, -7.3547e-03,  4.5471e-03,  1.6724e-02, -3.6621e-02, -7.9346e-04,
             -9.5963e-06,  9.1797e-02,  1.1597e-02, -3.9795e-02, -2.9663e-02, -2.7954e-02, -5.3406e-04, -1.3477e-01,
             -2.0630e-02, -7.1777e-02,  2.1696e-05,  7.5531e-04, -5.6028e-05,  2.5024e-03, -1.1475e-02, -5.9814e-03,
             -2.9445e-05,  3.9795e-02, -1.8066e-02, -1.4496e-04, -2.5177e-04, -9.6436e-03, -3.0762e-02,  1.4007e-05,
              7.9102e-02, -1.4551e-01, -7.4387e-05,  1.7285e-01,  5.1575e-03,  3.4180e-02, -5.9814e-02, -2.8992e-04,
             -9.9945e-04, -3.7384e-04, -5.3101e-03, -8.5831e-05, -2.1820e-03, -4.1809e-03, -1.1749e-03,  3.3936e-02,
              5.0781e-02,  5.6763e-03,  1.9775e-02,  2.9373e-04, -1.2875e-04, -5.0293e-02, -1.2398e-04, -2.6093e-03,
              1.2329e-02, -5.5237e-03,  2.6123e-02, -2.2430e-03,  4.1504e-03,  3.0708e-04,  6.9737e-06, -1.5545e-04,
              2.3071e-02,  1.9670e-05, -1.7944e-02,  8.4229e-03,  4.5395e-04,  4.1992e-02,  1.3855e-02, -1.4062e-01,
             -3.1494e-02, -3.4523e-04, -1.2589e-04,  3.4790e-03, -3.3264e-03, -6.9046e-04,  9.1934e-04, -5.5176e-02,
             -2.0996e-01,  2.2461e-02,  6.0797e-05,  3.5742e-01,  3.3722e-03,  1.1572e-01, -5.8289e-03, -2.8442e-02,
              6.7139e-03, -1.1520e-03, -2.4414e-02, -2.4902e-02,  1.6251e-03,  4.0588e-03, -6.0797e-05, -8.8692e-05,
              3.8330e-02,  5.6982e-05, -2.2949e-02,  7.9956e-03,  5.7983e-03, -2.8320e-02,  7.5150e-04, -2.1210e-03,
              6.3330e-07,  3.7842e-02, -7.8125e-02, -6.6895e-02,  1.7262e-04,  7.4863e-05, -8.5449e-03, -5.1270e-02,
              3.5553e-03,  1.7548e-04, -8.4877e-05,  1.0681e-02,  4.1504e-02, -2.4048e-02,  2.2697e-04, -5.9204e-03,
             -7.1716e-03,  1.0400e-01,  7.8678e-06, -2.7275e-04,  5.4598e-05,  3.7842e-02, -2.7222e-02, -4.4441e-04,
              3.2471e-02, -1.4648e-03, -8.4839e-03,  1.2256e-01,  9.5825e-03, -1.4156e-06, -2.8320e-02,  1.4941e-01,
              5.9570e-02,  8.6914e-02, -3.5400e-02, -4.1485e-05,  4.1962e-04, -1.2684e-04, -2.6001e-02,  1.6022e-04,
              4.8828e-03, -7.6660e-02,  5.7602e-04,  9.0122e-05, -2.9175e-02,  2.0508e-02,  6.1417e-04, -4.1406e-01,
              7.5195e-02,  7.0190e-03,  1.2894e-03,  6.6833e-03,  8.0566e-03,  3.1281e-03, -3.1982e-02, -2.9449e-03,
              1.4305e-04,  9.2773e-02, -7.9346e-04, -2.3315e-02,  1.9409e-02,  7.4219e-02,  1.9836e-03,  2.7847e-04,
             -1.4572e-03,  8.0109e-05,  4.0771e-02,  1.1536e-02,  5.2261e-04, -1.2146e-02,  1.1816e-01,  2.4567e-03,
             -5.3711e-02,  8.4877e-05, -1.0071e-02, -5.5237e-03, -5.6839e-04, -1.4305e-04, -5.2979e-02, -1.8677e-02,
              1.0681e-02, -4.5898e-02, -6.0272e-04,  1.4551e-01, -5.9009e-06,  2.8516e-01, -3.3691e-02, -2.0885e-04,
              1.8555e-01, -2.5024e-02,  1.5991e-02,  7.3242e-02, -1.9989e-03, -2.1172e-04,  2.2125e-03, -5.2734e-02,
              4.3164e-01,  2.8198e-02,  1.4877e-04, -2.7847e-04,  5.5790e-05,  2.2030e-04,  3.7109e-01,  2.2583e-02,
             -9.4727e-02,  2.2217e-02, -1.3275e-03,  1.0400e-01, -2.8076e-03, -1.4160e-02, -5.3024e-04,  1.3550e-02,
             -2.5635e-03,  5.7602e-04,  3.2959e-02,  1.8692e-04,  2.6550e-03, -3.0060e-03, -1.7738e-04, -5.7861e-02,
             -1.3580e-03,  8.6060e-03, -2.5177e-04, -8.4473e-02, -1.8005e-03, -1.0559e-02,  2.8442e-02, -8.3008e-03,
              1.8539e-03,  1.9684e-03, -8.2397e-04,  1.5076e-02,  3.0365e-03, -1.4771e-02,  1.6406e-01, -5.5420e-02,
              6.5613e-03, -3.8574e-02, -1.2207e-02,  6.3965e-02,  3.8330e-02, -1.1658e-02, -1.6451e-05,  5.0354e-03,
              1.0376e-02,  2.9541e-02, -2.0752e-02,  7.9102e-02, -6.4850e-05,  1.2398e-05, -7.8125e-02, -2.2559e-01,
              1.0681e-03,  4.7363e-02, -2.2949e-02, -5.4932e-03,  3.3379e-05, -1.5640e-03,  2.1484e-02, -1.5640e-03,
             -2.2411e-04,  1.5020e-05,  3.5286e-05,  4.8876e-05, -3.1471e-04,  9.6798e-05, -1.1169e-02,  2.5024e-03,
             -4.9973e-04,  2.3193e-02,  2.5635e-02, -3.3379e-04,  3.8147e-05,  3.1662e-04, -2.6123e-02,  1.6016e-01,
             -6.6895e-02,  5.4932e-02, -1.5503e-02, -8.8379e-02, -2.4780e-02,  5.9204e-03,  7.0190e-03,  1.2390e-02,
             -1.2573e-02,  3.0518e-04,  8.1055e-02,  2.7832e-02, -5.8594e-02,  4.4823e-04, -5.3406e-03,  4.8096e-02,
             -3.6240e-04,  1.1414e-02,  7.1716e-04,  9.9182e-05,  1.2207e-02,  3.3789e-01,  6.5308e-03,  2.5000e-01,
              2.9373e-04,  1.4551e-01, -1.8359e-01,  4.6968e-05,  4.0245e-04, -1.5450e-04, -4.6143e-02, -1.9287e-02,
             -1.7853e-03,  4.9316e-02, -4.8340e-02,  4.7119e-02, -1.7578e-02, -2.1219e-05, -1.4038e-02,  2.2827e-02,
             -1.6479e-02, -1.7090e-01,  5.0537e-02, -9.2983e-06, -2.3804e-02,  4.4250e-04,  5.8899e-03,  2.0630e-02,
              6.2012e-02,  4.7852e-01,  2.5391e-02,  2.6245e-03,  4.6387e-02,  3.2471e-02,  4.9973e-04,  3.2031e-01,
             -1.7014e-03, -2.5879e-02, -9.9487e-03, -2.2125e-03, -2.7148e-01,  8.8692e-05,  5.7697e-05, -5.6839e-04,
              2.0752e-02, -2.6001e-02, -6.3705e-04,  3.0762e-02,  2.0123e-04,  3.1494e-02, -2.9297e-01,  1.7578e-02,
              8.4400e-05, -6.2399e-08, -9.2697e-04, -5.1856e-06,  4.2617e-06,  9.6436e-03,  8.2779e-04,  8.6060e-03,
             -7.3853e-03,  3.6719e-01, -1.1902e-03, -8.7280e-03, -8.3542e-04, -5.1880e-03,  2.1240e-02, -5.3406e-04,
              4.6875e-02, -1.7548e-03, -2.1362e-02, -2.1362e-04, -8.2031e-02, -5.0049e-02, -7.6675e-04, -3.2959e-02,
             -7.4387e-04,  2.3682e-02, -3.4668e-02,  1.5747e-02, -2.2339e-02, -8.3008e-02,  1.6479e-02, -8.9722e-03,
              1.1084e-01,  1.3965e-01, -1.4526e-02, -2.7588e-02, -1.1562e+00,  3.4809e-05, -1.8978e-04,  2.4048e-02,
             -1.1146e-05, -2.5195e-01, -2.1973e-03,  8.8882e-04, -2.4414e-03, -3.9864e-04, -2.1851e-02, -5.9509e-03,
             -1.2451e-01,  2.4902e-02,  5.8594e-02, -1.6406e-01,  2.8687e-02, -1.4877e-03, -1.1279e-01,  3.6926e-03,
              1.4210e-04, -2.6855e-02,  5.4016e-03, -7.3433e-05, -2.8801e-04, -6.0797e-05, -1.8921e-03, -4.8340e-02,
              9.7046e-03,  7.2937e-03,  5.8174e-05, -5.8594e-03,  1.1816e-01, -2.7954e-02, -5.4932e-02, -3.8147e-04,
              2.8931e-02,  8.7402e-02,  2.3438e-01, -3.1445e-01,  6.6406e-02,  6.2988e-02,  6.0303e-02,  2.0996e-02,
             -9.9182e-05,  2.4219e-01,  3.1738e-02, -2.9297e-02, -2.0386e-02,  3.1738e-02, -6.1035e-02, -2.0027e-04,
             -1.8433e-02, -4.7874e-04,  1.9775e-02,  4.6631e-02, -2.4796e-04,  7.5195e-02, -1.0681e-02,  3.0756e-05,
              2.0142e-02,  1.9165e-02, -1.5182e-03, -6.6833e-03,  4.7363e-02, -3.1738e-03,  8.9722e-03,  2.5749e-04,
             -5.3346e-06, -1.8501e-04, -5.0783e-05, -2.1240e-02, -9.8267e-03, -7.9590e-02, -4.1748e-02,  3.1494e-02,
             -4.0436e-04, -2.3804e-02,  3.3691e-02, -2.8931e-02,  9.0942e-03,  6.5430e-02,  7.2861e-04, -4.2114e-03,
              8.7738e-05,  2.1973e-02,  2.1973e-02,  2.4033e-04, -2.5269e-02, -9.7046e-03, -1.9653e-02, -3.0029e-02,
              1.6846e-02, -1.0300e-03, -9.9609e-02,  1.6861e-03, -1.8692e-03, -1.0147e-03, -1.4648e-03, -2.3556e-04,
             -9.9182e-04, -5.6839e-04, -6.2943e-05,  5.5542e-03,  4.4189e-02, -3.3855e-05, -1.3916e-02, -8.5831e-06,
              7.4463e-03, -3.8086e-02, -6.7383e-02, -7.4707e-02,  1.8555e-02, -1.7319e-03, -6.7444e-03,  6.5002e-03,
             -2.6398e-03, -1.7212e-02,  1.7548e-04, -1.1086e-05,  2.2911e-07, -1.4114e-04,  4.4405e-06, -6.8247e-06,
              1.6689e-04,  2.3071e-02,  1.4160e-01, -3.4668e-02, -7.9102e-02,  1.7738e-04,  4.7852e-02, -1.4496e-04,
             -1.9434e-01,  2.5177e-04,  8.6426e-02, -9.4414e-05,  4.6692e-03,  7.6172e-02, -2.7832e-02, -1.6689e-04,
              2.3174e-04,  6.0120e-03,  1.1328e-01,  3.6240e-04,  1.0605e-03,  4.4678e-02,  7.0801e-02,  1.1292e-03,
             -9.4727e-02,  3.2616e-04, -1.4587e-02,  5.6763e-03, -1.3733e-03,  8.1062e-05,  7.2266e-02,  1.6846e-02,
              8.2779e-04,  4.7607e-02,  3.7842e-02, -1.7334e-02,  1.7700e-02,  6.5430e-02,  4.1748e-02,  6.2866e-03,
             -3.1494e-02,  1.4355e-01,  2.3041e-03, -1.8616e-03,  4.6968e-05,  2.5392e-05,  4.1008e-04, -4.4189e-02,
              4.5654e-02,  3.0708e-04, -1.9775e-02, -2.2507e-04,  4.3945e-01,  4.6692e-03, -5.5664e-02,  7.6172e-02,
             -3.0994e-05, -2.8442e-02, -1.2817e-02,  1.0610e-05,  7.6904e-03, -1.0986e-03,  8.6426e-02, -4.3213e-02,
              7.3242e-03, -1.0742e-01, -8.1055e-02,  6.8247e-06,  4.2114e-03,  1.1444e-03, -1.4771e-02, -1.9653e-02,
             -6.2012e-02,  7.0801e-02,  3.6163e-03, -3.6621e-02,  2.3842e-04, -1.4404e-02,  2.4902e-02, -1.3351e-03,
              2.9449e-03,  1.5831e-04, -1.7524e-05, -7.1777e-02, -7.0801e-02,  6.0059e-02, -2.6123e-02,  3.3379e-05,
             -3.9482e-04, -6.0425e-03, -1.0315e-02,  7.1777e-02,  4.7607e-03, -2.3079e-04, -4.2480e-02,  1.1377e-01,
             -1.0547e-01,  1.6479e-02,  5.6839e-04, -2.6978e-02,  2.7924e-03, -8.2031e-02, -4.6692e-03, -2.9755e-04,
             -5.2643e-04, -4.0283e-02,  5.3101e-03, -4.8828e-04,  1.8921e-02, -2.3071e-02, -2.1582e-01, -4.6253e-05,
             -8.6670e-03,  3.3722e-03,  8.4229e-03,  3.7231e-03, -4.8828e-04,  1.5076e-02,  5.7983e-03, -4.9973e-04,
              2.7222e-02,  5.3883e-05,  1.2268e-02, -5.8105e-02, -3.6719e-01,  4.1504e-03, -7.5073e-03, -8.6914e-02,
              7.2754e-02, -1.0192e-05,  2.0996e-02, -3.9795e-02,  9.3384e-03, -3.7598e-02,  7.6771e-05,  2.2531e-05,
             -2.3937e-04,  7.3730e-02, -6.5430e-02, -4.3164e-01, -4.1199e-04, -4.8584e-02, -2.7161e-03, -8.3618e-03,
              2.6978e-02,  1.2268e-02,  1.4587e-02, -1.5137e-02,  4.1748e-02,  8.3008e-03,  1.6602e-01,  6.5308e-03,
             -8.0566e-03, -1.4258e-01, -1.4114e-04,  4.2969e-02, -5.2979e-02, -2.4109e-03,  2.4414e-03, -8.7738e-04,
              1.7047e-05, -1.4832e-02,  1.1978e-03, -7.4219e-02, -3.2471e-02, -3.2959e-02,  9.7046e-03,  2.7148e-01,
              5.9204e-03,  7.5073e-03,  1.4551e-01, -4.2114e-03,  5.5552e-05, -2.8442e-02, -1.8501e-04,  1.4465e-02,
             -4.8633e-01,  1.2589e-04,  4.0283e-02,  4.1992e-02, -5.4199e-02, -1.8845e-03, -8.2031e-02,  7.4863e-05,
              3.5400e-02, -3.3447e-02, -4.0436e-04, -3.3447e-02,  4.2236e-02,  2.5024e-03, -4.9438e-03,  7.2632e-03,
             -2.3556e-04,  5.9082e-02,  5.2261e-04,  3.9864e-04,  4.4189e-02,  3.6049e-04, -1.1215e-03, -1.9897e-02,
             -2.6894e-04,  9.4986e-04, -1.0742e-02, -1.0824e-04, -1.8594e+00,  4.2236e-02,  3.8086e-02, -1.1492e-04,
              1.1902e-02, -3.9368e-03, -9.3994e-03,  2.5879e-02, -2.6093e-03, -1.0620e-02, -1.0107e-01, -1.9989e-03,
              3.6133e-02,  2.7313e-03, -4.9561e-02,  4.8161e-05, -1.7524e-05,  3.5763e-06,  9.9182e-04,  1.8921e-02,
             -5.1270e-03,  1.0452e-03, -3.7003e-04,  4.3030e-03, -1.1902e-02, -7.8613e-02,  2.9564e-04,  6.4850e-04,
             -3.2654e-03,  1.9897e-02, -3.4571e-05,  1.6992e-01, -2.5183e-06, -1.6327e-03, -2.6703e-04,  1.1035e-01,
             -5.0537e-02, -1.5717e-03, -7.8583e-04, -8.0566e-02, -1.3199e-03, -6.5804e-05, -8.2779e-04, -4.6492e-05,
             -3.2043e-03, -4.5300e-06, -8.1543e-02, -8.4473e-02,  2.0218e-04, -1.9434e-01, -2.0752e-02, -4.0894e-03,
             -9.9659e-05, -3.5400e-03, -3.6621e-02, -9.4604e-03,  2.0752e-03, -5.3101e-03,  7.8735e-03,  3.2043e-04,
             -4.8256e-04,  1.5039e-01, -1.1169e-02,  2.4048e-02, -2.5757e-02, -9.0599e-05, -1.2402e-01, -6.6895e-02,
             -1.1328e-01,  8.4229e-03, -1.8799e-02, -1.0449e-01, -4.3154e-05,  5.6763e-03,  4.5166e-03, -1.0986e-02,
              9.5703e-02, -1.9336e-01,  8.1062e-05, -3.7079e-03,  4.5586e-04,  1.2109e+00, -2.3499e-03, -4.3335e-03,
             -2.8076e-02, -2.5391e-01,  2.0264e-02,  9.9659e-05, -1.0400e-01,  6.8359e-03, -6.1035e-02, -1.3199e-03,
             -4.6631e-02, -7.5378e-03,  1.4191e-03, -2.0752e-03,  6.1340e-03,  1.7853e-03,  3.1006e-02, -8.7738e-04,
             -5.0049e-03,  2.4536e-02,  1.0315e-02,  7.5684e-02,  8.5938e-02, -1.5640e-04, -2.2697e-04,  1.3256e-04,
              3.4332e-04,  1.1670e-01, -6.0730e-03, -2.8992e-03, -1.8433e-02,  7.9727e-04, -2.1118e-02, -1.1520e-03,
             -8.1055e-02, -3.4809e-05,  1.3046e-03, -4.5013e-04,  3.5156e-02, -9.0332e-02, -1.0254e-02, -2.5391e-02,
              2.7275e-04,  1.2684e-04, -1.5015e-02,  6.3477e-02,  2.5368e-04,  4.9438e-03,  1.5182e-03, -2.6245e-02,
              1.4941e-01,  8.7891e-03,  2.8076e-02,  7.2937e-03, -2.6550e-03,  1.7285e-05, -2.5868e-05, -2.4121e-01,
             -8.7738e-04,  2.5558e-04, -2.0508e-02, -4.8828e-02, -2.8038e-04,  7.6675e-04, -4.4434e-02, -1.1047e-02,
             -6.9885e-03, -1.1230e-02, -4.7874e-04,  8.9111e-03,  1.5723e-01,  1.1749e-03,  2.4902e-02,  2.7466e-04,
             -1.4746e-01, -9.9487e-03, -4.0283e-02,  5.6396e-02, -8.4473e-02, -9.6191e-02, -2.1729e-02, -4.5586e-04,
              3.3936e-02,  7.0572e-05,  4.1992e-02,  1.4160e-02,  2.9419e-02,  6.2943e-05, -1.0967e-04, -1.8750e-01,
             -4.5776e-04,  2.7344e-02, -1.1230e-02, -4.3701e-02, -3.6316e-03, -3.9062e-03, -1.4746e-01,  9.1553e-03,
              5.3644e-05, -3.9795e-02, -1.0742e-01, -7.3730e-02, -1.8539e-03,  7.6890e-06, -1.5411e-03,  3.8818e-02,
             -1.4782e-04, -8.9111e-03, -3.0640e-02, -1.0252e-05,  2.0386e-02,  3.6240e-04, -9.0003e-06, -3.4912e-02,
             -2.3193e-02,  1.7643e-04,  8.6784e-05,  5.1498e-04, -2.0386e-02, -1.9531e-02, -3.6133e-02,  1.1108e-02,
              4.0770e-05,  5.0068e-05,  2.0294e-03, -1.4648e-02, -2.5940e-03, -9.5825e-03,  1.8799e-02, -9.5749e-04,
             -1.5198e-02, -6.5327e-05,  1.4453e-01, -5.3787e-04,  1.0352e-01, -1.0109e-04, -1.2064e-04, -1.4551e-01,
              3.8574e-02,  9.9121e-02,  2.2697e-04, -8.6670e-03, -9.2163e-03, -1.0254e-02,  5.4199e-02, -1.2360e-03,
              5.5176e-02,  2.3926e-02,  1.4687e-04, -1.3256e-04, -3.1471e-05,  5.0659e-03,  5.2185e-03,  1.9824e-01,
              1.4893e-02,  6.7139e-04, -2.7161e-03, -4.2725e-03, -3.0899e-04, -2.7710e-02,  5.0781e-02, -7.0190e-03,
              2.7344e-02, -4.5410e-02,  2.1851e-02, -4.1504e-03, -9.5215e-02,  5.7129e-02, -9.9121e-02, -4.7607e-02,
             -6.7234e-05, -1.5869e-03, -1.9165e-02, -6.6757e-06,  3.4790e-03, -6.9824e-02,  2.8809e-02,  3.0708e-04,
              7.2479e-05,  1.2054e-03,  3.0212e-03,  2.1267e-04,  1.2024e-02, -1.7090e-03,  9.8633e-02, -1.0803e-02,
             -9.1195e-06, -5.7373e-02, -6.5002e-03, -7.0312e-02, -3.3936e-02, -3.6926e-03, -1.7090e-02,  1.0889e-01,
             -1.2779e-04, -1.9836e-03,  1.7700e-02,  9.6191e-02,  2.2656e-01, -2.0752e-02, -3.9368e-03,  1.2061e-01,
             -3.6001e-05,  1.5076e-02,  5.3406e-05,  1.2817e-02,  7.0190e-03, -3.2043e-03, -9.0003e-06, -5.6152e-02,
              1.0300e-03, -1.9312e-05, -2.5146e-02, -2.6321e-04,  2.3804e-02,  1.0071e-02, -6.2256e-02, -5.5176e-02,
              1.6251e-03, -6.0081e-05,  1.4973e-04,  3.3203e-02, -5.2261e-04, -3.2501e-03, -1.6357e-02, -6.9046e-04,
              1.0645e-01,  9.3842e-04, -7.0496e-03, -2.8419e-04, -6.9885e-03,  5.2185e-03,  9.2773e-03,  1.3123e-03,
             -1.4038e-02, -7.2002e-05, -2.9785e-02,  2.5879e-02, -8.9645e-05,  5.0306e-05,  1.8477e-05, -4.7852e-02,
              5.2734e-02,  3.8300e-03,  1.9836e-03,  4.2152e-04,  4.5395e-04, -2.5757e-02,  5.0735e-04, -3.5248e-03,
             -2.5146e-02, -1.0559e-02, -7.7209e-03, -2.1191e-01, -1.8883e-04,  4.6387e-03, -9.4604e-04, -1.6308e-04,
              7.1049e-05, -9.0942e-03,  8.8379e-02, -9.4727e-02, -1.4400e-04, -2.4414e-03,  9.2983e-05, -4.6631e-02,
              7.9155e-05,  1.1047e-02,  4.7363e-02,  8.1177e-03,  1.3672e-01,  2.5781e-01, -9.2163e-03, -1.9897e-02,
              6.2500e-02, -8.3008e-03,  3.3569e-03,  3.2349e-03,  3.8818e-02, -4.7493e-04, -4.5586e-04, -2.0386e-02,
              1.2512e-02, -1.6308e-04, -3.0670e-03,  8.3923e-05,  6.0547e-02,  1.0193e-02,  1.3657e-03, -3.1494e-02,
             -3.1982e-02, -6.7234e-05, -5.9570e-02,  2.1118e-02, -8.0566e-03,  2.8931e-02,  1.1902e-03, -9.3937e-05,
             -1.1475e-02,  4.3945e-02, -8.8215e-05,  6.3965e-02, -9.0599e-05,  8.4229e-03, -2.2411e-04,  9.0122e-05,
              1.7822e-02, -8.3008e-02, -9.3079e-04,  1.7090e-01, -1.3184e-02,  1.0071e-02,  1.1169e-02, -6.8848e-02,
             -3.4668e-02, -4.6015e-05,  6.1340e-03,  2.0996e-02,  2.3193e-02, -1.6235e-02, -1.5869e-03, -1.5198e-02,
              1.5564e-02,  3.2471e-02, -5.3711e-02, -4.2915e-05, -4.2534e-04,  3.4332e-04,  1.5259e-02, -1.0777e-04,
              1.6479e-03,  1.6632e-03, -2.1606e-02, -7.3624e-04,  6.0547e-02, -4.4441e-04,  3.7842e-02, -9.7275e-04,
              4.8096e-02, -7.9274e-06,  2.0264e-02, -2.8610e-04,  5.0659e-03, -7.8678e-05, -3.7109e-02,  8.5938e-01,
             -1.2283e-03,  1.1963e-02,  1.1368e-03, -3.4668e-02, -2.0142e-02,  3.9795e-02, -6.6833e-03,  1.5030e-03,
             -6.3705e-04, -3.7994e-03,  1.8082e-03,  5.1758e-02, -1.2146e-02, -3.3188e-04,  2.4170e-02, -7.0953e-04,
              1.2398e-04,  3.6469e-03, -3.3447e-02, -8.6914e-02, -3.1641e-01, -5.2490e-02, -3.7384e-03, -4.3457e-02,
             -1.2573e-02,  2.4536e-02,  2.5757e-02, -2.4780e-02,  3.9978e-03,  1.8652e-01, -9.7046e-03,  3.5763e-05,
             -8.6060e-03, -4.6349e-04, -7.3910e-05, -2.2984e-04, -1.5163e-04,  3.6478e-05,  1.9409e-02, -2.0447e-03,
              3.7384e-03, -6.4453e-02, -1.6953e+00,  3.1853e-04,  6.9824e-02, -3.2617e-01, -3.4637e-03,  2.7847e-04,
             -1.2302e-04,  3.7109e-02,  6.5613e-03, -3.7575e-04, -3.8330e-02,  7.4208e-06, -3.0029e-02, -4.4441e-04,
             -1.8539e-03, -8.7891e-02,  2.6758e-01,  1.7871e-01,  7.5340e-05, -1.8677e-02,  2.6758e-01, -3.0327e-04,
             -3.2227e-02, -5.6458e-04,  5.9509e-04, -4.0293e-05, -8.0078e-02, -2.5630e-06,  3.7891e-01, -3.3203e-02,
             -1.9193e-05, -6.8665e-05,  6.5308e-03, -1.5015e-02,  1.7700e-03,  4.2725e-03, -1.7188e-01,  2.1729e-02,
              1.8358e-05,  9.7168e-02,  2.1973e-02, -7.8201e-05, -1.4587e-02,  1.6113e-02,  4.1809e-03, -3.4766e-01,
             -1.9222e-06,  4.9561e-02, -5.5420e-02,  6.6833e-03, -1.3733e-03,  1.0864e-02, -4.7852e-02, -4.0527e-02,
              1.2436e-03,  1.4771e-02,  5.5420e-02,  3.8867e-01, -1.5625e-02,  1.1536e-02, -9.5215e-03, -1.0300e-04,
             -7.5684e-02, -1.2493e-04,  7.1106e-03, -4.5703e-01, -6.8848e-02, -4.8584e-02,  3.6621e-03, -1.1658e-02,
             -3.6865e-02, -5.2979e-02, -4.9210e-04,  5.1117e-04, -5.2734e-02,  2.3438e-02, -1.9165e-02,  3.8574e-02,
              2.9206e-05, -3.6316e-03, -2.5391e-01, -2.1553e-04,  4.8096e-02, -1.6113e-02,  1.5625e-02,  2.2461e-02,
             -5.4932e-04, -3.6774e-03, -2.4902e-02,  6.9580e-03,  5.6763e-03, -1.6602e-02, -2.0752e-02,  4.6387e-02,
             -1.9531e-02,  8.5068e-04,  3.1128e-02, -6.3171e-03,  6.1279e-02, -8.6670e-03, -3.1281e-04,  3.0398e-05,
              2.7771e-03, -1.8433e-02,  4.2439e-05, -1.4746e-01, -3.4570e-01, -1.0547e-01, -4.1992e-02, -1.4709e-02,
             -9.7275e-04, -1.0610e-05, -6.2500e-01,  4.8065e-04, -3.0136e-04,  1.7090e-02,  1.4420e-03,  2.2217e-02,
              8.2016e-04, -6.7139e-04, -3.3691e-02,  8.6784e-05,  2.8253e-05,  2.2125e-04,  6.0425e-03,  9.3384e-03,
             -4.4441e-04,  4.8096e-02, -4.0283e-03, -5.7983e-03, -4.4556e-03, -1.5640e-04, -6.4087e-04,  3.0899e-04,
             -2.2461e-02,  7.5684e-03, -1.0300e-03,  2.9541e-02,  4.0283e-02, -1.2598e-01, -2.7100e-02, -4.8218e-03,
             -2.0447e-03,  1.5259e-03, -1.4832e-02, -2.2217e-02, -1.1377e-01,  1.1426e-01,  1.1826e-03,  4.2343e-04,
             -1.3867e-01,  3.5889e-02, -3.5645e-02, -2.5391e-02,  9.8145e-02, -1.2500e-01,  7.0190e-03, -2.4414e-02,
             -8.1062e-05,  3.2959e-02, -4.7302e-04, -4.7607e-02,  3.2715e-02, -9.9121e-02, -1.0071e-02, -1.2207e-04,
              1.5381e-02, -5.7983e-04, -8.2031e-02,  3.5095e-03,  1.2598e-01,  3.4142e-04,  7.5817e-05,  1.2684e-04,
             -2.2583e-03,  4.5654e-02, -1.6846e-02,  3.2471e-02,  1.2390e-02, -2.5749e-04, -1.2146e-02,  2.2949e-02,
              5.1575e-03, -3.7384e-03,  6.1523e-02,  1.7334e-02,  1.1230e-02,  1.3046e-03,  1.4160e-01,  1.6975e-04,
              2.1484e-02, -3.8757e-03, -7.2327e-03, -1.8066e-01,  2.0885e-04, -3.7689e-03,  1.2695e-01, -1.1536e-02,
              9.5703e-02, -1.1597e-03,  2.9144e-03, -1.9775e-02, -4.1127e-06, -1.5450e-04, -5.2929e-05,  4.6730e-05,
             -5.9814e-02, -3.3379e-05,  1.9264e-04, -9.3384e-03,  2.3193e-02,  4.2419e-03,  1.4973e-04,  7.8735e-03,
              1.9897e-02,  2.2888e-03, -9.2285e-02,  4.3457e-02, -1.0071e-02, -3.6377e-02, -2.2125e-03,  2.5482e-03,
             -1.4160e-01,  4.8828e-02,  6.7139e-03, -4.6387e-03,  2.0313e-04, -4.8584e-02, -1.4221e-02, -6.9580e-03,
             -3.4668e-02, -4.9744e-03,  2.5635e-02,  4.1016e-02, -2.4414e-04,  3.5400e-02,  1.4725e-03,  1.4893e-02,
             -1.5469e+00, -9.8877e-03,  9.7046e-03, -5.9814e-03, -3.6240e-04,  1.3550e-02,  9.9945e-04,  1.0498e-01,
              3.3417e-03, -1.4404e-02,  4.7684e-05,  1.5354e-04,  4.5654e-02,  2.9419e-02,  8.9111e-03,  8.3496e-02,
              2.1057e-03,  1.6764e-06,  1.5991e-02,  1.3489e-02,  9.5749e-04,  7.1289e-02, -8.7891e-03,  1.0437e-02,
              1.0282e-06, -2.5513e-02, -3.2471e-02, -1.5015e-02,  1.2573e-02, -5.6152e-02,  8.3984e-02, -1.6357e-02,
              2.0752e-02, -3.3984e-01, -9.9487e-03, -7.1716e-04,  2.0905e-03, -1.3161e-04, -4.1992e-02,  3.1233e-05,
             -5.2643e-04,  7.5817e-05, -3.6377e-02,  1.2970e-03, -8.3542e-04,  5.8838e-02, -7.7148e-02, -1.1169e-02,
             -4.1199e-03, -3.6430e-04,  6.4453e-02, -8.8501e-03,  1.3672e-01,  4.1962e-04,  2.3560e-02, -2.9373e-04,
              1.1826e-03, -3.3951e-04,  1.9836e-04, -8.9722e-03, -6.9809e-04, -6.5002e-03,  1.8692e-04, -1.9455e-04,
              1.8120e-04,  2.0123e-04,  2.9175e-02, -7.3242e-03,  1.0498e-01,  1.9531e-02,  9.1553e-05, -6.9336e-02,
              6.6406e-02,  9.1797e-02, -3.3936e-02,  1.4893e-02, -5.2734e-02,  1.4648e-02,  8.9722e-03,  2.7924e-03,
              1.0300e-03, -9.8705e-05,  9.5844e-05,  1.9836e-03,  5.3787e-04, -7.6294e-03, -9.4604e-03, -5.1117e-04,
              7.7515e-03, -5.7220e-04,  2.2949e-02,  1.0547e-01,  3.3936e-02,  5.7129e-02, -1.3885e-03,  8.4229e-03,
             -4.9316e-02, -4.9438e-03,  2.7161e-03, -6.3737e-09,  3.1433e-03,  3.7402e-06, -6.1951e-03,  6.4941e-02,
             -1.7773e-01,  8.7738e-05,  4.9438e-03,  1.8359e-01,  1.0376e-02, -2.1057e-03,  9.2773e-03,  4.2969e-02,
              6.4453e-02, -1.0620e-02, -1.0132e-02,  1.0254e-02,  1.1414e-02,  2.4986e-04, -2.9541e-02,  2.9419e-02,
              1.0376e-03, -1.9989e-03,  3.2471e-02,  4.8218e-03,  5.7373e-03,  3.9864e-04,  6.6376e-04,  2.9663e-02,
              3.7842e-02,  4.3678e-04, -2.7832e-02, -1.5076e-02, -2.5635e-03,  6.4697e-03, -1.2817e-03,  9.3842e-04,
              7.5195e-02,  1.0920e-04,  4.4434e-02, -3.0518e-03,  5.7220e-05, -1.3733e-02, -4.7125e-07, -6.0059e-02,
             -1.4099e-02, -4.1809e-03, -1.0986e-02,  6.8665e-03, -4.6289e-01,  2.3633e-01, -2.3193e-02,  3.9368e-03,
              1.7944e-02, -3.4668e-02, -1.7471e-03,  2.0508e-02,  6.6833e-03, -4.1199e-03, -6.0654e-04,  4.9353e-05,
             -2.0142e-02,  1.2085e-02,  2.5879e-02, -2.1973e-02,  1.5918e-01, -1.2338e-05,  1.2158e-01, -1.4801e-03,
              6.4941e-02, -2.2278e-03, -1.1719e-02, -6.1523e-02,  4.9072e-02,  8.1787e-03,  7.2327e-03, -2.5513e-02,
             -1.3611e-02, -1.8188e-02, -5.9509e-03, -1.1841e-02,  1.3306e-02,  1.8597e-04, -1.1597e-03,  1.0303e-01,
              1.9836e-04,  2.5391e-02,  9.8419e-04, -1.5564e-03, -6.1035e-03, -3.6049e-04, -6.0791e-02,  2.1118e-02,
             -1.0490e-04,  5.7678e-03, -4.9210e-04, -3.2043e-04,  6.7139e-03, -8.5831e-04,  7.7248e-05, -1.5442e-02,
              5.1025e-02,  1.1426e-01, -4.6387e-02, -8.9111e-03, -2.1191e-01, -6.3965e-02, -1.6594e-04,  6.5918e-02,
              2.7313e-03,  2.8992e-03,  4.4346e-05, -1.1015e-04,  1.4404e-02,  1.1873e-04, -3.1433e-03,  1.6357e-02,
             -6.0791e-02,  2.4414e-03,  6.7444e-03, -3.9816e-05,  1.3504e-03, -2.7466e-02,  2.6855e-02, -3.0160e-05,
             -3.1982e-02, -4.0283e-03,  1.0559e-02, -3.9307e-02,  2.2461e-02,  7.8125e-03, -7.9346e-04, -5.5469e-01,
             -7.5912e-04, -3.3203e-02, -9.3079e-04,  5.1737e-05,  6.2256e-02, -7.8735e-03, -5.5176e-02,  0.0000e+00,
              3.7598e-02,  1.1108e-02,  4.2439e-05,  3.0029e-02, -2.0020e-02,  2.4414e-04,  3.8574e-02,  1.1015e-04,
              9.3079e-04, -2.6245e-02, -3.0100e-06, -2.0752e-02, -4.9114e-05,  7.2632e-03,  9.5825e-03, -4.2383e-01,
             -9.0820e-02,  2.7222e-02, -2.5879e-02,  2.3127e-05,  2.6001e-02, -5.1117e-04, -5.6885e-02, -1.2158e-01,
             -3.6621e-04, -6.4453e-01, -1.2695e-02,  2.2583e-03, -8.9722e-03, -1.8188e-02,  6.9824e-02, -3.3569e-04,
             -5.9128e-05, -7.9102e-02, -2.7344e-02,  9.1309e-02, -4.7112e-04,  1.5564e-02, -8.3923e-04, -1.0156e-01,
              4.7266e-01, -8.4686e-04,  7.9727e-04, -6.2256e-03, -7.9102e-02, -2.7832e-02, -1.5442e-02,  2.3804e-02,
             -1.0681e-02, -5.0354e-03,  3.1281e-04,  5.7068e-03, -9.8877e-03,  6.1035e-04,  2.0266e-05,  2.4401e-07,
              2.7084e-04,  1.6602e-02,  4.9316e-02, -1.0223e-03, -8.7357e-04, -8.5449e-03,  1.6327e-03, -1.4191e-03,
             -1.1108e-02,  4.7874e-04,  2.2461e-02,  1.1301e-04,  5.9814e-03,  1.5137e-02, -8.4229e-03, -1.9312e-05,
             -8.1055e-02,  3.0845e-06, -3.9673e-03,  2.4567e-03, -1.7700e-02,  1.7822e-02, -1.9409e-02,  8.3984e-02,
             -4.8584e-02, -4.8340e-02,  1.3769e-05,  9.7046e-03,  1.7090e-02, -8.9844e-02,  1.7578e-02,  6.8970e-03,
             -3.1494e-02,  1.7700e-02,  3.5667e-04,  2.7466e-03, -9.5825e-03,  1.3367e-02,  9.8228e-05,  4.4922e-02,
             -8.3496e-02, -3.0975e-03, -6.1417e-04, -1.0986e-03,  7.5817e-05, -6.4392e-03, -2.2705e-02, -2.6321e-04,
              1.2891e-01, -5.0537e-02,  1.3062e-02,  1.3065e-04, -8.0490e-04, -5.7617e-02, -4.9316e-02,  9.3262e-02,
              2.6367e-02,  1.8164e-01,  2.4780e-02,  1.1139e-03, -2.3633e-01, -4.2969e-02, -7.0190e-03, -1.1658e-02,
             -4.7119e-02, -3.2616e-04, -1.1523e-01,  1.9836e-03, -5.5176e-02,  7.1716e-03, -2.1973e-02,  4.0527e-02,
              4.2969e-02, -8.3984e-02,  6.3477e-03, -4.3869e-04, -2.2949e-02, -3.6914e-01,  1.2756e-02,  6.2500e-02,
              1.7776e-03,  1.3672e-02,  3.9062e-02, -3.0640e-02,  4.8340e-02, -8.0566e-03,  6.1646e-03,  4.9133e-03,
             -4.9591e-04,  1.6785e-04, -1.5354e-04, -2.4414e-01, -5.2490e-03, -2.9541e-02, -5.2002e-02, -3.6430e-04,
             -1.6235e-02,  4.1260e-02, -9.3384e-03, -2.5787e-03, -1.3933e-06,  5.4443e-02, -9.1309e-02, -4.2725e-03,
              5.9082e-02,  5.8860e-07, -1.4687e-04,  1.9409e-02,  3.1281e-04,  1.7166e-04,  2.6894e-04, -1.8066e-02,
              4.4189e-02,  7.7637e-02, -3.6865e-02,  1.7071e-04,  1.5991e-02, -6.0722e-07, -1.2100e-05,  5.4016e-03,
             -2.5177e-04,  7.2479e-04,  6.5994e-04,  3.1982e-02,  1.6479e-02, -1.1169e-02,  4.2725e-02, -6.6833e-03,
             -5.3711e-02, -3.2997e-04,  7.3730e-02,  8.0872e-04,  9.0332e-02,  6.1989e-05,  2.3730e-01,  4.6387e-02,
             -1.3855e-02, -5.5908e-02, -6.2988e-02,  3.3617e-05,  8.3923e-05, -5.0293e-02, -3.7575e-04,  3.9795e-02,
             -1.6861e-03,  4.1504e-02,  3.9307e-02, -1.5015e-02, -6.9336e-02, -7.7637e-02, -6.6895e-02, -1.5442e-02,
              1.6403e-04, -3.9795e-02,  1.8677e-02,  1.9897e-02,  7.9155e-05,  2.2430e-03,  3.2997e-04, -1.6846e-02,
             -2.8809e-02,  3.0518e-04,  5.0306e-05,  1.9775e-02,  1.7334e-02,  2.5781e-01, -5.9128e-04, -1.3351e-04,
              1.9836e-03,  2.1118e-02,  6.7871e-02,  2.7084e-04, -4.0527e-02,  1.0529e-03, -9.9609e-02, -1.3489e-02,
              1.2207e-02,  7.4158e-03, -1.8555e-02, -2.1515e-03, -2.0905e-03, -2.5757e-02,  1.1520e-03,  9.8267e-03,
             -7.5912e-04,  3.1494e-02, -1.8311e-04,  3.5547e-01, -4.0894e-03,  3.1281e-03, -1.3447e-04, -2.4986e-04,
              1.1536e-02, -1.4465e-02,  1.9897e-02, -5.5237e-03,  7.3730e-02, -4.3640e-03,  1.5039e-01,  1.2390e-02,
              1.6403e-03,  3.3691e-02,  4.7112e-04, -4.0770e-05,  1.4771e-02, -6.0791e-02,  1.4126e-05,  4.7852e-02,
             -1.2573e-02, -2.7776e-05,  7.0572e-04, -7.2266e-02, -2.5940e-04, -2.5513e-02, -5.4016e-03,  2.5024e-02,
             -6.6528e-03,  1.6479e-02, -8.4877e-05,  4.5538e-05, -2.8198e-02, -7.4463e-03,  4.5776e-03, -1.6846e-02,
             -4.5776e-04, -1.7822e-02,  1.9932e-04,  1.5015e-02, -6.3705e-04,  1.1572e-01, -3.6377e-02,  4.8876e-05,
              1.4282e-02,  3.0029e-02, -2.6123e-02, -1.1873e-04,  2.1729e-02, -2.7657e-04, -8.3008e-03, -4.1260e-02,
              1.3574e-01, -5.6885e-02,  1.3351e-04, -2.9755e-03, -5.9232e-07, -1.0864e-02,  7.3853e-03, -2.3499e-03,
             -3.0884e-02,  2.4292e-02,  6.4453e-02, -2.5635e-02, -1.6602e-01,  7.2754e-02, -7.3242e-02, -2.2827e-02,
             -3.3379e-05, -4.5776e-03, -1.1133e-01,  1.2207e-03,  9.1553e-03, -1.4877e-03,  8.0078e-02,  4.3945e-02,
             -2.1729e-02, -3.5858e-03, -1.8921e-02, -3.2715e-02,  1.2589e-04,  6.4373e-06, -2.6703e-04,  3.8147e-03,
             -3.8910e-03, -2.8076e-03, -2.9785e-02,  2.0599e-03,  2.8992e-04,  6.6895e-02, -3.2043e-03, -3.6865e-02,
             -1.3638e-04, -6.0081e-05, -1.5747e-02, -5.2490e-03, -1.7456e-02, -4.3640e-03, -1.9897e-02,  7.3612e-06,
             -7.3433e-05, -4.6158e-04, -4.6387e-02, -5.4443e-02, -2.1515e-03,  7.2861e-04,  1.2939e-02,  1.1536e-02,
             -1.6504e-01, -7.8964e-04, -5.3711e-02, -4.4632e-04,  2.6758e-01,  3.9816e-05,  4.2725e-02, -5.6458e-03,
              1.3733e-02, -1.6846e-02, -1.8433e-02,  5.3467e-02, -1.1169e-02,  1.4355e-01, -1.0559e-02, -6.1989e-05,
             -2.2411e-04, -1.1902e-03, -5.5664e-02, -2.1729e-02,  5.7373e-02,  1.0620e-02,  8.2397e-03, -2.9419e-02,
             -4.7119e-02,  8.1062e-05,  2.4170e-02, -1.0742e-02, -1.2756e-02,  1.6968e-02, -1.1084e-01,  1.1816e-01,
             -3.3875e-03, -8.4043e-06, -4.4107e-05, -5.7068e-03,  4.7607e-03,  5.2490e-03,  4.6484e-01, -3.1281e-03,
              2.7588e-02,  5.6458e-04, -4.5395e-04,  8.1253e-04,  6.8848e-02,  1.1572e-01, -9.6321e-05, -4.2480e-02,
             -1.4343e-03, -3.1281e-04, -6.1768e-02, -1.8835e-05, -1.6451e-05, -4.4250e-03, -7.4219e-02, -1.3574e-01,
              4.2841e-07, -4.2480e-02, -3.0708e-04,  9.3750e-02, -2.9144e-03,  1.1730e-04, -3.5400e-03, -8.7738e-04,
              3.0518e-02,  2.8809e-02, -2.0020e-02, -1.5625e-02, -1.8358e-05, -1.0834e-03,  6.2561e-03, -1.9922e+00,
              5.7861e-02, -2.0142e-03, -1.6022e-04, -1.5723e-01,  4.8218e-03,  5.6396e-02, -2.1484e-02,  1.1444e-03,
              2.7588e-02,  3.1250e-02, -9.2983e-05,  1.1444e-04,  2.0703e-01,  2.6123e-02,  2.0801e-01,  1.3828e-05,
              4.4678e-02,  7.9155e-05, -3.5400e-02,  1.6602e-02,  8.3984e-02, -1.2493e-04,  2.5391e-02, -2.8198e-02,
              9.1797e-02,  7.2266e-02,  1.1084e-01, -5.1880e-04, -3.5763e-05, -4.9133e-03,  9.7275e-04, -3.6377e-02,
              1.2512e-03, -1.9073e-03,  4.1389e-04, -1.4591e-04, -1.7456e-02,  1.6602e-02,  1.4453e-01,  1.5747e-02,
              3.9291e-04,  1.7773e-01, -7.4158e-03,  6.3705e-04, -5.3406e-03, -1.0693e-01,  2.0027e-04, -2.8419e-04,
              3.9978e-03,  1.4160e-02, -2.9297e-02,  2.3804e-03,  9.5215e-03,  6.2500e-02, -4.8065e-04, -2.8198e-02,
             -3.7766e-04, -1.4725e-03, -2.3145e-01, -5.5237e-03,  2.5940e-03, -2.2095e-02,  6.9809e-04,  8.3008e-02,
             -1.2012e-01,  1.2817e-02, -8.2397e-04,  1.4893e-02, -1.7822e-02, -1.3984e+00,  7.3730e-02,  8.3008e-03,
             -7.7820e-03,  4.8340e-02,  2.8320e-01,  2.5177e-04, -2.7100e-02,  1.0498e-02, -1.0107e-01, -2.4609e-01,
             -3.9062e-03, -2.7466e-04, -5.3906e-01, -6.2012e-02,  3.8338e-04,  3.4943e-03,  4.8523e-03, -1.7834e-04,
             -1.9897e-02,  3.8605e-03, -3.9368e-03,  3.2425e-04,  7.9590e-02,  8.8501e-04,  3.9291e-04,  3.8528e-04,
              4.9805e-02,  5.1856e-06,  1.4687e-04, -1.3275e-03, -2.1815e-05, -2.3804e-03,  2.0385e-05,  3.9673e-03,
              7.9590e-02, -1.2665e-03,  7.8613e-02,  6.0547e-02,  2.8610e-04,  2.3804e-02,  7.0801e-02,  2.4902e-02,
              6.0547e-02,  7.5684e-03,  6.9885e-03,  9.9182e-04, -3.7575e-04, -9.2163e-03, -9.5215e-03, -1.8311e-02,
             -1.8883e-04, -5.2490e-02, -7.0496e-03,  9.7046e-03, -2.7832e-02, -1.3924e-04, -2.4170e-02,  3.6719e-01,
             -1.4343e-02, -7.2937e-03, -2.9663e-02, -1.5442e-02,  1.0605e-03, -1.8799e-02,  2.4536e-02, -8.9355e-02,
              7.2754e-02,  3.4668e-02,  4.1504e-02,  5.3167e-05,  3.5286e-04, -1.5869e-02,  2.7299e-05,  1.9922e-01,
             -1.1742e-05,  5.4688e-02, -1.4526e-02, -3.9307e-02, -1.9264e-04, -5.1514e-02, -4.9133e-03, -1.1621e-01,
              1.2390e-02,  7.2956e-05,  3.2715e-02,  1.9226e-03, -2.6245e-02,  7.7148e-02,  6.0547e-02, -3.5706e-03,
              9.2983e-05,  2.5757e-02, -3.4180e-03,  3.7384e-04, -1.4591e-04,  2.0313e-04, -1.2493e-04,  3.6133e-02,
             -7.7148e-02, -1.2894e-03, -1.0498e-02,  1.9141e-01,  1.1230e-01, -1.4038e-03, -5.0366e-06,  6.6895e-02,
             -1.2493e-04, -1.7700e-02,  7.8613e-02, -8.0566e-03,  5.6458e-03,  1.3367e-02, -4.7207e-05, -2.4033e-04,
              8.3984e-02,  2.0020e-02,  1.2279e-05, -5.3406e-04, -1.9775e-02,  8.4839e-03,  6.9141e-05,  2.8687e-02,
             -5.4169e-04, -1.1139e-03,  6.8665e-04, -2.8442e-02,  4.5471e-03, -3.3379e-05, -3.6865e-02,  1.2451e-02,
              6.1951e-03,  8.2016e-05, -1.1230e-01,  1.1265e-05, -7.2327e-03, -1.3184e-01,  5.4443e-02, -1.7071e-04,
             -2.0752e-02,  1.1292e-02, -3.6163e-03, -2.5146e-02,  2.5269e-02,  7.6890e-06,  1.2573e-02,  6.0120e-03,
              1.1749e-03, -5.7617e-02, -1.9897e-02, -7.5073e-03,  8.6975e-04, -1.7334e-02, -1.0681e-02,  2.0504e-04,
             -8.4229e-03, -6.0425e-03, -2.5879e-02,  3.4424e-02,  9.3994e-03,  3.0994e-05,  1.1353e-02, -6.5002e-03,
              1.8799e-02, -6.1035e-05, -5.7983e-03,  1.3086e-01, -4.0436e-04, -4.2725e-02,  1.1353e-02, -4.9438e-03,
              8.6799e-07, -8.3618e-03, -6.1035e-03, -7.0496e-03,  6.6833e-03,  4.0039e-02,  1.0559e-02, -1.1047e-02,
              3.5156e-02, -5.1880e-04, -2.1515e-03,  3.7193e-04,  4.5898e-02, -1.9287e-02, -4.6082e-03,  1.1215e-03,
             -1.1523e-01,  2.2217e-02, -4.9133e-03,  1.3550e-02,  1.2268e-02, -1.1108e-02, -3.4523e-04,  1.2875e-04,
             -2.3315e-02,  6.1095e-06, -3.1471e-04, -5.7602e-04,  2.9951e-06,  3.2715e-02,  1.0538e-04, -5.1270e-03,
             -4.3457e-02, -7.4506e-07, -9.0332e-03, -2.1815e-05, -7.3433e-05, -1.8066e-02, -9.8267e-03,  3.1494e-02,
             -9.8228e-05, -4.0054e-04,  1.2875e-04,  3.2227e-02,  1.0620e-02,  6.2561e-04,  1.8311e-02,  5.5695e-04,
              1.6212e-04, -1.2793e-01, -1.3611e-02, -1.3367e-02,  2.5940e-03,  7.4005e-04,  1.9531e-02,  3.7354e-02,
             -1.2878e-02,  9.7752e-05, -1.2207e-02,  3.3569e-03, -1.6968e-02, -3.6133e-02,  8.7402e-02,  1.5163e-04,
              2.4796e-05,  1.9238e-01, -3.4027e-03,  1.8001e-05, -1.7166e-03,  2.9175e-02,  1.1414e-02,  4.5410e-02,
             -9.1553e-03, -6.3477e-02, -2.0313e-04, -1.8677e-02,  3.3379e-05,  2.4719e-03,  1.1902e-02, -2.7418e-05,
             -1.8799e-02, -2.0898e-01, -2.6978e-02,  1.3965e-01,  1.4038e-02, -5.9814e-03,  1.5076e-02, -6.7383e-02,
             -6.5430e-02,  1.7944e-02,  1.1902e-02, -8.1787e-03, -1.6113e-02, -7.9956e-03,  3.5095e-04, -2.7895e-05,
              9.9945e-04, -8.4961e-02,  8.6594e-04, -2.5330e-03,  3.8086e-02,  5.2452e-05,  2.5977e-01,  2.0385e-05,
              1.3828e-04, -1.7944e-02,  2.4109e-03,  1.0400e-01, -1.7773e-01,  2.0996e-02, -9.2163e-03, -2.4048e-02,
              2.7100e-02,  4.7607e-02,  9.5215e-03, -1.1475e-02,  8.5831e-04, -7.7209e-03, -7.8583e-04,  7.4387e-04,
             -6.7871e-02,  3.5889e-02,  1.3828e-04,  6.6757e-05,  3.7842e-02,  1.6308e-04,  1.3184e-02, -3.8910e-04,
             -5.1117e-04, -6.1417e-04, -2.0020e-02, -1.8406e-04,  3.8719e-04, -2.0599e-03,  1.8188e-02,  5.2643e-04,
             -9.1553e-03,  1.3245e-02,  3.9978e-03,  2.6550e-03,  5.5237e-03, -9.1309e-02,  7.2266e-02, -2.8442e-02,
             -1.1780e-02,  9.7275e-04, -2.5177e-03,  5.8365e-04,  2.6822e-05,  1.7578e-02,  2.0752e-02,  1.1841e-02,
              1.8005e-03,  2.9053e-02, -6.8359e-03, -2.1729e-02,  9.8267e-03,  5.2246e-02, -1.0864e-02,  8.6784e-05,
             -8.7738e-05, -3.0640e-02,  1.1328e-01, -1.2158e-01,  7.2327e-03,  9.9609e-01, -3.5547e-01, -2.0264e-02,
             -4.8584e-02,  5.2261e-04, -6.4697e-03,  7.2479e-05,  2.6131e-04, -4.4678e-02, -1.0071e-02,  8.3618e-03,
             -8.0078e-02,  1.6403e-04,  2.6512e-04, -3.7500e-01, -1.0610e-05, -1.2329e-02,  7.8964e-04, -3.9062e-02,
              8.9111e-03, -9.3937e-05,  8.4839e-03, -1.5640e-03,  1.0376e-02,  1.1475e-02,  6.9580e-03, -3.1250e-02,
             -4.1580e-04, -1.1902e-03,  9.6436e-03, -4.3488e-04, -3.7253e-06, -3.9673e-03, -5.6076e-04, -4.9805e-02,
              6.8054e-03,  1.0729e-04, -6.0081e-05,  1.3672e-01,  5.4169e-04, -4.5586e-04,  1.7090e-02, -2.6978e-02,
             -4.1748e-02,  2.6367e-02, -1.3809e-03,  6.0059e-02, -5.1737e-05, -3.3875e-03,  5.1117e-04,  1.9287e-02,
              2.8564e-02, -4.2725e-02, -4.0531e-05, -2.3926e-02, -5.8289e-03,  7.5378e-03,  5.1498e-05, -9.2506e-05,
              6.4087e-04,  1.3916e-02,  3.0899e-04, -2.2705e-02,  6.5002e-03,  4.1016e-01,  4.9072e-02, -2.7344e-02,
             -1.1146e-05, -2.4902e-02, -4.0894e-03,  3.4912e-02, -3.8330e-02,  4.2725e-04, -4.8096e-02, -2.4658e-02,
             -6.6528e-03,  1.1328e-01,  4.7302e-03, -3.2812e-01, -3.2654e-03, -1.6602e-01, -3.0899e-04, -1.9908e-05,
             -2.0386e-02, -2.3438e-02, -2.7588e-02, -1.4648e-03, -1.0376e-03, -7.5378e-03,  6.0654e-04, -6.1768e-02,
              2.6855e-02,  1.0693e-01, -7.2754e-02,  2.0630e-02, -5.7861e-02,  1.8215e-04,  5.9326e-02, -1.1826e-04,
             -2.3193e-02, -2.2888e-04, -1.1230e-01,  3.7994e-03,  3.7891e-01,  3.0899e-04, -2.8687e-02, -7.8583e-04,
             -1.1230e-01, -5.0537e-02,  6.7749e-03,  3.7384e-03, -1.3367e-02, -2.1240e-02,  1.0254e-02, -9.8419e-04,
              2.7618e-03,  1.1597e-02,  1.0204e-04,  1.1015e-04,  2.0695e-04, -2.0630e-02,  1.3885e-03,  3.1006e-02,
             -2.0123e-04, -4.2969e-02, -1.1683e-04,  2.7466e-04,  1.4305e-04, -4.7302e-03,  4.6631e-02,  5.4932e-03,
              1.4893e-02, -1.2305e-01,  1.8005e-03, -4.1016e-02, -1.9836e-03, -1.3977e-02, -1.5918e-01, -3.7193e-05,
              1.8311e-02, -1.4591e-04,  4.2343e-04,  1.3542e-04,  1.0391e+00,  1.1780e-02, -5.0659e-03, -6.5918e-03,
              8.4473e-02, -1.9287e-02, -2.3499e-03,  3.2196e-03,  5.0354e-04,  8.6426e-02, -5.9814e-02, -2.2430e-03,
              4.3958e-07, -4.0527e-02, -1.4282e-02, -2.6321e-04,  4.1389e-04,  4.1809e-03,  2.0790e-04, -6.4468e-04,
              1.4404e-02,  4.0527e-02,  2.2559e-01, -5.7861e-02, -1.6785e-04, -7.5684e-02,  9.8877e-03,  5.3711e-03,
             -3.4332e-03, -9.5703e-02, -1.9836e-03,  2.1553e-04,  1.5199e-05, -6.1417e-04,  8.0566e-03, -9.0408e-04,
             -2.0294e-03, -6.7902e-04, -1.4267e-03,  1.2207e-02, -6.9336e-02,  5.1498e-04,  4.8633e-01,  3.7003e-04,
              2.6172e-01, -1.0889e-01, -6.1035e-02,  1.6602e-01,  2.7710e-02, -1.2817e-02,  2.0504e-04, -9.7275e-04,
             -3.0060e-03,  4.5204e-04, -1.2517e-05, -2.9182e-04, -4.1016e-02, -9.2030e-05, -4.3106e-04, -3.1494e-02,
              2.0752e-03,  3.3379e-04,  2.5635e-02,  1.0986e-02,  1.3977e-02,  1.2589e-03,  9.1553e-03,  1.2085e-02,
             -4.9973e-04, -3.9291e-04,  8.5449e-02,  5.6152e-03,  5.7518e-06, -1.3161e-04,  4.9114e-05,  1.2512e-02,
             -1.2793e-01, -1.1426e-01,  8.6060e-03,  4.6730e-04,  5.1575e-03, -4.1748e-02, -4.0894e-03,  4.5776e-03,
              3.9978e-03, -2.3499e-03, -1.0777e-04,  1.0791e-01, -1.9455e-03, -3.8330e-02,  2.3079e-04,  6.7383e-02,
              2.0386e-02,  6.4941e-02,  4.7363e-02,  5.7373e-02, -9.0599e-05, -9.7168e-02, -8.3008e-02, -2.5781e-01,
             -1.9897e-02,  6.5430e-02,  3.0151e-02, -1.6022e-04, -2.3956e-03, -1.5259e-02, -1.4400e-04,  6.8359e-03,
              2.4319e-04, -5.3223e-02,  2.6489e-02,  7.1716e-04, -6.4850e-05, -4.2114e-03,  4.1992e-02, -2.8687e-03,
              2.7954e-02,  2.4986e-04, -4.4250e-03, -2.6489e-02, -1.6724e-02,  2.4140e-06,  6.9336e-02,  1.1265e-05,
              1.0529e-03, -1.8311e-02,  5.6152e-03, -1.6235e-02, -2.1851e-02, -1.9169e-04,  3.1090e-04, -2.8711e-01,
             -1.2891e-01, -4.1580e-04,  3.4332e-05,  3.8910e-04,  3.0365e-03, -3.4180e-01, -2.6123e-02,  1.5182e-03,
             -7.5989e-03, -2.1362e-04, -1.5869e-02,  1.3123e-02, -6.1523e-02, -1.4267e-03,  2.1973e-02,  1.9165e-02,
              1.9897e-02,  2.3926e-02, -4.3701e-02, -1.8311e-02,  3.7109e-02, -4.1504e-03, -9.0122e-05, -8.0078e-02,
             -6.8359e-02, -1.7548e-03, -4.4861e-03, -4.5013e-04,  5.9509e-03,  1.6785e-03, -2.0874e-02,  6.9885e-03,
              1.4019e-04,  1.6724e-02, -1.8555e-02,  2.1839e-04,  2.3365e-04, -6.0272e-04,  2.3926e-02, -4.2419e-03,
              1.4038e-02, -7.4219e-02,  1.5717e-03,  2.8076e-02, -8.6426e-02,  3.6377e-02, -3.2187e-05, -2.1118e-02,
             -1.1139e-03,  2.8320e-02,  8.3008e-03, -5.7602e-04,  4.2725e-03,  7.5073e-03,  2.1973e-02, -2.3926e-02,
             -4.4922e-02, -4.8340e-02,  2.2339e-02,  1.3733e-02,  6.0425e-03,  2.6123e-02, -6.1646e-03, -4.6730e-04,
              3.0469e+00,  3.3188e-04, -4.3945e-02,  7.9956e-03, -1.3965e-01, -9.1553e-03,  4.6492e-05, -8.8379e-02,
             -2.2461e-02,  6.7871e-02,  3.8818e-02, -7.5684e-03,  7.0572e-05,  6.2180e-04,  2.9541e-02, -2.4605e-04,
              2.5177e-03,  9.8877e-03,  1.1035e-01, -2.1851e-02,  7.4463e-03, -3.1738e-02, -4.9561e-02, -6.5918e-02,
              3.1471e-05,  4.0527e-02, -3.7537e-03, -1.8997e-03,  1.8597e-04,  7.3547e-03, -7.5684e-02, -2.3633e-01,
              5.7129e-02, -5.3101e-03, -1.4941e-01, -5.6458e-03,  7.3242e-03, -3.5095e-04, -9.3384e-03,  2.2095e-02,
              8.7280e-03, -2.7275e-04,  2.7100e-02, -4.0771e-02,  6.5613e-03, -1.7776e-03, -4.7119e-02,  2.8076e-03,
             -2.4170e-02,  2.5146e-02,  1.4746e-01, -3.2501e-03, -4.3106e-04,  1.7944e-02, -6.4468e-04,  4.7445e-05,
              2.0752e-02,  1.4551e-01,  1.0596e-01,  3.6865e-02,  4.2969e-02, -2.3804e-02,  6.2256e-03,  1.6602e-02,
             -1.9302e-03,  2.4223e-04, -4.4434e-02,  4.3945e-03, -3.6774e-03,  1.0059e-01,  3.5524e-05, -6.4453e-02,
             -4.5898e-02,  6.5430e-02, -5.2691e-05, -9.9121e-02, -4.4861e-03, -4.0054e-05,  6.6757e-05, -1.9836e-03,
              1.3000e-02, -2.3438e-01,  2.3145e-01, -7.6294e-05, -2.3651e-04, -1.1230e-02, -1.2512e-03,  1.0605e-03,
             -2.2827e-02, -9.2773e-03, -8.1635e-04, -6.0425e-03,  2.7710e-02, -4.2969e-02, -4.6015e-05,  2.6611e-02,
              2.4605e-04,  4.3335e-03,  1.0490e-04,  2.3438e+00, -2.6512e-04, -1.6724e-02, -1.1475e-02, -9.9487e-03,
              1.0073e-05,  1.3046e-03, -7.3547e-03,  1.8978e-04, -1.6479e-03,  4.7607e-03,  1.6479e-02,  5.3787e-04,
              5.2979e-02,  8.0078e-02, -2.2583e-03,  1.2329e-02,  1.7944e-02,  3.2959e-02,  0.0000e+00, -3.5524e-05,
             -2.3438e-02, -7.6172e-02, -1.8188e-02, -8.0078e-02,  4.0527e-02,  2.6512e-04,  4.5703e-01,  1.3542e-04,
              2.5630e-05,  1.9550e-04,  1.4099e-02, -2.5391e-02,  6.1279e-02, -3.7354e-02,  1.5545e-04,  3.0518e-02,
              5.4389e-07, -3.5248e-03,  3.8818e-02, -1.2988e-01, -9.2030e-05, -6.1951e-03, -1.9455e-03,  1.6484e-07,
              2.0508e-02,  3.5763e-06, -4.7852e-02,  1.8463e-03, -5.6152e-03, -9.5749e-04,  1.0610e-05, -5.9082e-02,
              1.1301e-04, -1.2573e-02,  9.1553e-03,  2.8076e-03, -5.7983e-04, -2.5024e-03,  3.6240e-04,  1.6602e-01,
              1.2512e-02,  5.7459e-05,  2.4707e-01, -9.5215e-03, -3.7766e-04,  2.1744e-04, -9.9487e-03, -1.0132e-02,
              2.3438e-02,  1.1597e-02, -2.1729e-02, -2.6489e-02, -3.0273e-02, -1.1015e-04,  3.2654e-03,  2.3460e-04,
              3.0212e-03,  2.4986e-04,  4.6692e-03,  6.2561e-03, -1.5430e-01, -4.6692e-03,  3.8330e-02,  3.0899e-04,
             -1.8463e-03, -5.4836e-05, -8.6784e-05, -2.5177e-04, -1.4648e-03,  1.3855e-02,  8.3008e-03, -2.6584e-05,
             -1.0559e-02, -4.8584e-02, -1.3924e-04,  7.7724e-05, -1.0986e-02, -7.0190e-03, -5.1025e-02, -2.0027e-05,
              1.9336e-01, -1.6880e-04,  6.0120e-03,  5.1575e-03,  2.6562e-01,  3.3264e-03, -4.5898e-02, -3.3760e-04,
             -8.1177e-03, -5.0049e-03,  1.3199e-03,  1.2589e-04, -1.4496e-04,  3.1738e-02,  5.4016e-03,  0.0000e+00,
             -2.4902e-02,  1.0824e-04,  2.9419e-02,  7.7637e-02, -1.3062e-02,  8.9264e-04, -6.5430e-02,  2.0020e-02,
              1.5015e-02,  6.0120e-03, -1.2988e-01,  4.8447e-04,  4.7363e-02, -5.2002e-02,  1.7188e-01, -5.5420e-02,
             -1.3733e-04, -4.4823e-05, -2.3956e-03,  4.7607e-03, -2.2602e-04,  3.7842e-02, -2.7222e-02, -9.6875e-01,
             -2.8711e-01,  6.8970e-03, -3.0273e-02,  3.9101e-04, -2.1820e-03,  3.1433e-03, -5.3787e-04,  1.0986e-02,
              7.3314e-06,  2.5787e-03, -3.7537e-03,  1.7456e-02, -1.2302e-04, -1.1035e-01,  1.0431e-05, -7.3547e-03,
             -9.8877e-03, -1.4099e-02, -1.2756e-02,  1.3794e-02,  5.8289e-03,  3.0640e-02, -2.9663e-02,  7.4005e-04,
              9.1553e-03, -9.4604e-03,  5.7697e-05,  9.7752e-06, -2.8931e-02, -1.3086e-01,  1.1816e-01,  1.1623e-05,
             -2.2217e-02,  1.4343e-02, -2.9297e-02,  3.2234e-04, -8.2016e-05,  8.4229e-03,  1.8835e-05,  3.8086e-02,
             -2.6489e-02,  6.1035e-03, -3.2715e-02, -3.6316e-03,  1.5991e-02,  1.1292e-03, -6.0547e-02, -2.0874e-02,
             -1.5259e-02,  1.3000e-02, -2.5749e-04, -4.5204e-04, -1.3885e-03, -3.2997e-04,  8.2397e-03, -5.3406e-03,
             -3.5286e-04,  1.2817e-02,  2.2339e-02,  4.1199e-03,  2.9449e-03,  2.5635e-02, -7.6294e-03, -5.3406e-04,
             -1.1635e-04, -3.6133e-02,  1.5625e-02,  1.1328e-01,  1.0538e-04,  4.9438e-03,  2.4567e-03,  2.3145e-01,
             -1.9836e-04, -2.8038e-04, -1.3379e-01,  2.4902e-02, -2.9297e-02, -5.4199e-02,  3.3203e-02,  2.1851e-02,
              1.0107e-01,  7.8125e-02,  8.2031e-02, -5.5552e-05,  5.1880e-03,  7.7820e-03, -2.0752e-02,  1.0347e-04,
             -1.2878e-02, -4.1389e-04,  3.4790e-03,  6.3477e-03,  5.9082e-02, -2.3315e-02,  1.0742e-02, -5.1270e-03,
              4.1016e-02,  2.0409e-04,  1.3672e-02,  3.3447e-02, -1.7262e-04, -7.0312e-02,  5.5176e-02,  6.4392e-03,
              1.2207e-02, -6.8970e-03,  3.2959e-02,  1.1444e-04,  1.0059e-01,  4.6730e-05, -2.8442e-02, -1.4771e-02,
             -2.4902e-02, -6.2561e-03,  1.0010e-02,  9.5825e-03, -2.5368e-04, -1.1182e-01,  2.3071e-02,  1.8692e-04,
             -2.7222e-02,  4.4861e-03, -1.0437e-02,  5.4550e-04, -1.4114e-04, -1.7242e-03, -1.4221e-02,  2.1484e-02,
             -1.4114e-04, -1.4496e-04, -1.8066e-02,  1.4603e-06, -1.3649e-05, -3.7109e-02, -1.4400e-04,  1.3123e-02,
              4.9353e-05, -2.4319e-04, -1.3306e-02,  2.0294e-03,  6.3965e-02,  2.8931e-02, -1.3542e-04, -2.6550e-03,
             -1.7262e-04,  2.6512e-04,  2.6733e-02, -2.3193e-02, -9.0820e-02,  8.8120e-04, -9.8267e-03,  9.1553e-03,
              3.6719e-01, -1.4832e-02,  1.2894e-03,  6.8359e-03,  3.8338e-04,  3.1250e-02,  1.2024e-02, -2.4605e-04,
             -1.7456e-02, -1.0376e-02,  1.1444e-03,  3.3447e-02,  7.7515e-03, -4.3945e-03,  1.8024e-04, -2.6398e-03,
              2.6123e-02, -1.3411e-05, -1.0252e-04, -2.6512e-04,  4.5410e-02,  4.5166e-02,  8.0566e-03, -5.8651e-05,
              6.6280e-05,  3.0670e-03,  1.7969e-01,  9.2773e-02,  3.5667e-04,  6.2943e-05,  1.0010e-01,  7.3547e-03,
             -4.5967e-04,  1.1139e-03, -8.6914e-02, -6.8848e-02,  4.3213e-02, -6.4453e-02,  1.9531e-01,  2.9373e-04,
             -4.9316e-02, -5.4169e-04, -7.5195e-02, -2.6093e-03,  5.8174e-05, -2.4170e-02, -1.5926e-04, -5.3406e-04,
             -5.8594e-03, -2.2221e-04,  2.1362e-02,  3.0327e-04,  4.0771e-02, -3.8086e-02, -6.6833e-03, -1.6357e-02,
             -2.6855e-03,  2.0294e-03,  1.1292e-02, -2.8320e-02,  3.1586e-03, -6.4453e-02,  4.2534e-04,  1.7524e-05,
              3.0518e-02,  3.6621e-04, -3.1738e-02,  1.9073e-04, -2.4658e-02, -1.9165e-02, -9.2316e-04, -6.1798e-04,
              6.0938e-01, -2.1362e-02,  2.7588e-02, -2.0625e+00,  4.2969e-02, -6.6833e-03,  5.0537e-02,  3.6240e-05,
              2.2363e-01, -9.8633e-02,  4.4922e-02, -4.6253e-05, -4.3457e-02, -1.3199e-03,  1.1230e-02,  1.8311e-02,
             -4.5538e-05, -1.7776e-03, -9.2773e-03, -3.4142e-04, -7.2002e-05, -3.2234e-04,  1.2695e-02,  2.2827e-02,
             -2.1118e-02, -3.3203e-02, -4.4189e-02,  2.0752e-02, -2.5749e-04,  1.8188e-02,  2.0695e-04, -1.2131e-03,
             -3.3617e-05, -5.9814e-03, -4.2725e-03, -1.6708e-03,  5.4321e-03, -4.4346e-05,  1.6479e-02, -7.1049e-05,
              6.3281e-01,  1.3245e-02,  8.8501e-04,  9.6436e-03, -4.5898e-02, -1.1921e-04, -2.6398e-03, -7.6294e-03,
             -3.6359e-06,  4.6082e-03,  6.4392e-03,  9.5825e-03,  1.1475e-01,  8.5449e-03, -6.3705e-04,  6.7383e-02,
             -1.9455e-04,  6.2287e-06, -4.2725e-04,  6.5234e-01,  1.8120e-05, -7.2098e-04, -1.9653e-02, -9.3384e-03,
             -2.4080e-05,  5.1270e-03,  1.4038e-02,  6.0120e-03, -1.4160e-01,  2.6131e-04,  3.5858e-03, -1.0205e-01,
              1.7643e-05, -1.1158e-04,  2.6367e-01, -8.0566e-02,  1.0742e-02, -1.6708e-03,  4.3701e-02, -1.0437e-02,
             -7.5195e-02,  9.6512e-04,  5.7373e-02, -9.1076e-05,  8.0566e-02, -8.0490e-04, -2.0385e-05,  1.1658e-02,
              2.6123e-02, -2.2583e-03,  8.2397e-03,  3.7994e-03,  1.1086e-05,  2.5146e-02, -6.1646e-03, -1.5747e-02,
              3.2227e-02, -2.2125e-04, -2.2095e-02, -1.5488e-03, -1.5234e-01,  2.8992e-04,  1.9434e-01,  2.0801e-01,
              3.2959e-03,  2.4512e-01,  3.4714e-04, -1.6699e-01,  1.0824e-04, -5.4688e-02,  3.5553e-03,  3.7598e-02,
             -3.0823e-03, -2.0752e-03,  1.5869e-02, -4.0054e-04,  4.4107e-05,  6.9336e-02, -3.7598e-02,  9.2285e-02,
             -3.1662e-04, -1.0681e-02, -5.0735e-04, -2.5269e-02, -7.7148e-02,  7.9346e-03, -9.5215e-02, -1.5831e-04,
              2.9325e-05,  2.0508e-02,  2.4414e-02, -1.7166e-03, -1.9379e-03,  2.6894e-04, -3.6469e-03, -2.3071e-02,
             -3.6163e-03,  2.6855e-02,  1.1353e-02, -6.2212e-07, -5.5313e-04, -5.8350e-02,  2.1905e-06, -2.0142e-03,
             -5.3101e-03, -1.1523e-01,  3.1250e-01,  5.8289e-03,  1.1597e-02, -2.5868e-05,  9.7752e-06, -1.8164e-01,
              7.2754e-02, -2.5024e-03,  7.3242e-03,  1.2329e-02, -6.1512e-05,  1.9043e-02,  6.6757e-04,  1.5991e-02,
              2.5368e-04, -2.6978e-02, -1.1902e-02, -6.0654e-04,  1.6357e-02,  1.1063e-03,  5.5313e-04, -2.0146e-05,
              8.5831e-05, -6.5002e-03, -3.5400e-02, -7.9632e-05,  1.5015e-02, -1.2398e-04,  1.8945e-01, -1.3504e-03,
             -5.8174e-05,  2.6953e-01, -1.9287e-02,  5.7220e-05,  1.3367e-02, -3.0029e-02,  1.3351e-04,  5.2261e-04,
              4.3701e-02, -1.9897e-02, -2.9419e-02, -2.8687e-02,  5.3516e-01, -2.5781e-01, -2.8320e-02, -1.3379e-01,
             -2.2363e-01,  7.2632e-03,  3.8757e-03, -1.8921e-02,  6.2943e-05,  2.8076e-02, -1.8406e-04,  4.1016e-02,
             -9.0332e-03, -3.1662e-04,  1.3351e-05,  5.2643e-04, -2.5868e-05,  2.5879e-02, -4.6143e-02, -2.6489e-02,
              9.9487e-03, -7.3242e-02, -6.0499e-06,  4.8218e-03,  4.4727e-01,  1.1536e-02,  3.4809e-05, -2.2583e-02,
             -2.0117e-01, -8.7738e-05, -3.1250e-02,  2.5000e-01,  2.5272e-05, -1.0824e-04, -2.3365e-04, -2.3633e-01,
             -4.1199e-03,  3.0518e-05, -1.9372e-06, -4.0436e-04,  5.1514e-02,  4.2677e-05,  1.2207e-02,  1.3855e-02,
             -3.4180e-02, -1.2436e-03, -6.4453e-02, -1.0986e-02, -1.0559e-02,  1.4844e-01,  1.0223e-03, -3.5645e-02,
              7.3624e-04, -1.5106e-03, -5.3711e-02, -1.7929e-03, -1.3916e-02,  1.1063e-03,  4.0771e-02,  2.8461e-06,
              6.3705e-04,  1.2934e-05, -2.1387e-01, -2.7466e-02, -4.4632e-04, -2.0218e-04,  6.8359e-02,  4.6387e-02,
             -3.2471e-02,  5.9509e-03, -6.1798e-04, -1.3855e-02,  8.6594e-04, -2.0508e-02,  1.5527e-01,  5.6396e-02,
              7.8613e-02,  4.3213e-02, -2.0266e-06,  2.9297e-02,  1.5411e-03,  1.8215e-04, -3.2043e-03, -4.5471e-03],
            [-6.7383e-02,  1.2390e-02, -1.5564e-02,  2.7008e-03,  9.5825e-03,  1.2012e-01, -9.1553e-05, -1.3977e-02,
             -1.8311e-02, -2.0386e-02, -1.3184e-02,  1.0620e-02, -1.1520e-03,  8.9645e-04, -1.3733e-03,  1.4099e-02,
              1.0010e-02,  3.4912e-02, -1.1673e-03,  1.9409e-02, -8.8867e-02,  1.2573e-02,  3.2043e-04,  2.0874e-02,
              6.3477e-03, -1.5137e-02, -3.4714e-04, -7.0572e-05, -6.2988e-02, -3.1662e-04,  8.9645e-04, -1.1133e-01,
              1.1780e-02,  1.5747e-02,  2.9297e-03, -1.7090e-02,  1.2695e-02, -1.9409e-02, -4.4922e-02, -5.0659e-03,
              7.2098e-04,  6.3896e-05, -3.7384e-03, -3.9673e-04,  4.4861e-03, -5.0781e-01, -2.1362e-03,  4.2114e-03,
             -4.3154e-05,  5.9570e-02,  1.5717e-03, -1.4210e-04, -4.2725e-03,  1.0757e-03, -1.2875e-04,  3.9795e-02,
              1.5163e-04, -8.8501e-03,  3.4523e-04,  1.2939e-02,  2.5177e-04, -6.6757e-06,  1.4465e-02, -5.4321e-03,
              2.8320e-02,  1.4019e-04,  4.7607e-02,  5.9570e-02,  4.2236e-02,  9.3132e-07,  6.4850e-05,  2.1362e-04,
             -1.7578e-02,  1.3809e-03, -3.2425e-05, -1.8311e-04,  1.8616e-03,  7.7148e-02, -5.1270e-02, -5.7373e-03,
             -2.3484e-05,  4.4434e-02, -1.3672e-02, -1.8555e-02,  5.4550e-04,  9.6798e-05, -2.1667e-03, -8.4229e-03,
              1.4526e-02, -7.5989e-03, -3.2349e-03,  1.3855e-02,  3.4912e-02,  1.2793e-01,  6.4392e-03,  2.9785e-02,
             -2.8809e-02, -6.9580e-03,  5.9814e-02,  3.5095e-03, -1.4954e-02, -1.8945e-01, -2.0874e-02, -6.1523e-02,
             -3.2654e-03, -6.0425e-03, -2.2705e-02, -6.4941e-02, -1.8311e-02, -7.1777e-02, -1.1139e-03,  9.6798e-05,
              7.7209e-03,  1.5564e-03, -9.8267e-03, -2.4261e-03,  1.5076e-02, -3.3691e-02, -1.0681e-03,  3.2227e-02,
              9.2773e-03, -3.1982e-02, -3.0670e-03,  7.5195e-02, -7.0801e-03,  8.6975e-04, -1.2756e-02,  7.4463e-03,
              1.9165e-02,  6.6895e-02, -2.3315e-02, -1.4420e-03,  7.8678e-06,  6.0059e-02,  2.2754e-01, -8.4229e-03,
              3.8300e-03, -6.1035e-04, -1.1978e-03, -4.5061e-05, -7.6771e-05, -1.8188e-02, -1.4901e-05, -2.1076e-04,
              3.0640e-02, -3.6163e-03,  9.7656e-03, -1.7822e-02, -1.0681e-04, -1.6689e-04,  3.1250e-02,  1.3262e-06,
              1.6880e-04,  6.1417e-04, -1.2500e-01, -4.6015e-05,  8.8501e-03,  1.0498e-02,  3.0708e-04,  2.4780e-02,
             -7.5989e-03,  2.2461e-02, -1.7929e-04, -2.6131e-04,  2.2217e-02, -5.9082e-02,  4.6387e-03,  1.8066e-02,
             -1.3733e-03, -5.9082e-02,  2.6855e-02, -6.8843e-06,  4.4434e-02,  3.4485e-03, -1.8555e-02,  1.0010e-02,
              5.6152e-02, -1.9287e-02,  3.6621e-02, -3.9062e-03,  4.6082e-03, -3.3379e-05,  3.8330e-02,  1.6357e-02,
             -1.8555e-02, -6.6376e-04, -5.8594e-02, -6.0059e-02,  4.0588e-03,  7.8580e-09, -6.6895e-02,  6.4392e-03,
             -1.6174e-03, -2.1240e-02, -1.1230e-02,  4.9438e-03,  2.4780e-02, -3.5248e-03,  3.0884e-02, -3.4375e-01,
             -2.5988e-05, -3.2196e-03, -6.1393e-06, -1.9287e-02, -1.1841e-02,  6.0120e-03,  2.6001e-02, -1.5974e-05,
              2.1820e-03, -4.2969e-02, -3.9368e-03,  6.3181e-06, -1.3065e-04, -8.8692e-05,  4.0039e-02, -1.2283e-03,
              2.9373e-04, -2.8038e-04, -1.0925e-02, -2.8610e-04, -2.6894e-04,  6.0120e-03, -1.2684e-04, -1.2064e-04,
              2.3041e-03,  2.5749e-04,  2.8320e-02, -1.8555e-02,  3.6812e-04,  9.4604e-03, -4.8828e-03,  4.6997e-03,
              1.5991e-02, -7.8735e-03, -8.6060e-03,  1.3489e-02,  1.8311e-02,  1.0147e-03, -2.3438e-02,  9.0599e-05,
             -1.7578e-02, -1.3916e-02,  1.5068e-04,  2.1191e-01,  7.4158e-03, -5.4932e-04, -1.1635e-04,  9.1076e-05,
             -8.1543e-02, -7.4463e-03, -1.3428e-03,  6.9046e-04, -3.4180e-02,  1.9836e-03,  8.1055e-02, -4.0039e-02,
              5.4932e-03, -1.9455e-04,  3.9577e-05, -3.0975e-03,  3.3447e-02, -1.2512e-03, -2.1240e-02,  1.5564e-03,
              8.2031e-02,  6.2561e-04,  2.5391e-02, -2.9449e-03, -1.2064e-04,  6.6833e-03, -1.1719e-02,  1.4782e-04,
             -6.2988e-02,  3.4027e-03,  3.8818e-02,  6.8359e-01, -6.1523e-02, -5.7373e-02, -3.0975e-03, -4.6997e-03,
             -1.5076e-02, -1.6797e-01,  6.0425e-03,  1.3794e-02, -1.2741e-03, -6.1768e-02, -3.1090e-04,  3.6865e-02,
             -8.0109e-04, -6.7444e-03, -1.9922e-01,  1.3672e-02, -5.2734e-02, -2.5630e-05,  8.2520e-02,  5.0049e-02,
             -4.1250e+00, -3.7598e-02, -6.2256e-03, -4.0527e-02,  2.3723e-05, -7.0801e-02, -1.9653e-02, -6.4087e-04,
              1.0681e-03, -5.3167e-05,  4.8637e-05, -3.5400e-02, -1.2085e-02,  5.0964e-03, -6.3965e-02,  1.5234e-01,
             -1.0071e-02, -3.5858e-03,  4.2480e-02, -4.3640e-03, -1.0529e-03, -1.3794e-02,  1.6403e-04,  1.4038e-02,
             -1.6895e-01, -1.6022e-03,  1.5625e-02,  1.7700e-02,  5.7373e-02,  7.0801e-03,  1.3550e-02,  5.0049e-03,
             -3.6011e-03, -1.0596e-01,  3.8528e-04,  6.5430e-02, -3.0994e-05, -2.2095e-02,  3.5547e-01,  2.7344e-02,
             -8.6060e-03,  1.8921e-02,  1.1780e-02, -1.1963e-02, -3.3203e-02,  4.1748e-02, -3.1494e-02,  1.9932e-04,
             -9.4727e-02,  1.8311e-02,  3.4668e-02, -6.5994e-04, -9.5825e-03,  8.3984e-02,  1.6937e-03,  1.3611e-02,
             -1.4648e-02,  6.0272e-04,  1.7643e-04,  9.4986e-04, -7.7148e-02,  2.1362e-04, -2.3682e-02,  1.2970e-04,
             -7.3242e-02, -2.6978e-02, -4.6997e-03,  5.7373e-02,  1.3256e-04,  4.7302e-03, -1.1230e-01, -1.2817e-02,
              1.1730e-04,  7.4863e-05,  8.0566e-03,  2.3926e-02,  1.7456e-02,  2.6733e-02, -2.7008e-03, -4.8218e-03,
              9.7046e-03,  7.5150e-04,  1.7357e-04, -2.6855e-02, -1.2207e-04,  2.7771e-03, -6.4453e-02, -8.3447e-05,
             -3.3722e-03,  3.6621e-04,  1.4771e-02,  2.6123e-02, -1.2634e-02, -4.1485e-05, -2.2736e-03, -5.7373e-03,
              1.6499e-04, -2.8442e-02, -1.2695e-01,  9.3994e-03,  2.1875e-01, -3.4332e-04, -2.0752e-02, -4.0283e-02,
             -1.9531e-02, -5.7220e-04, -1.0803e-02, -2.4986e-04, -2.4805e-01, -5.4443e-02, -2.2461e-01,  2.2221e-04,
              6.8665e-03,  1.7090e-02, -6.7383e-02, -1.9775e-02, -1.0803e-02,  1.7929e-04,  6.2180e-04, -1.0986e-02,
             -4.8637e-05,  1.0132e-02, -3.9551e-02,  2.2736e-03, -7.2266e-02,  3.6377e-02,  7.8735e-03, -2.5177e-03,
             -4.3106e-04,  1.7944e-02, -1.2756e-02,  5.5237e-03, -6.1512e-05, -5.4169e-04, -8.6060e-03, -4.1809e-03,
              3.1738e-02, -3.4668e-02,  4.1809e-03,  5.0659e-03, -4.3869e-05, -6.5430e-02,  2.0742e-05,  1.2109e-01,
              2.1729e-02, -1.2970e-03, -7.6599e-03,  1.1444e-04, -1.2146e-02,  7.2098e-04,  9.3384e-03, -1.0010e-02,
             -1.2207e-03, -5.4598e-05,  2.0752e-03, -1.0645e-01,  3.0365e-03,  7.4707e-02, -5.4169e-04,  1.7881e-06,
             -4.3701e-02,  3.5763e-05, -5.1880e-03,  1.0443e-04,  9.1797e-02,  2.8687e-02,  1.9043e-02,  2.8992e-03,
              5.1880e-03,  8.5068e-04, -4.8447e-04,  2.5635e-02,  1.2061e-01, -1.2024e-02, -4.1016e-02,  3.1494e-02,
              4.6692e-03,  1.1873e-04, -7.2266e-02,  4.0436e-04, -1.1475e-02, -1.3574e-01,  8.9645e-04, -6.4453e-02,
              6.7902e-04,  5.5847e-03,  1.1826e-04, -2.3346e-03, -3.2959e-02, -8.9844e-02,  3.3379e-04, -1.3245e-02,
             -1.5442e-02,  2.5391e-02, -2.2583e-03,  1.3086e-01, -9.7656e-03,  3.0518e-02,  2.1118e-02, -5.1514e-02,
              1.0840e-01,  2.5024e-03, -4.3945e-02, -2.2852e-01,  5.4199e-02,  6.7520e-04, -4.2236e-02,  2.4885e-06,
             -5.6152e-03, -6.8359e-03, -6.0059e-02, -4.8828e-02, -3.2616e-04,  2.5391e-02, -3.4027e-03,  2.1057e-03,
             -2.8076e-02,  2.7344e-02, -6.4373e-05,  2.5586e-01, -1.8188e-02,  5.2490e-03,  6.7871e-02, -3.2997e-04,
             -1.8677e-02,  3.4180e-02,  3.8910e-03, -4.3945e-03, -1.2112e-04, -3.6621e-03, -2.9419e-02,  4.8096e-02,
              1.1426e-01, -1.6708e-03,  2.6584e-05, -7.6294e-03, -4.5166e-03,  3.3691e-02,  3.0029e-02, -3.2349e-03,
              8.5354e-05,  1.8921e-02,  2.6001e-02, -5.1270e-02, -5.1025e-02, -1.2146e-02, -2.7895e-05,  4.3701e-02,
             -9.7656e-03, -1.7456e-02,  5.2452e-06, -4.9210e-04,  4.8637e-05, -2.2583e-02, -2.7344e-02, -1.6724e-02,
             -9.3842e-04,  4.2725e-02,  6.5002e-03, -1.3447e-04, -1.4687e-04,  1.4832e-02,  1.5442e-02,  3.3855e-05,
              4.4434e-02, -8.5449e-02, -4.8828e-04,  8.2031e-02,  9.5825e-03, -8.1177e-03, -2.9663e-02, -3.3760e-04,
              2.6703e-04,  1.1826e-03,  1.9409e-02, -1.6093e-05,  3.4523e-04, -6.5002e-03,  6.1035e-04,  1.8677e-02,
              6.3965e-02,  3.1128e-03,  3.7842e-03,  1.4725e-03, -5.4169e-04,  2.4609e-01, -1.6212e-05,  2.1729e-02,
              2.0874e-02, -1.4160e-02,  1.1841e-02, -3.7079e-03, -8.5831e-04,  2.3270e-04, -1.2875e-04,  4.5776e-04,
              1.9165e-02, -2.5034e-05,  9.3994e-03,  2.7344e-02, -1.8501e-04,  3.4180e-02,  2.3346e-03, -1.8359e-01,
             -6.8359e-03, -1.0681e-04,  1.9550e-05, -2.4292e-02, -1.1597e-02,  1.7212e-02, -5.4169e-04,  9.7275e-04,
             -1.4844e-01, -3.0762e-02, -3.8385e-05,  5.9766e-01,  4.1016e-02,  5.6396e-02, -9.0790e-04, -2.5635e-02,
              2.9907e-03,  7.2632e-03, -2.0142e-02, -5.1514e-02, -7.7820e-04,  5.7373e-03, -2.1011e-06,  3.1853e-04,
             -5.3406e-03, -1.2695e-02, -2.0020e-02,  4.3945e-03, -6.5002e-03, -1.6113e-02,  1.6022e-03,  4.4250e-03,
             -1.4424e-05, -8.0566e-03,  2.8564e-02, -6.5918e-02, -9.2387e-06,  1.8954e-05, -7.2632e-03,  4.7607e-03,
             -5.0537e-02,  2.9683e-05,  6.7902e-04, -4.8340e-02,  3.4424e-02, -2.0874e-02,  9.7656e-04, -3.8300e-03,
              1.4572e-03, -1.0400e-01, -1.0431e-05, -3.2425e-04, -4.6968e-05, -2.0020e-02, -7.5073e-03, -4.8637e-04,
              6.2988e-02, -1.1063e-03, -6.7444e-03, -2.1606e-02, -8.5449e-03,  8.5831e-06, -4.8828e-03,  5.5908e-02,
              1.9409e-02,  1.5430e-01,  5.1575e-03,  4.3678e-04,  4.4250e-04, -9.1076e-05,  8.5449e-04,  1.9550e-04,
              1.8616e-03, -1.9653e-02,  1.8311e-04, -4.2915e-04, -3.1982e-02,  6.5918e-03,  9.1553e-04,  4.0820e-01,
              2.6978e-02,  2.8076e-02,  1.3672e-02,  1.4465e-02,  5.4932e-03,  1.8066e-02, -2.1118e-02, -4.9133e-03,
              1.8787e-04,  5.4199e-02, -4.6730e-04,  8.2520e-02,  1.4343e-03,  6.9336e-02,  5.2185e-03, -2.2411e-04,
             -1.3046e-03, -5.5075e-05,  3.3691e-02, -2.0264e-02,  5.0735e-04, -9.8419e-04,  2.2705e-02,  1.5488e-03,
             -4.8584e-02, -5.1022e-05, -1.0986e-03,  1.6479e-03,  9.3079e-04, -1.4210e-04,  1.4893e-02, -9.2773e-03,
             -1.4343e-03, -3.0518e-02,  7.2479e-05, -1.5820e-01, -4.7982e-06, -1.9141e-01,  1.6968e-02, -1.0757e-03,
              1.3965e-01,  7.6294e-03,  2.9907e-02, -3.7842e-02,  2.2278e-03,  2.7466e-02,  1.2512e-03, -1.6846e-02,
              3.6133e-01,  8.9355e-02, -3.1471e-05,  5.5075e-05,  2.5368e-04,  1.4591e-04,  1.6797e-01, -8.6670e-03,
             -4.6631e-02,  3.9551e-02,  2.0752e-02,  8.0566e-02,  2.0409e-04, -2.9419e-02, -3.7994e-03, -3.2959e-02,
              3.2654e-03,  2.1362e-04,  1.3000e-02, -2.4438e-05,  2.8992e-03,  1.5411e-03, -2.5177e-04,  3.6316e-03,
              5.2261e-04,  5.1880e-03, -6.5994e-04, -4.2969e-02,  9.1934e-04,  1.0315e-02,  7.0312e-02, -7.7209e-03,
             -5.1737e-05,  1.4725e-03, -1.5106e-03, -3.9307e-02, -8.0109e-04, -1.4709e-02, -5.4443e-02, -2.6733e-02,
              1.0376e-02, -2.0630e-02, -8.3008e-02,  7.1289e-02,  2.3926e-02,  4.6631e-02, -3.6716e-05,  1.2988e-01,
              1.0071e-02,  1.8555e-02,  2.9144e-03,  3.8086e-02,  1.5450e-04, -5.5075e-05, -4.8584e-02, -1.0107e-01,
              3.0518e-04,  1.0303e-01, -3.2959e-02,  3.2959e-03,  1.1086e-05, -2.4292e-02,  1.8555e-02, -6.7139e-03,
             -2.7084e-04, -1.2964e-06,  3.9637e-06,  6.2943e-05, -2.8610e-05,  1.0443e-04, -5.0049e-03,  2.8534e-03,
              7.0572e-05,  2.8687e-02, -1.2573e-02,  8.1635e-04, -1.9741e-04,  4.8399e-05, -4.8096e-02,  2.2070e-01,
             -3.5156e-02, -1.7212e-02, -6.3782e-03, -4.5654e-02, -1.0840e-01, -6.1951e-03, -9.3994e-03,  2.4261e-03,
              9.6436e-03,  1.9670e-05, -1.7871e-01,  5.6641e-02, -7.3242e-02, -5.3406e-04,  2.2705e-02,  6.4087e-03,
             -2.4319e-04,  1.4465e-02,  8.3542e-04,  4.2439e-05,  1.2756e-02,  6.7871e-02,  3.0884e-02, -2.6978e-02,
              4.4346e-05,  5.6152e-03,  3.9062e-01,  3.2187e-05, -2.0218e-04,  2.8229e-04,  3.0212e-03, -1.5991e-02,
             -1.3794e-02,  7.2266e-02,  1.0059e-01,  3.3203e-02, -1.2207e-02, -1.4484e-05, -1.2024e-02,  2.8320e-02,
             -1.8433e-02, -2.2949e-02, -4.6921e-04, -1.6332e-05,  5.4932e-02, -3.7956e-04, -2.2736e-03, -2.9175e-02,
              5.5664e-02,  2.9492e-01, -7.2479e-04, -1.5503e-02, -3.9062e-02,  3.9795e-02, -3.1281e-04,  1.9165e-02,
             -1.3916e-02, -3.8330e-02, -1.8677e-02, -4.4861e-03,  1.2512e-02, -4.1127e-06,  1.5199e-05,  8.4686e-04,
              2.2461e-02,  3.4668e-02,  7.9155e-05, -9.7168e-02,  3.0327e-04,  5.5176e-02, -1.7578e-01,  3.0396e-02,
              6.2466e-05, -3.1143e-06, -1.0620e-02, -1.6689e-04,  2.2173e-05,  9.4727e-02, -4.1962e-04, -2.8687e-03,
              7.6294e-03,  2.8711e-01,  1.0834e-03, -1.4893e-02, -5.4626e-03, -1.5747e-02,  4.6997e-03,  3.5706e-03,
              5.3223e-02, -3.4714e-04, -5.1758e-02, -2.7466e-04, -2.4170e-02, -3.2227e-02, -3.1853e-04,  1.7700e-02,
              3.1433e-03,  4.7363e-02,  3.1738e-02,  1.4771e-02, -1.2573e-02, -1.1035e-01,  3.2471e-02,  2.0630e-02,
              2.1362e-02,  3.1494e-02, -4.0771e-02, -1.1536e-02, -8.2422e-01, -3.5763e-05,  1.1063e-03,  7.4219e-02,
              6.0499e-06,  6.7871e-02,  4.1016e-02,  9.7656e-04, -8.3008e-03,  2.2984e-04,  9.8267e-03,  1.0498e-02,
             -1.9409e-02, -3.2227e-02, -8.1055e-02,  4.8096e-02,  8.1055e-02, -2.9907e-03, -2.5635e-02,  5.0659e-03,
              7.2956e-05, -1.3306e-02, -2.4780e-02, -3.5667e-04, -4.4107e-05, -4.3488e-04, -4.2236e-02, -5.7617e-02,
             -2.2278e-03,  2.8442e-02,  3.2043e-04,  4.2480e-02,  5.1758e-02, -4.6631e-02,  1.2512e-02, -3.5858e-04,
              2.2949e-02,  3.4668e-02,  2.7832e-02, -1.1108e-02,  1.7700e-02,  1.4404e-02,  2.7222e-02,  3.6774e-03,
             -3.5524e-05, -1.9727e-01,  1.0205e-01,  2.4902e-02,  1.9165e-02, -2.2461e-02,  2.7710e-02, -3.6240e-04,
             -3.7842e-03,  9.4604e-04, -6.1035e-02,  2.2125e-03, -1.8787e-04,  8.3008e-02, -1.0376e-02,  3.8743e-06,
              2.9297e-02,  1.1063e-03,  1.1921e-04, -1.6602e-02,  6.3965e-02, -3.2196e-03,  6.7139e-03,  1.9360e-04,
             -3.2902e-05, -7.7248e-05, -2.1267e-04, -3.5400e-02, -1.4160e-02, -6.6406e-02,  7.8613e-02,  1.1279e-01,
             -4.5204e-04, -6.0547e-02, -3.9062e-02, -1.3916e-02,  1.8188e-02,  3.0884e-02, -2.5391e-02, -1.8677e-02,
              4.1246e-05,  6.2866e-03,  7.5378e-03,  2.6123e-02,  3.3722e-03, -8.9722e-03, -1.5259e-02, -2.3438e-02,
              2.8076e-03, -1.9989e-03,  1.9043e-02, -2.0447e-03,  1.3504e-03, -4.8065e-04, -3.8300e-03,  4.6730e-04,
             -2.3804e-03, -5.2261e-04, -2.3395e-06, -1.1749e-03,  6.2180e-04, -9.1791e-06, -2.7466e-02,  3.7956e-04,
              2.0447e-03, -4.9744e-03, -7.1716e-03, -1.7090e-01,  1.5015e-02, -7.4158e-03,  6.8970e-03,  3.1738e-02,
              2.5330e-03, -2.5787e-03,  5.1880e-04,  3.9339e-05, -3.9637e-06,  5.7220e-04, -1.0133e-05,  9.1195e-06,
              1.2970e-04,  1.1328e-01,  6.1768e-02, -3.9062e-02, -1.4404e-02,  6.7234e-05,  4.5166e-02,  1.9932e-04,
              4.0039e-01,  7.5340e-05,  3.3691e-02,  1.3447e-04,  9.2773e-03, -1.6724e-02, -5.2002e-02, -2.6345e-05,
              5.5313e-04, -8.8501e-03, -7.8735e-03,  1.6689e-05,  1.1292e-03,  3.3447e-02,  5.3467e-02,  9.6893e-04,
             -4.9561e-02, -2.3842e-04, -4.0283e-03,  9.0942e-03,  2.4567e-03,  1.5974e-05,  1.2054e-03,  2.7100e-02,
             -3.6049e-04,  2.0264e-02, -6.2256e-02,  2.1057e-03,  3.2227e-02, -3.1494e-02,  8.6060e-03, -2.4414e-04,
             -4.4434e-02,  9.0332e-02, -1.1597e-03,  3.7575e-04,  1.9455e-04, -5.7697e-05,  1.4782e-04, -9.9487e-03,
             -5.3955e-02,  2.8729e-05, -2.3438e-02, -4.8447e-04,  1.4746e-01, -5.4626e-03, -1.0254e-01,  4.5654e-02,
             -1.2759e-07, -5.4688e-02,  3.5248e-03, -8.7738e-05, -1.9922e-01, -4.1748e-02,  6.8359e-02, -1.0620e-02,
             -1.4221e-02,  3.1641e-01, -4.1504e-02, -9.7603e-07,  7.6599e-03, -3.8338e-04, -3.9673e-03, -2.5879e-02,
              1.4282e-02,  8.0078e-02,  3.9551e-02, -2.0264e-02, -1.9789e-05, -2.8992e-03,  2.9297e-02, -7.4768e-04,
             -1.5869e-02,  1.6403e-03, -2.0409e-04,  1.3733e-02, -6.7383e-02,  4.7852e-02, -2.8809e-02, -3.2485e-06,
             -7.1335e-04, -2.9053e-02, -3.4180e-03,  3.8574e-02,  1.1139e-03, -6.0081e-05, -1.7456e-02,  2.9541e-02,
              7.8613e-02, -1.5137e-02,  8.3923e-05, -2.1729e-02, -4.0588e-03, -4.7363e-02, -7.2021e-03, -3.1853e-04,
             -4.7852e-02, -9.7046e-03, -4.6539e-04, -1.2283e-03,  2.6123e-02,  5.1880e-03, -1.2665e-03, -5.5075e-05,
              3.5400e-03, -1.3504e-03,  2.8038e-04,  1.5869e-02, -1.0223e-03,  4.1809e-03, -1.9409e-02,  1.9043e-02,
              2.3560e-02,  5.7220e-05, -4.5776e-03, -2.2095e-02, -3.4027e-03,  9.1553e-03, -1.0376e-02, -7.3730e-02,
              5.2490e-02,  2.1172e-04, -5.6152e-02, -1.6235e-02,  1.5991e-02, -2.7588e-02,  6.9141e-05,  2.0447e-03,
             -5.1117e-04,  7.8125e-02, -3.9062e-02,  1.9824e-01, -1.9169e-04, -2.6733e-02, -3.8300e-03, -1.8921e-03,
             -7.9956e-03, -2.1484e-02, -1.9409e-02,  1.1658e-02,  3.6133e-02, -1.0010e-02,  1.0449e-01, -1.6846e-02,
              1.8692e-04, -2.0142e-02, -5.5313e-04, -6.6528e-03, -1.9897e-02,  1.3504e-03, -4.6387e-03, -1.9531e-03,
              6.5863e-06, -2.3438e-02,  1.9431e-05, -4.0039e-02, -2.7466e-02, -1.2390e-02,  4.6143e-02,  2.1680e-01,
              6.4697e-03, -1.3275e-03,  2.9175e-02, -8.2397e-03,  2.3127e-05, -4.4434e-02, -1.0431e-05,  2.0142e-02,
             -4.1504e-02,  4.1008e-05,  2.3804e-02, -6.8970e-03, -6.4941e-02, -6.5430e-02, -5.9204e-03, -3.1281e-04,
              2.8320e-02, -5.6763e-03, -1.0376e-03,  2.1118e-02, -2.3926e-02, -1.1841e-02,  1.1780e-02, -1.3306e-02,
             -3.1662e-04,  2.2461e-02,  2.0996e-02, -6.0320e-05,  5.7617e-02,  1.9932e-04, -1.0729e-04, -7.6599e-03,
             -6.2943e-04, -5.6076e-04,  6.5918e-03, -7.8201e-05, -1.9609e+00,  5.4932e-02,  1.3977e-02, -6.0797e-05,
              6.8054e-03,  6.3782e-03,  1.4648e-03,  1.7700e-02, -1.6724e-02, -4.2343e-04, -1.6235e-02,  1.6022e-03,
              4.2725e-02,  4.2725e-03,  3.2959e-02,  4.2319e-06, -4.9174e-06, -1.5497e-05, -4.8447e-04,  2.6855e-03,
             -2.7313e-03, -5.8289e-03,  4.7112e-04, -2.9602e-03,  4.1504e-02,  2.0874e-02,  5.4932e-03, -7.0190e-04,
             -9.0790e-04,  2.2583e-02,  2.3346e-03, -1.2573e-02,  4.7207e-05,  1.6556e-03, -1.9550e-04,  3.0518e-02,
              9.4727e-02, -1.6479e-03, -5.2643e-04, -8.8867e-02,  1.7624e-03, -4.4823e-05, -2.2984e-04,  2.1553e-04,
             -8.6060e-03,  2.5940e-04, -6.6895e-02, -3.1494e-02,  2.1935e-04, -1.6699e-01,  2.3499e-03,  2.7710e-02,
             -2.3270e-04,  4.0894e-03, -6.5918e-03, -1.0864e-02,  1.8845e-03,  2.3682e-02,  2.4261e-03, -2.7269e-06,
             -8.7891e-03,  3.1494e-02, -1.4709e-02,  6.6223e-03, -2.6855e-02,  1.2970e-04, -6.6895e-02, -5.0049e-02,
              3.9062e-02,  1.2268e-02, -9.5215e-03, -1.2891e-01,  1.0395e-04, -5.6076e-04,  6.0120e-03,  4.5471e-03,
              2.6611e-02,  2.1729e-02,  3.7575e-04, -1.3809e-03,  2.0504e-04,  5.3906e-01, -1.5163e-04,  1.4099e-02,
             -5.9509e-03, -5.5859e-01,  1.7334e-02, -1.6594e-04, -5.9082e-02, -1.0605e-03, -1.0742e-02,  1.1396e-04,
             -1.8799e-02, -2.3804e-02,  7.4387e-04,  2.8534e-03,  1.9684e-03,  5.4626e-03, -5.5176e-02, -7.2327e-03,
              2.9144e-03,  9.4604e-03, -1.7166e-03,  1.0107e-01,  2.5879e-02, -8.8120e-04, -1.1396e-04, -1.8477e-05,
              9.9945e-04,  1.0693e-01, -2.3926e-02, -1.4954e-02, -3.2959e-03,  1.0757e-03, -1.7334e-02, -2.5481e-06,
             -9.5215e-03, -5.5313e-05, -2.2316e-04, -2.4223e-04,  3.1494e-02, -4.5410e-02, -5.3406e-03,  0.0000e+00,
             -4.6730e-04,  7.8082e-06, -2.0142e-02,  3.5400e-02, -6.0081e-05, -4.1504e-03, -4.4250e-03,  5.9128e-04,
              1.6235e-02,  5.7678e-03, -1.6113e-02, -8.0490e-04, -6.0425e-03, -1.5199e-05, -1.4782e-05,  4.2725e-02,
             -1.5450e-04,  5.1498e-04,  3.1586e-03, -2.3193e-02, -1.1292e-03,  1.8787e-04,  1.1475e-02, -7.1335e-04,
             -3.6133e-02,  1.2970e-03, -1.1292e-03,  5.0964e-03, -1.9989e-03,  1.1673e-03,  3.5156e-02,  6.5327e-05,
              7.7637e-02,  1.5015e-02,  5.9814e-02,  4.0527e-02, -1.7700e-02,  4.8096e-02, -4.7302e-03,  1.0109e-04,
              3.5645e-02,  5.4626e-03,  7.3730e-02,  9.0942e-03,  2.5757e-02,  1.4782e-04, -1.5354e-04,  9.5703e-02,
             -4.5013e-04, -2.9419e-02, -3.9673e-03,  1.9043e-02, -2.4414e-03, -4.8523e-03, -7.4707e-02,  9.2163e-03,
              1.9550e-04,  1.4496e-03,  2.0264e-02,  1.3184e-01, -2.4414e-02, -4.7302e-04,  4.8523e-03,  2.0630e-02,
             -1.9932e-04,  1.8311e-02, -1.0559e-02,  2.3842e-05,  2.4902e-02, -1.7548e-04,  4.7207e-05, -2.5757e-02,
             -4.0527e-02,  6.4697e-03, -1.9073e-04,  1.3123e-03,  1.9684e-03, -1.2085e-02, -1.5747e-02,  3.3203e-02,
             -1.5855e-05,  1.7643e-05,  8.0109e-04, -4.6349e-04,  1.2283e-03,  5.9509e-03,  6.1768e-02, -4.1771e-04,
             -1.5442e-02,  5.2750e-06,  2.5195e-01,  3.2902e-05, -3.7537e-03, -4.8637e-05, -6.4373e-05, -4.6875e-02,
              6.9336e-02,  8.9355e-02, -1.9360e-04, -1.3794e-02, -3.9368e-03,  5.0964e-03, -7.2327e-03, -8.5831e-04,
              3.3203e-02,  1.6479e-02,  1.2493e-04, -1.4210e-04,  1.9264e-04,  1.0193e-02, -1.8845e-03,  3.5400e-02,
              3.5889e-02, -8.8692e-05,  7.0801e-03,  3.1281e-03,  6.1417e-04, -2.1484e-02,  1.7822e-02,  8.1177e-03,
              1.2573e-02, -2.6855e-02, -1.0315e-02,  1.8768e-03,  1.8457e-01,  7.7637e-02, -6.4392e-03, -3.1494e-02,
             -2.3961e-05, -2.1820e-03, -1.2085e-02,  1.9185e-07,  1.5015e-02, -2.6733e-02,  9.1797e-02, -1.7166e-05,
              5.2452e-05,  3.5667e-04, -7.0801e-03,  1.5450e-04, -4.4189e-02,  2.5330e-03,  1.6235e-02, -1.5869e-02,
             -4.6194e-06, -9.3750e-02, -6.9427e-04, -3.4912e-02, -1.9897e-02,  5.2002e-02, -3.7842e-02, -7.8613e-02,
              4.8828e-02, -2.1973e-02,  1.2878e-02, -7.5195e-02,  2.1094e-01,  4.2419e-03, -2.0599e-03,  1.5137e-02,
             -4.3297e-04, -2.5513e-02, -2.5630e-05,  1.6113e-02,  1.2512e-02,  3.4332e-04, -3.3617e-05, -2.2461e-02,
              7.4387e-04, -3.0327e-04, -6.0425e-03, -4.7302e-04,  1.5137e-02,  1.7853e-03,  3.4424e-02, -5.1758e-02,
              4.9133e-03,  1.7548e-03, -6.3896e-05,  5.9509e-03,  8.4229e-03, -5.0964e-03, -1.7822e-02, -5.1498e-04,
              2.7710e-02, -5.0735e-04,  2.2583e-02,  3.2616e-04, -3.2501e-03,  5.4932e-04,  1.4038e-02, -6.5308e-03,
              5.0781e-01, -9.2030e-05, -9.4604e-03,  3.4424e-02,  2.7299e-05, -7.9632e-05,  8.4639e-06, -9.7656e-03,
              5.3467e-02,  1.2436e-03,  0.0000e+00,  1.3638e-04,  6.8283e-04, -1.4420e-03, -6.7902e-04,  2.8839e-03,
              7.3242e-02, -1.9531e-02, -9.5825e-03,  1.8203e+00, -1.9169e-04, -9.6512e-04, -5.5313e-04, -4.2915e-05,
              9.8348e-06, -2.1851e-02,  7.7637e-02, -1.2695e-01, -3.9101e-04,  3.7384e-03,  2.3079e-04, -4.5654e-02,
             -1.2159e-05,  1.9989e-03, -4.4189e-02, -4.0283e-03, -3.0670e-03,  1.4746e-01, -2.3071e-02, -1.7090e-02,
             -2.5513e-02,  1.3199e-03,  1.9287e-02,  1.2894e-03,  4.6082e-03, -2.0905e-03, -4.1771e-04, -1.8677e-02,
              6.7139e-04,  2.0599e-04,  3.7079e-03,  5.6839e-04,  2.4414e-02,  2.9419e-02,  1.6594e-04, -3.0060e-03,
             -6.0730e-03,  1.9455e-04,  1.8799e-02,  4.4250e-03, -1.7456e-02,  8.6060e-03, -5.6076e-04, -5.5313e-05,
             -2.2339e-02, -4.2236e-02,  3.6812e-04,  2.8687e-02,  2.8419e-04,  5.9509e-04, -3.1090e-04, -1.1635e-04,
              1.5320e-02,  1.5747e-02, -2.2602e-04,  7.5195e-02,  1.6113e-02, -8.3008e-02,  1.3962e-03, -4.4189e-02,
              3.3447e-02, -3.3379e-05,  5.9814e-03,  9.8877e-03,  2.0752e-02, -1.3245e-02, -1.5717e-03,  5.1270e-03,
             -1.0986e-02,  2.0752e-02,  4.8828e-03,  2.2650e-05, -9.6130e-04,  2.4414e-04, -2.9663e-02, -6.2180e-04,
              4.2114e-03,  1.6861e-03, -1.0803e-02,  1.8311e-02,  5.9814e-02,  4.0627e-04, -7.7515e-03,  1.4496e-04,
              4.2969e-02,  1.7524e-05, -2.7710e-02, -2.2793e-04,  3.6621e-02,  2.3079e-04, -1.3794e-02,  6.1328e-01,
              5.5542e-03,  2.0801e-01,  5.2261e-04, -2.1729e-02, -2.1935e-04,  4.7607e-02,  4.7607e-03,  2.1839e-04,
             -4.5013e-04, -3.9673e-03, -1.6098e-03,  2.2583e-02, -2.8839e-03,  1.8597e-04, -1.0498e-02, -5.9509e-04,
              8.9169e-05,  2.2736e-03, -3.1494e-02, -1.2695e-01, -1.3477e-01, -5.1270e-02, -2.0020e-02,  2.4780e-02,
              7.4768e-03,  4.0527e-02, -1.5625e-02, -4.5410e-02,  1.5259e-02,  1.6113e-02, -4.1809e-03, -2.6107e-05,
             -3.5156e-02,  7.9727e-04, -5.8651e-05, -3.3379e-05, -9.9945e-04,  1.6332e-05,  6.4087e-03, -8.6212e-04,
              1.3733e-02,  2.5879e-02, -3.9307e-02, -1.6880e-04, -1.3275e-03,  1.0559e-02,  6.3477e-03, -1.3542e-04,
             -5.9891e-04,  6.5430e-02,  1.7822e-02, -4.5538e-05, -2.0386e-02, -1.6615e-06, -2.9602e-03,  5.9204e-03,
             -7.4387e-04, -3.0029e-02,  1.5137e-01,  1.8750e-01,  3.1662e-04, -4.8828e-03,  1.5527e-01,  1.1396e-04,
             -7.6599e-03, -1.9989e-03, -1.1749e-03, -1.9431e-05, -8.1543e-02, -1.8120e-05,  6.0791e-02, -2.7222e-02,
              4.5300e-06,  2.5868e-05,  1.9836e-03,  9.6436e-03,  5.7617e-02,  5.1758e-02, -1.7319e-03, -7.5684e-03,
             -9.1553e-03,  4.8584e-02, -1.0498e-02,  4.0770e-05,  7.2632e-03,  9.2773e-03, -6.5002e-03, -1.7480e-01,
             -2.1338e-05,  6.7871e-02, -5.5664e-02,  2.3346e-03,  1.9531e-03, -2.9297e-03,  7.3853e-03, -3.1128e-02,
              6.1798e-04,  3.6469e-03, -3.2959e-02, -1.3770e-01, -2.5635e-02,  2.1729e-02, -2.6367e-02, -9.6798e-05,
             -5.0293e-02,  1.4305e-04,  9.8267e-03,  1.0596e-01, -9.6191e-02, -1.6235e-02, -8.2779e-04, -1.4099e-02,
             -3.1738e-02, -8.8501e-03, -1.0400e-01,  1.1902e-03, -2.6855e-03,  3.7354e-02, -3.7193e-04, -7.6294e-03,
              2.5988e-05,  1.4267e-03, -3.3008e-01, -2.2292e-05, -4.5776e-03,  4.1504e-02,  2.0996e-02,  2.9755e-03,
             -3.6812e-04,  1.8387e-03, -1.8677e-02, -1.1902e-03,  4.8828e-03, -2.9175e-02, -4.9133e-03,  3.1982e-02,
             -1.4771e-02,  4.2419e-03,  2.3315e-02, -7.0496e-03,  4.5410e-02, -1.1047e-02, -2.4796e-04, -4.6492e-05,
              1.3232e-05, -2.6489e-02,  1.9684e-03, -1.6895e-01, -2.1118e-02, -2.9883e-01,  1.0071e-02,  1.7456e-02,
              4.2725e-04,  3.5286e-05, -5.4016e-03, -1.6403e-03, -4.3869e-04,  3.8330e-02, -2.2217e-02,  1.1658e-02,
              5.1880e-04, -1.6689e-04, -1.5488e-03,  1.4961e-05,  4.4107e-05,  5.8413e-05,  1.9531e-03,  2.1973e-02,
             -2.7275e-04,  7.4387e-04,  7.2861e-04,  4.8340e-02, -7.0190e-04,  7.3433e-05, -5.3024e-04, -2.6226e-06,
              2.4719e-03,  6.7139e-03, -5.3024e-04,  3.2715e-02,  3.2806e-04,  4.8096e-02, -2.0142e-02, -1.3770e-01,
             -9.1934e-04, -1.6403e-04,  2.0386e-02, -8.8501e-03,  1.4160e-01,  2.6245e-02,  7.8125e-03,  1.2398e-04,
             -1.3379e-01,  3.0029e-02, -1.1841e-02, -2.2583e-02,  7.5195e-02, -3.6865e-02, -3.1128e-02, -3.5553e-03,
              1.0633e-04, -5.4688e-02,  7.6294e-05,  2.0020e-02,  1.3428e-02, -3.1006e-02, -4.3640e-03, -6.4850e-05,
              3.0823e-03, -3.6240e-04, -4.5898e-02, -3.0151e-02,  1.2500e-01,  9.1171e-04, -2.1362e-04, -8.6784e-05,
             -4.4250e-04,  5.8350e-02,  1.0223e-03,  7.9956e-03, -8.6060e-03, -2.4796e-04, -1.6968e-02,  4.1016e-02,
             -6.1035e-03, -9.3994e-03,  1.2988e-01,  2.7161e-03,  4.0527e-02, -6.5994e-04,  1.1719e-01, -7.5102e-06,
              9.1553e-03, -1.7548e-03,  3.1982e-02,  3.1250e-02,  1.0061e-04, -7.6294e-03,  1.5918e-01, -1.6846e-02,
              2.3438e-02,  5.1270e-03,  2.4414e-03,  7.9956e-03, -4.3213e-06,  1.4687e-04, -3.6955e-05,  2.8801e-04,
             -4.1504e-03, -2.1696e-05,  3.7956e-04, -1.3733e-02,  1.5747e-02,  9.3750e-02,  7.4863e-05, -4.2114e-03,
              7.4387e-04, -2.6611e-02, -2.0508e-02,  3.5889e-02,  2.6703e-03,  5.8594e-02,  6.4468e-04, -4.5471e-03,
             -2.5757e-02, -1.0010e-02, -3.0640e-02, -1.4526e-02,  1.3161e-04, -3.6865e-02,  1.9043e-02, -6.4392e-03,
              1.5259e-02, -1.3123e-02,  1.9409e-02,  6.6223e-03, -1.1492e-04,  3.3203e-02, -1.1368e-03, -1.8066e-02,
             -1.2266e+00, -5.4626e-03, -3.7842e-03, -8.1253e-04, -3.6240e-04,  2.2095e-02,  1.3962e-03, -4.3945e-02,
             -1.3733e-02,  7.8735e-03, -2.8372e-05,  2.2316e-04,  1.1414e-02, -3.3203e-02,  9.5825e-03, -5.3955e-02,
              1.5991e-02,  4.9919e-07,  1.5503e-02,  8.1539e-05,  3.9673e-04, -8.6975e-04, -1.1230e-02, -4.9744e-03,
             -1.6928e-05, -1.0437e-02,  2.3193e-02, -4.5898e-02,  4.5654e-02,  1.2512e-02,  1.4062e-01, -1.6235e-02,
              4.2725e-03, -4.9219e-01,  5.0659e-03,  6.1035e-05,  1.6968e-02,  1.1015e-04, -2.0142e-03, -1.2457e-05,
             -3.5553e-03,  1.6785e-04, -3.2227e-02, -9.2316e-04, -1.0910e-03,  1.0498e-01, -9.3262e-02,  3.6774e-03,
              2.2705e-02, -1.2589e-04,  3.8086e-02, -3.0899e-04,  1.8433e-02, -2.6978e-02,  1.6479e-02,  3.6955e-05,
             -7.3547e-03, -7.4208e-06,  2.3460e-04,  6.4392e-03, -6.9046e-04, -2.4261e-03,  1.9741e-04, -2.3079e-04,
             -1.0452e-03,  3.5477e-04,  5.4443e-02, -1.1169e-02,  5.0537e-02,  5.0293e-02, -1.9836e-04, -2.4414e-03,
              2.5635e-02,  7.6172e-02, -4.0771e-02,  6.3477e-02, -4.3945e-03, -2.1118e-02,  1.2817e-03, -1.9684e-03,
             -7.4005e-04, -1.1563e-05,  2.4796e-04, -4.7607e-03,  3.2616e-04, -5.1025e-02,  4.0527e-02,  4.5013e-04,
             -1.2512e-03, -4.5061e-05, -1.6602e-02, -3.8574e-02,  9.0790e-04,  2.5635e-02, -8.3008e-03,  1.1978e-03,
             -5.2261e-04, -5.6076e-04,  2.2888e-03, -4.5635e-08,  3.5645e-02, -1.0952e-06, -4.3678e-04,  3.5156e-02,
              4.0283e-02, -5.4359e-05, -3.7231e-03,  5.5176e-02, -3.1128e-03,  6.9275e-03,  5.9814e-03,  5.9814e-02,
              7.8613e-02, -7.8735e-03,  2.9541e-02, -4.4861e-03, -3.2654e-03,  1.1902e-03, -7.7148e-02,  2.7222e-02,
              5.9204e-03, -3.1891e-03,  1.5198e-02, -3.1281e-03,  3.7231e-03,  4.8637e-05,  3.8719e-04, -2.3956e-03,
              1.6602e-02,  2.1648e-04, -8.1177e-03, -1.2024e-02, -3.6011e-03, -2.4872e-03,  3.5095e-04,  3.0327e-04,
             -2.4902e-02, -1.7405e-05,  2.2827e-02, -5.3711e-02,  6.9618e-05,  3.8330e-02,  1.6212e-04, -9.0332e-03,
             -4.5166e-02, -1.8555e-02,  2.0294e-03, -3.8300e-03, -3.7305e-01,  1.5723e-01,  1.0071e-02, -6.3419e-05,
              4.4189e-02, -1.4725e-03, -7.9727e-04,  1.2939e-02, -3.0029e-02,  9.4727e-02, -6.5994e-04,  2.4796e-05,
             -1.4709e-02,  8.4839e-03,  2.9297e-02,  6.4453e-02,  4.1199e-03,  3.6001e-05, -3.8477e-01, -3.1281e-04,
              3.4668e-02,  2.3041e-03, -2.2095e-02, -1.5320e-02, -9.1553e-03,  5.9814e-03, -1.8311e-02, -1.7700e-03,
             -2.0752e-02,  3.1586e-03,  1.1108e-02, -1.2024e-02,  1.3855e-02,  3.8574e-02,  1.6251e-03,  9.3262e-02,
             -1.8883e-04,  4.2419e-03,  1.5182e-03,  9.9945e-04,  5.4016e-03, -2.0142e-03,  3.2349e-03,  4.4678e-02,
             -3.8719e-04,  4.9133e-03, -3.3722e-03,  2.8610e-04,  1.6357e-02,  2.1210e-03,  1.9550e-05,  1.8799e-02,
              4.2725e-02,  8.2520e-02, -1.9165e-02,  1.0452e-03,  2.5781e-01,  2.8564e-02, -1.3828e-05,  1.2695e-02,
             -6.4392e-03,  4.5166e-03,  2.2602e-04, -1.2064e-04,  7.8735e-03,  2.9325e-05, -3.4027e-03,  2.4048e-02,
             -3.4912e-02, -6.2180e-04,  2.5513e-02, -3.0994e-05,  2.0752e-03, -1.8921e-02,  2.0996e-02, -2.0862e-05,
             -2.6398e-03, -1.2146e-02, -6.2561e-03, -3.1128e-02,  2.1606e-02,  4.6997e-03, -1.3733e-03, -3.3789e-01,
             -3.7003e-04, -7.2266e-02,  2.2125e-04, -2.2531e-05, -3.7842e-03, -2.7222e-02, -2.7466e-02,  1.0498e-02,
              3.5645e-02, -9.9945e-04, -1.0071e-03,  2.7618e-03, -4.0245e-04,  5.0068e-05,  1.1719e-02,  6.5231e-04,
             -9.4986e-04, -3.8605e-03,  3.2425e-04, -1.0864e-02, -2.6550e-03, -3.8605e-03, -2.9144e-03, -4.2188e-01,
             -8.3984e-02, -2.9175e-02, -1.0010e-02,  7.4506e-06,  2.6367e-02,  2.5558e-04, -2.4536e-02, -8.9355e-02,
              2.8992e-03, -9.3359e-01,  1.4893e-02,  3.6469e-03, -6.2561e-03, -8.8501e-03, -1.0400e-01,  4.7445e-05,
             -6.6376e-04, -6.3171e-03, -3.4424e-02, -2.9053e-02,  7.5340e-05, -2.8687e-02,  3.5858e-03, -8.8867e-02,
              5.7812e-01, -1.7776e-03, -6.1798e-04, -2.8534e-03, -6.2500e-02, -4.6082e-03, -6.3477e-02,  7.1289e-02,
             -1.1536e-02, -1.8188e-02,  3.2997e-04,  7.1777e-02, -2.3193e-03,  1.1673e-03,  1.1802e-05, -8.7172e-07,
              1.0223e-03,  1.0071e-02,  3.6621e-02,  5.7373e-03, -5.6458e-04,  4.8218e-03,  8.2031e-02, -1.7624e-03,
             -5.8594e-03, -2.1820e-03,  4.0527e-02, -4.7493e-04,  1.6846e-02,  5.2490e-03,  2.6550e-03,  3.5763e-05,
              2.5391e-01, -8.7619e-06, -1.1597e-03,  7.5684e-03, -2.2507e-04,  4.5898e-02,  1.9302e-03,  5.4688e-02,
              1.0596e-01,  2.2461e-02,  3.0060e-03, -3.4523e-04, -3.7109e-02, -1.1182e-01, -3.1738e-02,  1.0529e-03,
              4.8523e-03,  3.1494e-02, -1.8692e-04,  2.3041e-03, -1.8597e-04,  2.3682e-02, -9.2983e-05,  1.9043e-02,
             -2.4048e-02, -3.3760e-04, -2.8229e-04, -3.3569e-03, -1.6689e-04,  2.3315e-02,  3.2959e-02,  5.6458e-04,
             -3.1738e-02, -5.5176e-02, -7.4768e-03,  1.4343e-03, -5.6076e-04, -8.2520e-02, -1.6479e-02, -4.3701e-02,
              6.0425e-03,  4.8340e-02,  2.6001e-02,  1.0443e-04, -1.0742e-01, -6.0547e-02,  8.2016e-04, -8.3008e-03,
             -3.5889e-02,  6.7234e-05,  1.7212e-02,  1.3123e-02,  8.6426e-02,  1.5381e-02,  0.0000e+00,  6.2500e-02,
              1.3428e-02, -1.0059e-01, -1.0254e-02, -3.3951e-04, -2.2705e-02, -4.6484e-01,  2.2095e-02,  1.3733e-02,
             -5.0964e-03, -2.2949e-02,  6.7139e-03, -2.6758e-01,  2.0142e-02, -1.1841e-02, -2.0752e-02,  1.7319e-03,
             -2.1267e-04, -7.1106e-03,  5.2154e-06, -1.8359e-01,  1.0071e-02, -3.4180e-02, -1.2354e-01,  2.7275e-04,
              5.7678e-03,  8.5938e-02, -1.1902e-02, -1.4526e-02,  4.5598e-06,  1.5137e-01, -8.6914e-02, -6.8359e-03,
              5.6152e-02,  1.8358e-05, -8.9722e-03,  3.5400e-03,  4.5598e-06,  2.1172e-04, -6.6280e-05,  6.5231e-04,
              8.1177e-03, -8.4473e-02, -6.2988e-02, -4.6921e-04, -6.3477e-02,  4.0054e-05,  5.8746e-04,  1.0620e-02,
             -1.0824e-04, -2.1076e-04,  9.5844e-05,  4.4434e-02,  2.1118e-02, -6.0120e-03,  5.6152e-02, -3.5400e-02,
              8.5449e-03, -9.2163e-03,  2.6855e-02, -1.8066e-02,  3.4912e-02, -7.0095e-05,  3.0273e-01,  1.7242e-03,
             -2.4292e-02, -2.9297e-02, -4.4922e-02, -2.8372e-05, -1.0061e-04, -2.8320e-02,  1.3657e-03,  1.0681e-02,
             -1.6785e-03,  1.0803e-02,  2.0264e-02,  9.3842e-04,  4.2725e-02, -3.3447e-02, -1.1475e-01, -5.2643e-04,
              1.1683e-04,  2.5940e-03, -2.8198e-02,  2.5940e-04,  2.4796e-05,  7.1716e-04,  1.3504e-03, -3.5400e-02,
             -2.9541e-02, -3.4332e-04, -7.2002e-05, -6.5918e-03,  1.2085e-02, -5.6763e-03, -7.3242e-04,  1.4901e-05,
             -1.4973e-04,  1.4709e-02,  6.8359e-02, -1.9789e-05,  1.9989e-03, -1.3199e-03, -1.6406e-01, -7.9346e-03,
              5.1880e-04,  3.8330e-02, -6.1340e-03, -4.6158e-04, -2.4872e-03, -2.1362e-02,  7.1716e-04, -3.4523e-04,
             -3.0518e-03,  5.3955e-02, -1.0300e-04, -9.4727e-02, -2.5558e-04,  3.0060e-03,  4.5395e-04,  9.6798e-05,
             -2.1820e-03, -2.7008e-03,  8.7280e-03, -1.7242e-03,  1.6357e-02,  1.0010e-02, -5.3711e-03,  1.0559e-02,
              1.7166e-03, -1.9989e-03,  6.1417e-04,  7.7724e-05,  2.2583e-02, -1.9043e-02, -4.1962e-05,  3.0762e-02,
              1.8234e-03, -1.5068e-04,  4.4823e-05, -4.9316e-02, -7.0572e-05, -2.1606e-02,  1.1444e-03,  1.8066e-02,
             -5.7617e-02,  1.0986e-02,  9.1076e-05, -1.2779e-04, -1.1963e-02, -8.9111e-03,  6.4087e-04,  1.3550e-02,
             -8.6594e-04, -2.5879e-02,  6.2943e-04,  6.8665e-03,  1.9073e-04, -1.0529e-03, -1.8555e-02,  2.4915e-05,
              4.3640e-03,  8.7280e-03,  2.0264e-02, -1.6880e-04,  2.2583e-02, -5.4550e-04, -7.0801e-03, -6.4453e-02,
              6.9336e-02, -3.1738e-02, -4.0245e-04,  9.7275e-04,  3.0920e-07, -1.6357e-02, -2.1458e-04, -6.2943e-04,
             -6.7383e-02,  1.0742e-02,  4.4922e-02,  2.0898e-01, -1.2891e-01,  6.5918e-02,  2.8442e-02, -4.0283e-02,
              2.0415e-06, -1.0681e-02, -2.8198e-02, -3.5048e-05,  4.9072e-02,  2.3071e-02,  4.9805e-02, -1.0986e-03,
             -2.4292e-02,  1.3885e-03, -1.5442e-02, -4.2480e-02,  7.9155e-05,  1.1492e-04,  2.2411e-04, -1.0010e-02,
              6.1035e-03,  5.6458e-03, -2.3193e-02,  4.7493e-04, -5.4932e-04,  2.4780e-02,  1.2207e-03,  8.4839e-03,
              3.5667e-04, -4.3392e-05,  1.8463e-03, -1.6724e-02, -1.8799e-02, -6.5231e-04, -4.4922e-02,  3.8910e-04,
             -6.7234e-05, -1.2779e-04, -4.3945e-02, -7.2754e-02,  4.7493e-04,  7.4158e-03, -2.1362e-02,  2.6001e-02,
              1.8164e-01, -3.8338e-04, -8.8379e-02, -1.3542e-04,  8.3496e-02,  2.5988e-05,  5.1514e-02, -1.5198e-02,
              3.5706e-03, -1.7822e-02, -7.8613e-02,  2.0312e-01, -1.0376e-02, -6.7383e-02,  2.8320e-01, -5.4359e-05,
              1.9455e-04,  1.5564e-03, -1.1047e-02, -8.2016e-04,  7.4219e-02,  2.6855e-03, -1.7285e-01, -3.4668e-02,
              5.9509e-03, -1.1730e-04,  3.9795e-02, -1.8677e-02, -4.3030e-03,  9.0942e-03, -3.2959e-02,  6.2012e-02,
              5.7068e-03, -2.5749e-04, -4.4346e-05,  1.9043e-02,  3.4912e-02,  6.4087e-03,  3.9453e-01,  1.9226e-03,
             -5.6396e-02, -3.6240e-04,  1.8120e-04,  9.2773e-03, -1.1719e-02,  1.1182e-01, -5.8889e-05,  4.8340e-02,
             -2.7008e-03,  4.3869e-04, -3.4766e-01, -4.9770e-06,  4.2200e-05, -1.1658e-02,  1.2598e-01,  3.1445e-01,
              8.7544e-07, -1.0681e-02, -1.9455e-04, -7.2266e-02, -1.0193e-02, -2.2888e-04, -4.2725e-03, -4.4823e-05,
              2.6611e-02,  1.1597e-02, -2.8687e-02, -5.0293e-02,  2.8610e-06, -1.0132e-02,  4.7874e-04, -3.4531e+00,
              4.3213e-02,  7.5150e-04, -2.4414e-04,  3.3417e-03,  2.3651e-03,  1.0986e-01, -2.8687e-02,  6.5565e-06,
              7.8125e-03,  3.0273e-02, -1.8978e-04,  1.9646e-04,  1.2012e-01,  6.6895e-02,  9.2773e-02, -6.8665e-05,
             -2.8931e-02, -1.1921e-04, -2.3193e-02, -2.1515e-03,  1.8677e-02, -1.1015e-04,  9.8145e-02,  1.7357e-04,
              3.7842e-02, -3.0762e-02,  3.4912e-02,  1.9073e-04, -3.8528e-04,  1.7456e-02,  3.6240e-04, -3.3691e-02,
              8.7261e-05, -2.8687e-03,  1.2512e-03, -1.9908e-05, -6.9580e-03, -1.7212e-02,  4.7119e-02,  3.4485e-03,
              4.3869e-04,  1.9531e-02,  9.5215e-03,  7.1335e-04, -2.2430e-03, -2.0020e-02,  3.2425e-04,  2.8014e-05,
              3.8147e-03,  1.7944e-02,  3.1982e-02, -2.8076e-03,  1.8164e-01,  9.8145e-02,  1.1587e-04, -7.6904e-03,
             -2.3270e-04, -1.1826e-03, -2.3926e-01,  2.0264e-02,  3.0823e-03, -2.0508e-02, -1.6785e-04,  3.6621e-02,
             -1.4526e-02,  8.4839e-03,  2.4658e-02,  1.1414e-02,  2.4567e-03, -8.9453e-01, -1.0547e-01, -2.5787e-03,
              4.8828e-02,  3.2227e-02, -2.4023e-01, -5.9366e-05, -4.9438e-03, -7.5684e-03, -1.0449e-01, -2.8809e-02,
             -3.7842e-02,  1.9908e-05, -3.7891e-01,  2.3926e-02, -4.0436e-04,  3.7994e-03,  1.7242e-03, -9.8228e-05,
              3.2227e-02,  3.1128e-03,  5.0049e-03, -2.1606e-02,  5.7373e-02,  5.2643e-04,  1.1253e-04,  1.6975e-04,
              2.0142e-02,  3.0327e-04, -1.5616e-05, -5.1498e-04,  2.6345e-05, -5.8899e-03, -6.0797e-06,  9.8877e-03,
              4.2236e-02,  3.6430e-04, -5.4443e-02, -6.3782e-03,  1.9073e-04,  4.3213e-02,  6.3477e-02,  7.4158e-03,
             -1.0010e-01, -4.5166e-03, -2.1210e-03, -1.4267e-03, -1.5736e-04, -1.0803e-02,  3.7354e-02,  3.7842e-02,
             -1.2589e-04, -1.5259e-02,  5.2643e-04,  7.9956e-03, -8.0585e-05, -6.4373e-05, -3.5400e-03, -1.3574e-01,
             -3.4424e-02,  1.5411e-03, -1.6968e-02,  4.7363e-02,  1.1902e-03, -9.7656e-03, -7.7637e-02, -3.5645e-02,
              1.9922e-01, -2.8320e-02,  4.1992e-02, -2.6703e-04, -2.8461e-06, -5.5237e-03,  7.9274e-06,  8.1543e-02,
              2.2888e-05,  4.4189e-02,  6.9336e-02, -2.4292e-02, -1.0109e-04, -7.2266e-02, -1.5076e-02,  5.7129e-02,
              8.6594e-04, -8.5831e-04, -8.0566e-03,  4.5967e-04,  5.0354e-03,  6.4941e-02,  6.2988e-02,  9.2316e-04,
             -2.9325e-05,  1.5259e-02,  1.1902e-02, -4.5013e-04,  7.2479e-05,  1.0109e-04, -3.2902e-05,  2.9053e-02,
             -1.7578e-02,  7.6675e-04, -5.6458e-03,  6.2500e-02, -7.9297e-01, -1.2360e-03, -4.8637e-05,  1.0559e-02,
             -3.0756e-05, -1.5259e-02, -1.2109e-01,  1.4954e-02, -3.1494e-02,  9.0332e-02,  2.3693e-06, -3.2806e-04,
              5.0781e-02, -5.7068e-03, -5.2452e-06,  3.9978e-03,  8.7891e-03,  5.4016e-03,  1.7357e-04,  5.8350e-02,
             -3.8338e-04,  8.0490e-04,  2.1210e-03, -1.7090e-02, -2.6398e-03,  1.4687e-04,  1.7834e-04, -4.1008e-04,
             -2.8198e-02,  4.4107e-06, -3.5156e-02,  3.8147e-05,  1.5527e-01, -4.9438e-03,  1.6785e-04, -1.7834e-04,
             -4.3945e-03, -3.0212e-03, -1.3638e-04,  2.2949e-02,  2.1118e-02,  4.5002e-06,  1.3123e-02, -1.9836e-03,
             -3.0518e-03, -3.2471e-02, -7.9590e-02,  5.1025e-02, -2.3193e-03, -3.8086e-02,  3.3417e-03,  2.0146e-05,
             -1.3916e-02,  3.5553e-03, -8.5449e-03, -6.3965e-02,  9.9487e-03,  7.5817e-05,  2.7771e-03, -8.4229e-03,
              5.6152e-02,  1.5616e-05,  1.8005e-03,  3.2471e-02,  9.4414e-05, -1.3489e-02, -2.7618e-03, -7.5989e-03,
              9.1270e-07, -1.0376e-02, -6.9580e-03, -3.3722e-03,  6.3171e-03,  1.2817e-02,  3.4790e-03, -1.5442e-02,
              8.2397e-03,  3.9864e-04, -8.4877e-05, -4.3678e-04,  2.9297e-02, -1.2329e-02,  4.9438e-03,  5.7220e-04,
              1.1426e-01,  5.7068e-03,  7.4158e-03,  3.5400e-03,  5.0049e-03, -7.3547e-03, -2.2888e-04,  2.7418e-05,
             -9.4604e-03, -9.5367e-06, -1.8406e-04, -4.4556e-03,  1.2636e-05,  7.3853e-03, -1.2779e-04,  1.2146e-02,
             -3.8818e-02, -4.2319e-06, -5.3406e-03,  1.1921e-04, -7.3016e-06,  3.5400e-02, -4.5166e-03,  1.8433e-02,
              8.7023e-06, -5.6076e-04,  1.8954e-05,  2.5195e-01, -3.6469e-03,  6.4468e-04,  1.1292e-02, -1.7452e-04,
             -7.1049e-05, -9.9121e-02, -1.9455e-03,  1.8066e-02,  1.0071e-03, -2.7084e-04,  1.1658e-02,  1.4160e-02,
              5.5542e-03,  2.3270e-04, -1.3733e-02,  8.7357e-04, -1.7212e-02, -5.7861e-02,  6.6406e-02, -9.9945e-04,
             -6.8665e-05,  1.8262e-01,  1.4404e-02,  1.5378e-05,  9.3994e-03, -3.8574e-02,  1.6357e-02,  3.2227e-02,
             -2.8320e-02, -8.0566e-03,  9.5749e-04, -3.3936e-02,  3.1710e-05, -4.3335e-03,  2.1240e-02,  1.9455e-04,
              4.6539e-04, -1.2793e-01, -1.7090e-02,  1.1768e-01, -1.2085e-02,  5.8899e-03, -5.3711e-03, -4.2236e-02,
              4.0283e-02,  1.1139e-03, -5.7373e-02, -2.1057e-03, -5.0049e-03, -2.8687e-02,  1.4019e-04, -2.8491e-05,
              1.2573e-02,  4.6387e-02,  5.3406e-04, -3.7842e-03, -1.2329e-02, -1.0824e-04, -3.3398e-01,  1.0252e-05,
              2.5749e-04, -1.0986e-02, -2.6855e-03, -5.3467e-02, -5.3406e-03,  2.3193e-02, -7.1526e-05, -4.7852e-02,
              1.2207e-02,  4.6387e-02,  3.0151e-02, -9.0332e-03,  6.5994e-04, -4.6082e-03, -1.8883e-04, -6.2180e-04,
             -2.5977e-01,  5.4199e-02, -2.0862e-05, -1.2684e-04, -1.3123e-02,  1.9073e-04,  2.3682e-02,  1.3542e-04,
              1.1444e-03,  5.0735e-04, -6.5918e-02,  9.5844e-05, -3.2043e-04, -2.9449e-03, -5.5908e-02, -1.7548e-03,
             -1.0559e-02,  1.3367e-02, -8.6670e-03,  2.3499e-03,  1.5259e-03, -6.6895e-02,  2.8931e-02,  2.9053e-02,
             -1.3855e-02, -6.6528e-03,  5.5237e-03,  9.2506e-05,  7.1526e-05, -1.7090e-02, -1.0352e-01,  3.4912e-02,
              1.1215e-03,  4.2419e-03, -1.2268e-02, -2.1240e-02, -2.8564e-02,  4.0054e-04,  1.1780e-02,  3.0518e-05,
              3.4142e-04, -6.3477e-03,  3.3203e-02,  1.0498e-02,  3.2349e-03,  1.1094e+00, -3.8086e-01, -2.6855e-02,
              8.4229e-03,  5.4169e-04, -4.6387e-03,  8.0109e-05, -2.1057e-03, -5.2734e-02, -2.2339e-02,  4.6692e-03,
             -5.1758e-02,  6.7520e-04, -2.9755e-04, -5.8350e-02,  1.9455e-04, -7.1106e-03, -1.2085e-02,  4.7119e-02,
             -1.0742e-02, -1.0443e-04, -6.9275e-03,  9.8267e-03, -2.5635e-02,  1.9684e-03, -1.9409e-02, -2.1851e-02,
             -6.9046e-04, -1.3733e-03, -1.8555e-02, -1.0061e-04, -3.5018e-06, -9.2773e-03,  3.7689e-03, -5.8838e-02,
              1.6846e-02,  5.8746e-04, -2.7657e-04,  6.1768e-02,  4.9591e-04, -1.1139e-03,  4.7913e-03,  3.2959e-02,
             -4.0039e-02,  4.7119e-02,  1.0803e-02,  1.4648e-03,  4.7302e-04,  5.5542e-03, -1.0443e-04, -6.8359e-03,
              1.8066e-02, -1.9531e-02, -7.9346e-04, -4.9805e-02,  9.1553e-03,  2.1118e-02, -8.1062e-06, -1.8358e-05,
             -2.0599e-04,  2.7588e-02,  4.5166e-03, -2.5635e-03,  1.9409e-02, -1.8848e-01,  5.0964e-03, -5.4688e-02,
             -1.0777e-04, -4.2236e-02, -2.5787e-03,  2.0264e-02,  7.1289e-02,  4.9438e-03, -5.2261e-04,  1.7822e-02,
              1.3733e-02,  8.5449e-02,  4.2236e-02,  8.0078e-02,  1.2054e-03,  3.8574e-02, -3.0327e-04, -2.8491e-05,
             -9.5215e-03, -2.5757e-02,  8.4305e-04,  8.1055e-02, -5.0964e-03,  9.8267e-03, -3.4790e-03, -1.4160e-02,
             -1.2360e-03,  3.9795e-02,  1.7944e-02,  9.7046e-03, -5.8105e-02,  1.0633e-04, -1.4160e-02, -1.8120e-04,
             -7.2754e-02, -9.2506e-05,  2.2461e-02, -4.3945e-03, -2.2070e-01, -8.2016e-05, -8.5449e-02,  3.6316e-03,
             -1.2500e-01, -1.8164e-01,  1.1292e-02, -7.2937e-03,  9.2773e-03, -7.8735e-03,  2.3438e-02, -5.7602e-04,
             -1.0681e-02, -1.8463e-03,  9.1934e-04,  3.2425e-05,  6.7139e-04,  1.2146e-02,  3.4637e-03,  3.1738e-03,
             -1.9741e-04, -2.0264e-02,  1.4877e-04,  1.8215e-04, -4.7607e-03, -2.1362e-02, -7.7148e-02,  3.0151e-02,
              2.0020e-02, -4.4189e-02,  1.8555e-02,  6.4453e-02,  2.3804e-02,  1.8677e-02, -4.7852e-02, -9.7752e-05,
              3.4180e-02,  2.0742e-05,  5.7220e-04, -3.4332e-04,  6.3672e-01, -1.3489e-02, -9.0332e-03, -2.4536e-02,
             -4.7119e-02,  2.1851e-02, -6.9275e-03,  1.2054e-03,  3.0518e-03, -6.2988e-02, -4.9072e-02, -1.6235e-02,
             -1.7881e-06, -3.3203e-02, -4.5776e-03, -1.5545e-04,  7.7438e-04,  1.4954e-03, -1.6689e-05,  4.4632e-04,
              6.2988e-02,  6.2500e-02,  3.9368e-03, -1.8555e-02, -1.5640e-04,  2.3560e-02, -7.8735e-03,  4.9133e-03,
              2.8839e-03,  2.6611e-02, -8.6594e-04,  2.0695e-04, -9.6560e-06, -5.4550e-04,  4.0771e-02,  8.2493e-05,
              2.9602e-03, -6.5613e-04,  1.7822e-02,  2.4902e-02, -5.3955e-02,  4.5586e-04,  9.6191e-02, -9.6321e-05,
              1.1572e-01, -2.5513e-02, -3.4912e-02,  3.6523e-01,  1.1914e-01, -1.0254e-02, -4.2439e-05, -6.5994e-04,
              9.5215e-03,  3.3188e-04,  4.3809e-06, -1.8406e-04,  3.6621e-02, -1.9302e-03, -3.2997e-04, -2.3682e-02,
              1.8921e-03, -4.5061e-05,  1.3123e-02,  4.7913e-03,  1.5869e-02,  3.0518e-03, -1.3428e-02,  3.6621e-02,
             -2.9564e-04, -1.7643e-04,  1.0547e-01, -4.1809e-03, -3.5286e-05,  3.2616e-04, -3.1710e-05,  1.0376e-02,
              1.1108e-02,  1.4941e-01,  1.1963e-02,  2.0599e-04,  1.3657e-03, -5.5664e-02,  2.5177e-03,  1.7944e-02,
              2.4872e-03, -3.8605e-03, -3.5858e-04,  6.4453e-02, -1.9165e-02,  1.9824e-01,  8.2493e-05,  3.4912e-02,
              2.3193e-02, -2.8076e-02,  3.7109e-02,  1.2634e-02,  4.1008e-04, -6.2500e-02, -1.2207e-01, -2.7148e-01,
             -1.7700e-02, -8.3008e-03,  2.5391e-02, -1.2970e-04, -4.6730e-04,  4.1389e-04, -7.2956e-05, -1.8433e-02,
             -3.2663e-05,  7.9956e-03, -1.7090e-02,  4.4060e-04, -7.0572e-05, -6.5613e-04, -2.6855e-03,  2.2827e-02,
              2.8931e-02, -2.7771e-03,  7.5531e-04,  1.0864e-02, -2.7222e-02, -5.7220e-04, -2.7588e-02, -2.0146e-05,
              2.4567e-03, -1.3550e-02,  1.9897e-02,  1.2512e-02,  2.9297e-02,  1.3292e-05,  1.4114e-03, -4.5117e-01,
             -2.8125e-01, -2.5868e-05,  1.0443e-04,  2.7466e-03, -2.1606e-02, -2.1118e-02, -4.6143e-02, -1.2451e-02,
             -1.2634e-02, -4.1771e-04, -2.3926e-02, -2.2278e-03, -6.5918e-02,  1.3000e-02,  1.2695e-02,  1.0132e-02,
              1.4587e-02,  5.1575e-03, -3.1128e-02, -3.6621e-02, -2.3041e-03, -1.2878e-02, -2.7418e-05, -9.4238e-02,
             -1.8359e-01,  1.9531e-03,  8.2397e-03, -3.4332e-04,  2.7657e-05, -5.6152e-03, -2.1875e-01,  5.0964e-03,
             -1.8597e-05,  3.0884e-02, -4.2725e-02, -3.4142e-04,  2.4796e-04, -7.0572e-04, -3.3569e-04, -3.6621e-02,
              1.0437e-02, -7.2266e-02, -1.4954e-03, -1.0803e-02,  7.2266e-02,  4.6143e-02, -6.8188e-05,  1.6602e-02,
              1.0300e-03, -2.0447e-03, -2.3499e-03,  8.7261e-05,  8.9111e-03,  2.5757e-02,  2.7954e-02, -4.9561e-02,
              4.6387e-02, -4.4922e-02,  1.8433e-02, -4.3335e-03, -7.7438e-04,  6.2866e-03, -6.5308e-03, -3.9291e-04,
              3.6562e+00,  3.3379e-04, -4.4189e-02,  1.4267e-03,  1.6113e-01, -5.6763e-03,  2.5635e-03,  3.2959e-02,
             -2.1729e-02,  3.4668e-02, -4.1260e-02, -2.3560e-02,  1.3199e-03, -6.1417e-04, -6.5002e-03, -4.5776e-05,
             -1.6113e-02,  3.8147e-03,  1.6992e-01, -3.1982e-02,  8.9722e-03, -3.8330e-02, -1.9775e-02, -6.5430e-02,
             -2.2030e-04,  2.2705e-02, -6.8054e-03, -4.2725e-03,  7.9155e-05,  1.5488e-03,  4.1016e-02, -6.5234e-01,
              4.9316e-02,  1.2512e-03, -6.5430e-02, -3.7109e-02, -2.8076e-02, -2.7084e-04,  2.0996e-02,  2.4902e-02,
             -2.8076e-03, -8.9645e-05,  5.8105e-02, -2.0020e-02, -5.9814e-03, -3.9062e-03, -1.6357e-02,  5.5847e-03,
             -3.0273e-02,  6.2256e-03,  1.5015e-02, -4.0588e-03, -3.2043e-04, -1.1475e-02,  8.1539e-05,  2.6464e-05,
              3.9551e-02,  7.3730e-02,  4.6631e-02,  1.6235e-02,  2.6367e-02,  2.0874e-02,  5.3787e-04,  6.5994e-04,
             -3.7994e-03, -7.9727e-04,  4.8218e-03, -9.1553e-03, -1.5564e-03,  1.0840e-01,  1.5259e-05, -1.2146e-02,
             -5.2795e-03,  9.7656e-02, -6.7055e-06,  3.9307e-02, -2.5330e-03, -6.7711e-05,  8.3923e-05,  2.3651e-03,
              8.3542e-04, -1.7090e-01, -2.7466e-02,  5.8174e-05,  2.8038e-04, -1.7700e-02,  1.1683e-04,  1.1292e-03,
              1.1719e-02, -3.6469e-03, -1.2159e-04,  2.0752e-03,  6.0547e-02,  3.1738e-02, -1.8239e-05,  6.5918e-02,
             -2.4796e-04, -2.6733e-02, -1.4424e-05,  2.0469e+00, -5.5695e-04, -1.0498e-02, -3.0151e-02,  7.3853e-03,
             -1.1969e-04, -7.0496e-03,  4.3106e-04,  7.5531e-04, -3.1738e-03,  8.1635e-04, -9.7656e-03,  1.5259e-03,
             -7.3853e-03,  5.1758e-02,  2.8076e-02, -7.2327e-03,  2.0264e-02, -1.8692e-03, -7.8735e-03, -1.3065e-04,
             -7.9346e-04,  1.5625e-01,  3.8330e-02, -3.1006e-02,  6.2988e-02,  3.2043e-04, -1.6211e-01, -1.1520e-03,
              2.8968e-05, -3.2425e-04,  3.1494e-02, -9.5825e-03,  3.1250e-02, -1.9653e-02,  1.3447e-04,  1.5747e-02,
              5.1022e-05,  8.2397e-04, -3.1006e-02, -6.3965e-02, -2.8908e-06, -1.0376e-02,  9.7656e-03, -5.3644e-06,
              4.1809e-03, -2.4959e-07, -5.2002e-02,  1.3123e-03,  2.9755e-03, -1.0300e-03, -1.9968e-06, -3.4180e-02,
             -6.7711e-05, -2.7832e-02, -1.9226e-03,  1.2360e-03, -4.6158e-04, -2.9297e-02,  7.1049e-05,  5.8289e-03,
             -9.3079e-04, -3.7402e-06,  2.7539e-01,  6.8970e-03, -8.1062e-06,  1.4687e-04,  3.9062e-02,  2.7161e-03,
              4.4434e-02,  9.6130e-04, -1.2207e-02, -1.7212e-02, -1.8799e-02, -2.9683e-05, -4.5776e-03,  5.6624e-06,
             -4.6997e-03,  2.5034e-05,  3.2715e-02, -3.4668e-02, -5.2490e-02, -6.0272e-04, -2.0874e-02,  1.9646e-04,
             -3.7537e-03,  3.9101e-05, -7.2956e-05,  7.3624e-04, -1.1902e-03,  1.4019e-04, -9.9945e-04,  6.0797e-05,
             -1.0742e-02, -7.2266e-02, -6.3419e-05,  7.9870e-06,  7.2327e-03, -5.1880e-03,  8.8867e-02,  1.5926e-04,
             -6.8359e-01,  1.1206e-04,  8.3618e-03, -1.4496e-03,  2.3535e-01,  6.9885e-03,  1.3916e-02, -3.2806e-04,
             -5.6763e-03,  4.2725e-03,  3.0823e-03,  4.1389e-04,  9.1553e-05,  1.2817e-03, -1.2878e-02, -1.1719e-02,
             -9.5215e-03,  7.8678e-05, -4.4434e-02, -1.0498e-01, -2.1729e-02,  2.6894e-04, -4.7363e-02,  1.5503e-02,
              9.3384e-03, -7.0496e-03,  1.7383e-01, -1.7090e-03,  8.4961e-02, -4.5776e-03, -5.2246e-02, -5.2246e-02,
              4.3106e-04, -2.2984e-04, -9.4986e-04,  6.8665e-03, -4.5776e-05,  1.0498e-02, -1.2817e-02, -1.7188e-01,
             -1.9836e-03, -2.9053e-02, -2.3804e-02,  5.0545e-05, -9.5825e-03,  1.0986e-02,  1.5640e-03,  1.4526e-02,
             -1.8358e-05,  5.2490e-02, -1.3657e-03, -1.9836e-03,  7.3910e-05, -1.5991e-02, -4.1485e-05,  6.3171e-03,
             -9.0942e-03, -1.0376e-02, -3.5156e-02, -2.8442e-02,  1.0252e-05,  2.6001e-02,  1.7334e-02, -8.1253e-04,
              6.8359e-02, -1.0498e-02, -3.6716e-05,  9.9093e-07, -1.4221e-02, -6.4453e-02,  2.0996e-02, -4.0829e-06,
             -5.0354e-04,  1.8066e-02, -3.8330e-02,  2.8381e-03,  1.6975e-04,  5.8594e-03,  4.5013e-04,  1.5564e-02,
             -1.4587e-02,  1.5106e-03, -5.5664e-02, -2.8229e-03,  7.0801e-02,  1.9684e-03, -3.1250e-02,  6.3782e-03,
              1.5564e-02,  2.9144e-03, -6.6280e-05, -2.4261e-03,  5.3711e-03, -7.3242e-04, -1.4893e-02,  1.7212e-02,
              1.2493e-04,  1.7166e-03,  2.3071e-02, -1.9409e-02, -2.3041e-03, -5.9128e-04,  5.8899e-03, -2.4414e-04,
             -2.6822e-05,  2.7344e-02,  3.8086e-02,  1.2158e-01, -1.2589e-04, -1.5015e-02, -1.6251e-03,  3.1250e-01,
              2.1839e-04,  1.4114e-04, -9.6436e-03, -4.4250e-03,  4.1016e-02,  3.9062e-02,  3.5400e-02,  2.0020e-02,
             -4.0588e-03,  3.1250e-02, -8.3618e-03,  1.9073e-05,  9.7046e-03, -2.3926e-02, -2.9419e-02, -1.8406e-04,
             -3.6865e-02, -8.9722e-03,  1.6403e-03,  3.8818e-02,  8.3008e-03,  2.9297e-03,  2.5482e-03, -2.0020e-02,
             -5.2734e-02,  2.6894e-04,  1.9531e-02, -4.6539e-04,  4.1016e-02, -2.6855e-02,  1.2268e-02,  2.1118e-02,
              1.4160e-02, -8.7891e-03,  3.2715e-02,  8.6784e-05,  4.4922e-02, -1.2100e-05,  2.0599e-03, -2.9297e-03,
             -2.5635e-03, -1.3062e-02, -1.1719e-02, -9.0027e-04, -1.4877e-04, -7.1777e-02,  2.8076e-02,  3.4904e-04,
              2.9785e-02,  1.1063e-03, -1.2878e-02, -5.6458e-04, -3.7575e-04, -8.5831e-04, -1.0498e-02,  1.9653e-02,
             -9.5844e-05, -1.2493e-04,  2.4319e-04, -1.7732e-06, -4.6566e-07, -2.1851e-02,  4.2915e-04,  1.4587e-02,
              3.0160e-05, -5.1498e-05, -8.7280e-03, -3.0708e-04,  3.7354e-02,  2.6489e-02, -7.1526e-05, -1.9150e-03,
             -1.0061e-04,  8.8501e-04,  4.1992e-02, -3.1471e-04, -3.3203e-02,  8.4400e-05,  2.8198e-02, -4.0771e-02,
              1.0840e-01, -2.7466e-03,  4.3945e-03, -1.9836e-04, -1.5736e-04,  2.2949e-02,  2.2827e-02, -2.0885e-04,
             -4.4250e-03, -7.7209e-03, -1.0529e-03, -3.7537e-03,  5.8899e-03, -6.5002e-03,  1.7738e-04, -1.7624e-03,
              3.8910e-03, -3.0845e-06, -1.4687e-04, -1.8406e-04, -1.9287e-02,  3.6377e-02, -3.2043e-03, -1.2302e-04,
              2.5749e-05, -2.5940e-03, -1.0803e-02,  1.1133e-01,  4.8447e-04, -1.0157e-04,  3.1982e-02, -5.0781e-02,
              1.2131e-03,  2.0313e-04, -2.2583e-02, -5.9814e-02,  2.2705e-02,  1.1523e-01,  4.7363e-02, -1.2398e-05,
             -3.5156e-02, -6.7520e-04, -2.9419e-02,  1.6327e-03, -5.0964e-03, -5.6152e-03, -5.9009e-06,  1.3809e-03,
             -1.9287e-02,  3.6478e-05, -1.7212e-02, -1.2360e-03,  5.4016e-03, -5.6458e-04,  8.8882e-04, -3.3691e-02,
             -1.3504e-03,  7.2632e-03,  3.2806e-03, -2.0142e-03,  7.4768e-03, -3.4424e-02, -2.9755e-04, -2.6321e-04,
              9.8877e-03,  3.0327e-04,  1.0071e-02,  7.3433e-05, -4.5654e-02, -4.3945e-03, -2.6550e-03,  1.6117e-04,
              3.2812e-01,  3.0640e-02,  2.2705e-02, -2.2344e+00,  4.7119e-02,  6.6223e-03, -1.3550e-02, -7.9870e-06,
             -4.7607e-02, -1.8005e-03,  3.7598e-02, -8.0109e-05,  1.2817e-02,  1.6357e-02, -2.6855e-03,  3.6621e-02,
              4.9829e-05,  5.2795e-03, -3.7354e-02,  2.8610e-04, -2.3460e-04,  1.3542e-04,  2.6001e-02, -1.1353e-02,
             -2.1744e-04, -1.2779e-04, -5.0781e-02, -1.7578e-02, -5.5695e-04,  1.5564e-02,  3.4027e-03, -2.5368e-04,
             -2.0117e-07, -7.6904e-03, -1.3733e-02, -1.3962e-03,  4.8828e-03, -4.7922e-05,  1.2329e-02,  1.9908e-05,
              9.7168e-02,  3.4424e-02,  3.1853e-04, -2.9297e-03, -4.6143e-02,  1.2970e-04,  2.6894e-04, -6.0730e-03,
              4.0233e-06,  7.5684e-03,  2.6550e-03,  4.6875e-02, -6.1035e-02,  3.6774e-03,  5.6982e-05,  1.0254e-02,
              1.6861e-03,  1.1921e-05, -4.4632e-04,  1.4258e-01,  3.8862e-05,  4.7302e-03, -3.7598e-02,  2.5635e-02,
              7.2956e-05,  5.3406e-03, -2.1458e-04, -6.4087e-03, -2.2168e-01,  1.1730e-04,  1.1108e-02, -3.0151e-02,
              2.3961e-05, -3.3569e-04, -1.4941e-01, -6.3477e-02,  8.3618e-03,  3.1250e-02,  2.9907e-03, -2.1515e-03,
             -2.8320e-02,  8.7738e-04,  9.7046e-03, -1.4687e-04,  2.0703e-01,  2.0266e-05, -1.7047e-05, -4.0588e-03,
              4.4678e-02, -1.8768e-03,  3.3569e-03,  1.6968e-02,  1.8477e-05, -5.9128e-04, -2.4414e-02, -7.6294e-03,
              8.3496e-02, -9.8228e-05,  6.4087e-03,  9.7752e-05, -2.5000e-01,  3.3951e-04,  2.5977e-01,  1.5137e-01,
              4.8523e-03,  7.6172e-02, -1.4842e-05, -1.6797e-01, -5.3883e-05, -5.1025e-02, -2.5940e-04,  1.3916e-02,
             -1.5945e-03,  3.9291e-04, -2.2888e-03,  9.8228e-05, -4.2725e-04, -6.4941e-02, -2.8931e-02,  3.3203e-02,
              2.1648e-04,  9.3384e-03, -5.0735e-04, -1.1353e-02, -7.2266e-02, -1.3580e-03, -3.2715e-02, -6.4468e-04,
              7.2122e-06,  6.4453e-02,  1.9287e-02,  2.9297e-02,  2.0020e-02,  1.6499e-04, -2.8687e-02,  1.8188e-02,
              2.3346e-03,  2.6367e-02, -2.3041e-03, -1.4544e-05,  3.3569e-04,  2.7588e-02, -5.9605e-06, -5.7678e-03,
              5.3711e-03, -4.5898e-02,  9.3750e-01,  3.4637e-03,  6.2561e-04,  6.1095e-06, -3.7909e-05,  2.1094e-01,
              1.2256e-01,  4.1504e-03, -1.5974e-05,  1.1902e-03, -1.9264e-04,  1.1414e-02,  3.1662e-04,  5.3711e-03,
              1.0586e-04, -5.7373e-02, -3.3569e-04,  3.7231e-03,  2.3071e-02, -2.2125e-03,  3.7193e-04, -1.5450e-04,
              4.7207e-05,  1.4420e-03,  6.4697e-03, -1.2817e-03, -4.1199e-03, -3.4485e-03, -1.4038e-02,  2.6131e-04,
             -5.8889e-05,  3.3398e-01, -4.3945e-02,  1.4210e-04,  9.1553e-03,  5.4321e-03, -8.2493e-05, -2.5024e-02,
              1.3428e-02, -7.0496e-03, -5.9082e-02,  2.5940e-03, -2.8711e-01,  1.1719e-01, -4.3335e-03, -1.2988e-01,
             -5.6885e-02,  7.6294e-04,  1.4114e-03, -3.5156e-02,  1.4210e-04,  2.8320e-02,  2.7466e-04,  6.6406e-02,
              1.0681e-02,  2.4605e-04,  1.0848e-05,  3.2234e-04,  4.9353e-05, -1.3794e-02, -2.1851e-02, -1.2634e-02,
              3.6865e-02, -4.9316e-02, -8.7619e-06,  1.4404e-02,  6.7188e-01,  2.6001e-02,  2.3603e-05, -1.5503e-02,
             -1.2451e-01, -4.7207e-05, -3.4180e-02, -1.8262e-01, -5.1022e-05, -4.0627e-04, -1.2875e-04, -1.3916e-02,
             -4.7302e-03,  5.6267e-05, -2.9206e-05,  1.4551e-01,  2.0142e-02, -1.1027e-05,  7.4463e-03, -1.4771e-02,
             -2.0996e-02,  2.6611e-02, -6.8054e-03,  1.5564e-02, -1.1475e-02, -2.8320e-01, -3.2806e-03, -1.2390e-02,
             -1.2085e-02, -2.2125e-03, -3.4180e-02,  2.4719e-03,  3.8910e-03, -1.1368e-03, -2.4048e-02, -2.9057e-06,
             -9.3937e-05,  5.3048e-06, -3.2617e-01, -3.2471e-02,  4.4250e-04,  2.5368e-04,  5.8594e-02, -3.2501e-03,
             -3.2471e-02,  2.5787e-03,  1.0605e-03, -2.1729e-02, -2.8014e-06, -1.0742e-02,  1.6797e-01,  9.8267e-03,
              5.4443e-02,  4.7363e-02, -1.5259e-04, -9.2163e-03, -8.1787e-03,  1.9521e-06,  1.2634e-02, -1.3794e-02]]
            hidden_states = torch.tensor(layer_0, dtype=torch.bfloat16, device=hidden_states_device)
            topk_weights = torch.tensor([[0.2192, 0.2091, 0.1996, 0.1506, 0.0700, 0.0679, 0.0489, 0.0347],
            [0.2740, 0.1783, 0.1433, 0.1048, 0.0883, 0.0842, 0.0709, 0.0561],
            [0.2713, 0.2548, 0.1030, 0.0881, 0.0789, 0.0777, 0.0675, 0.0587],
            [0.2816, 0.1682, 0.1656, 0.1310, 0.0929, 0.0648, 0.0572, 0.0387]], dtype=torch.float32, device=hidden_states_device)
            topk_ids = torch.tensor([[ 72, 101,  28,   5,  57,  60,  31,  45],
            [  5,  32,  28, 101,  16,  72,  57,  58],
            [ 89,  31,  19,  95,  51,  74,  84,  56],
            [ 88,  52,  68,  95,  70, 122,  31, 110]], dtype=torch.int32, device=hidden_states_device)
            router_logits = torch.tensor([[-7.1875, -6.7188, -5.6875, -7.7500, -5.6875, -2.3750, -5.3750, -5.0312, -5.2500, -6.5312, -5.5000, -5.5000,
             -4.4688, -6.0000, -5.5312, -5.3438, -3.8750, -4.9688, -7.0625, -6.5312, -7.3438, -4.7500, -7.7188, -6.0625,
             -5.1250, -4.7500, -6.5312, -5.8438, -2.0938, -5.4688, -4.9375, -3.5000, -4.8750, -4.6875, -4.8750, -4.3750,
             -5.8125, -7.5938, -7.4375, -6.1562, -6.6250, -5.9688, -4.8438, -5.6562, -5.9062, -3.8438, -6.3750, -5.8750,
             -4.3125, -5.2500, -5.6562, -7.0312, -5.2500, -5.2188, -6.1562, -5.0625, -4.6562, -3.1406, -3.8750, -6.7812,
             -3.1719, -5.8125, -5.4375, -4.9375, -4.3750, -7.0938, -5.4688, -5.8750, -4.6875, -6.3750, -5.6250, -4.6875,
             -2.0000, -6.3438, -5.6562, -5.3438, -5.3750, -6.7500, -8.0000, -4.8438, -4.4688, -7.7188, -7.2188, -4.9375,
             -4.9688, -4.9062, -6.8438, -6.2500, -6.4062, -6.1562, -5.2188, -6.5625, -5.7188, -6.2188, -6.4062, -4.0312,
             -6.2500, -4.9062, -6.4375, -5.2812, -5.5625, -2.0469, -6.3750, -7.2188, -6.8438, -5.5000, -5.0312, -4.0625,
             -6.5625, -5.4688, -5.9062, -6.3125, -6.3438, -7.2188, -6.4688, -6.0312, -7.0938, -5.8125, -6.3125, -6.5625,
             -5.0625, -5.4062, -5.9375, -5.1875, -5.5625, -7.2812, -5.5000, -6.4375],
            [-7.2188, -7.3750, -4.8438, -6.6250, -5.4375, -1.8047, -5.8125, -4.5625, -5.4688, -5.3438, -6.4062, -5.8750,
             -5.2188, -6.4062, -5.4375, -4.6250, -2.9375, -5.0938, -7.0625, -7.5938, -7.0312, -4.9062, -6.5000, -8.3125,
             -4.1875, -5.6875, -6.7500, -4.7500, -2.4531, -4.5625, -4.0938, -3.5469, -2.2344, -3.5156, -4.5625, -3.6094,
             -5.6250, -6.5000, -6.3125, -5.3125, -5.7812, -6.7500, -5.0938, -4.5625, -6.3750, -3.4375, -6.5000, -6.0938,
             -5.1875, -5.9062, -5.2188, -6.7812, -6.9062, -5.4688, -7.2500, -5.5312, -4.9688, -3.1562, -3.3906, -6.2500,
             -3.5625, -5.9062, -5.0312, -4.7812, -3.6562, -7.3125, -6.1250, -8.0625, -6.8438, -6.6875, -5.4375, -5.3125,
             -2.9844, -6.3750, -6.9688, -4.2500, -5.7500, -6.7500, -7.1562, -4.4062, -6.2188, -6.6250, -6.1875, -6.0000,
             -5.9375, -4.1250, -6.3438, -6.0938, -7.0625, -6.4688, -5.1875, -6.4375, -7.1250, -6.2812, -6.8750, -5.9688,
             -5.5938, -5.2812, -6.5938, -5.6250, -5.7812, -2.7656, -6.1562, -6.5312, -6.2500, -4.6250, -4.1250, -3.6562,
             -7.5312, -6.3125, -6.0312, -7.2812, -5.4375, -7.7500, -6.3438, -6.0938, -5.4688, -7.0938, -6.5938, -6.7188,
             -5.0312, -5.8125, -5.7500, -6.1875, -6.0938, -6.7500, -5.8125, -6.5625],
            [-7.1250, -7.1875, -5.7500, -8.0000, -7.2500, -6.4688, -7.6250, -6.4688, -6.2188, -7.6250, -5.9688, -5.0938,
             -3.7500, -6.0000, -5.4375, -7.1875, -6.3750, -4.3438, -7.8125, -3.0000, -6.8750, -6.5312, -6.8750, -7.5625,
             -3.6875, -7.8750, -7.4062, -6.0938, -7.1562, -5.9688, -6.5938, -2.0938, -8.1875, -4.5312, -6.5938, -5.4688,
             -5.4688, -8.8125, -8.0000, -6.9688, -6.3750, -6.0000, -7.1562, -7.2812, -4.2500, -3.7031, -7.6562, -7.0625,
             -5.7812, -7.5938, -5.4062, -3.2656, -4.8125, -6.0625, -3.7812, -7.6562, -3.5625, -7.0938, -6.0938, -6.9688,
             -6.3438, -7.0000, -4.6562, -7.2188, -7.3750, -7.5000, -6.0625, -7.7188, -4.1875, -5.9688, -5.3125, -6.8438,
             -5.7812, -7.4062, -3.2812, -7.2500, -7.1250, -6.8125, -6.7188, -5.8750, -5.3125, -8.5625, -8.0625, -7.9375,
             -3.4219, -6.0000, -7.7500, -7.3438, -6.4375, -2.0312, -6.1875, -7.6250, -7.6562, -5.8438, -5.8750, -3.1562,
             -5.1875, -6.0625, -5.9688, -4.9688, -5.4062, -4.4375, -6.9375, -7.8125, -7.5000, -5.9375, -5.3125, -4.9688,
             -8.6875, -6.9688, -5.4375, -4.7500, -8.2500, -8.5625, -7.9688, -7.2812, -8.0625, -4.4375, -7.8750, -6.2500,
             -5.7812, -6.3438, -4.5312, -6.6875, -6.5312, -4.5938, -6.1562, -7.0625],
            [-6.5312, -5.9375, -5.9688, -7.2812, -7.5938, -8.0625, -9.5000, -7.5938, -6.5000, -8.7500, -6.8125, -6.3125,
             -6.2500, -6.3125, -6.4062, -7.3750, -7.7812, -4.4062, -7.4688, -6.2188, -5.8750, -7.2188, -7.5000, -7.7812,
             -3.9219, -9.1875, -5.7500, -6.5312, -8.6875, -5.5938, -7.0625, -3.3750, -9.1250, -5.7188, -6.7500, -6.5625,
             -6.6562, -9.5625, -8.2500, -7.3750, -5.4688, -5.8125, -8.9375, -7.3438, -5.0938, -3.9844, -8.5625, -6.9688,
             -5.9062, -7.8125, -5.3750, -5.1562, -2.2969, -6.6250, -6.1875, -8.3750, -4.7188, -8.3125, -7.3125, -5.3125,
             -6.5625, -6.9688, -4.6875, -7.5312, -8.1875, -8.2500, -5.5938, -8.2500, -2.3125, -7.9688, -2.8906, -7.6562,
             -7.0938, -8.1875, -4.3438, -7.5000, -7.8750, -6.2188, -6.9375, -6.3438, -6.1875, -8.0625, -7.8125, -9.1250,
             -3.9219, -7.8750, -7.1250, -7.2500, -1.7812, -5.1562, -7.6562, -8.5000, -9.2500, -4.5312, -5.9375, -2.5469,
             -6.6875, -5.3750, -6.6562, -4.1875, -5.9375, -5.2188, -5.7188, -7.7500, -8.1250, -7.0938, -6.0938, -5.9375,
             -9.3125, -7.7812, -3.7656, -4.9375, -9.0625, -8.3750, -8.7500, -8.4375, -8.0625, -5.9062, -8.1875, -7.7188,
             -5.7500, -6.0938, -3.2500, -6.0312, -6.6875, -5.0000, -6.3438, -7.9688]], device=hidden_states.device)
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardTopKOutput
            topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
            # logger.info(f"layer_id: {self.layer_id}, topk_output: {topk_output}")
        # if self.layer_idsr_id: {self.layer_id} hidden_states: {hidden_states}, topk_output: {topk_output}")

        torch.set_printoptions(
            threshold=float(1000),  # 显示所有元素
            precision=4,  # 小数点后4位
            linewidth=80  # 每行宽度
        )
        # dispatch send
        for round_id in range(self.num_rounds):
            peo_overlap_args = PeoOverlapArgs(
                use_expert_overlap=True,
                num_rounds=self.num_rounds,
                round_id=round_id,
                send_num_sms=self.num_device_sms,
                recv_num_sms=self.num_device_sms if round_id == 0 else self.num_deepep_sms,
                hook_use_comm_stream=False,
            )
            state = self.dispatcher.dispatch_a_peo(
                hidden_states=hidden_states,
                topk_output=topk_output,
                peo_overlap_args=peo_overlap_args,
            )
            states.append(state)

        # dispatch recv and GEMM
        for round_id in range(self.num_rounds):
            dispatch_output = self.dispatcher.dispatch_b_peo(
                inner_state=states[round_id],
            )

            PeoDeepEPMoE.gemm_stream.wait_stream(current_stream)
            with torch.cuda.stream(PeoDeepEPMoE.gemm_stream):
                num_experts_per_round = self.num_experts // self.num_ranks // self.num_rounds
                start_idx = num_experts_per_round * round_id
                end_idx = start_idx + num_experts_per_round
                torch.cuda.synchronize()
                if self.layer_id == 0 and dispatch_output.hidden_states.shape[0] > 0:
                    logger.info(f"layer_id: {self.layer_id}, round_id: {round_id}, dispatch recv: hidden_states: {dispatch_output.hidden_states[start_idx:end_idx]}")
                moe_hidden_state = self.run_moe_core(dispatch_output, start_idx, end_idx)
                moe_hidden_states.append(moe_hidden_state)
                gemm_done_event = torch.cuda.Event()
                PeoDeepEPMoE.gemm_stream.record_event(gemm_done_event)
                gemm_done_events.append(gemm_done_event)

        # combine send
        for round_id in range(self.num_rounds):
            current_stream.wait_event(gemm_done_events[round_id])
            peo_overlap_args = PeoOverlapArgs(
                use_expert_overlap=True,
                num_rounds=self.num_rounds,
                round_id=round_id,
                send_num_sms=self.num_device_sms if round_id == (self.num_rounds - 1) else self.num_deepep_sms,
                recv_num_sms=self.num_device_sms,
                hook_use_comm_stream=False,
                is_x_in_round=True,
            )
            if self.layer_id == 0 and moe_hidden_states[round_id][0].shape[0] > 0:
                logger.info(f"layer_id: {self.layer_id}, round_id: {round_id}, combine send: hidden_states: {moe_hidden_states[round_id][0]}")

            combine_state = self.dispatcher.combine_a_peo(
                combine_input=moe_hidden_states[round_id],
                peo_overlap_args=peo_overlap_args,
            )

        # combine recv
        combined_x = self.dispatcher.combine_b_peo(inner_state=combine_state)

        if self.layer_id == 0 and combined_x.shape[0] > 0:
            logger.info(f"layer_id: {self.layer_id}, combine_x recv: hidden_states: {combined_x}")
        current_stream.wait_stream(PeoDeepEPMoE.gemm_stream)
        return combined_x

    def forward_overlap_2_3(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        current_stream: torch.cuda.Stream,
    ):
        gemm_done_events = list()
        moe_hidden_states = list()

        global dispatch_output, combine_state
        hook_use_default_stream = self.overlap_method == 2
        # dispatch and GEMM
        for round_id in range(self.num_rounds):
            # dispatch send
            peo_overlap_args = PeoOverlapArgs(
                use_expert_overlap=True,
                num_rounds=self.num_rounds,
                round_id=round_id,
                send_num_sms=self.num_device_sms,
                recv_num_sms=self.num_device_sms if round_id == 0 else self.num_deepep_sms,
                hook_use_comm_stream=not hook_use_default_stream,
            )
            state = self.dispatcher.dispatch_a_peo(
                hidden_states=hidden_states,
                topk_output=topk_output,
                peo_overlap_args=peo_overlap_args,
            )

            # dispatch recv
            if hook_use_default_stream:
                dispatch_output = self.dispatcher.dispatch_b_peo(
                    inner_state=state,
                )
            else:
                self.comm_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.comm_stream):
                    dispatch_output = self.dispatcher.dispatch_b_peo(
                        inner_state=state,
                    )
                PeoDeepEPMoE.gemm_stream.wait_stream(self.comm_stream)

            # GEMM
            PeoDeepEPMoE.gemm_stream.wait_stream(current_stream)
            with torch.cuda.stream(PeoDeepEPMoE.gemm_stream):
                num_experts_per_round = self.num_experts // self.num_ranks // self.num_rounds
                start_idx = num_experts_per_round * round_id
                end_idx = start_idx + num_experts_per_round
                moe_hidden_state = self.run_moe_core(dispatch_output, start_idx, end_idx)
                moe_hidden_states.append(moe_hidden_state)
                gemm_done_event = torch.cuda.Event()
                PeoDeepEPMoE.gemm_stream.record_event(gemm_done_event)
                gemm_done_events.append(gemm_done_event)

        # combine send
        for round_id in range(self.num_rounds):
            peo_overlap_args = PeoOverlapArgs(
                use_expert_overlap=True,
                num_rounds=self.num_rounds,
                round_id=round_id,
                send_num_sms=self.num_device_sms if round_id == (self.num_rounds - 1) else self.num_deepep_sms,
                recv_num_sms=self.num_device_sms,
                hook_use_comm_stream=False,
                is_x_in_round=True,
            )
            current_stream.wait_event(gemm_done_events[round_id])
            if not hook_use_default_stream:
                current_stream.wait_stream(self.comm_stream)
            combine_state = self.dispatcher.combine_a_peo(
                combine_input=moe_hidden_states[round_id],
                peo_overlap_args=peo_overlap_args,
            )

        # combine recv
        if hook_use_default_stream:
            combined_x = self.dispatcher.combine_b_peo(inner_state=combine_state)
        else:
            self.comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.comm_stream):
                combined_x = self.dispatcher.combine_b_peo(inner_state=combine_state)
            current_stream.wait_stream(self.comm_stream)

        current_stream.wait_stream(PeoDeepEPMoE.gemm_stream)
        return combined_x

    def forward_overlap_4(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        current_stream: torch.cuda.Stream,
    ):
        gemm_done_events = list()
        moe_hidden_states = list()
        # dispatch
        self.comm_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.comm_stream):
            dispatch_output = self.dispatch(hidden_states, topk_output)

            # current_stream.wait_stream(comm_stream)
            for round_id in range(self.num_rounds):
                num_experts_per_round = self.num_experts // self.num_ranks // self.num_rounds
                start_idx = num_experts_per_round * round_id
                end_idx = start_idx + num_experts_per_round
                moe_hidden_state = self.run_moe_core(dispatch_output, start_idx, end_idx)
                moe_hidden_states.append(moe_hidden_state)
                gemm_done_event = torch.cuda.Event()
                self.comm_stream.record_event(gemm_done_event)
                gemm_done_events.append(gemm_done_event)

        # combine send
        with torch.cuda.stream(current_stream):
            for round_id in range(self.num_rounds):
                current_stream.wait_event(gemm_done_events[round_id])
                peo_overlap_args = PeoOverlapArgs(
                    use_expert_overlap=True,
                    num_rounds=self.num_rounds,
                    round_id=round_id,
                    send_num_sms=self.num_device_sms if round_id == (self.num_rounds - 1) else self.num_deepep_sms,
                    recv_num_sms=self.num_device_sms,
                    hook_use_comm_stream=False,
                    is_x_in_round=True,
                )
                combine_state = self.dispatcher.combine_a_peo(
                    combine_input=moe_hidden_states[round_id],
                    peo_overlap_args=peo_overlap_args,
                )

            # combine recv
            combined_x = self.dispatcher.combine_b_peo(inner_state=combine_state)

        current_stream.wait_stream(self.comm_stream)
        return combined_x

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        current_stream = torch.cuda.current_stream()

        with torch.cuda.stream(current_stream):
            if self.overlap_method == 1:
                return self.forward_overlap_1(hidden_states, topk_output, current_stream)
            elif self.overlap_method == 2 or self.overlap_method == 3:
                return self.forward_overlap_2_3(hidden_states, topk_output, current_stream)
            elif self.overlap_method == 4:
                return self.forward_overlap_4(hidden_states, topk_output, current_stream)
            else:
                raise ValueError(f"Invalid overlap_method: {self.overlap_method}")


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        if is_peo_enabled():
            return PeoDeepEPMoE
        else:
            return DeepEPMoE
    if get_moe_a2a_backend().is_ascend_fuseep():
        return NpuFuseEPMoE

    if get_moe_runner_backend().is_flashinfer_trtllm():
        # NEW: Direct FP4 detection (bypasses EP requirements)
        # Check for FP4 quantization with TRTLLM flag, regardless of EP
        # FlashInferFP4MoE must be paired with ModelOptNvFp4FusedMoEMethod.
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFP4MoE

            return FlashInferFP4MoE
        elif (
            quant_config is None
            or quant_config.get_name() == "fp8"
            or quant_config.get_name() == "modelopt_fp8"
            or quant_config.get_name() == "compressed_tensors"
        ):
            # FlashInferFusedMoE support bf16, fp8 and compressed_tensors
            return FlashInferFusedMoE

    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    return FusedMoE

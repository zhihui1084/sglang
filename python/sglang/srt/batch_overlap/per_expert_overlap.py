from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.moe.topk import TopKOutput

GEMM_STREAM = torch.cuda.Stream()

@dataclass
class PeoOverlapArgs:
    use_expert_overlap: bool
    num_rounds: int
    round_id: int
    send_num_sms: int
    recv_num_sms: int
    hook_use_comm_stream: bool = False
    is_x_in_round: bool = False


def gen_new_dispatch_output(
    experts,
    dispatch_output,
    round_id: int,
    need_clone: bool,
):
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput
    num_experts_per_round = experts.num_experts // experts.num_ranks // experts.num_rounds
    start_idx = num_experts_per_round * round_id
    end_idx = start_idx + num_experts_per_round
    mask = torch.zeros_like(dispatch_output.masked_m, dtype=torch.bool)
    mask[start_idx:end_idx] = True
    masked_m = dispatch_output.masked_m * mask
    return DeepEPLLDispatchOutput(
        hidden_states=dispatch_output.hidden_states.clone() if need_clone else dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale.clone() if need_clone else dispatch_output.hidden_states_scale,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
        masked_m=masked_m,
        expected_m=dispatch_output.expected_m,
    )


def forward_overlap_1(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
):
    global combine_state, dispatch_output
    states = list()
    gemm_done_events = list()
    moe_hidden_states = list()

    # dispatch send
    for round_id in range(experts.num_rounds):

        peo_overlap_args = PeoOverlapArgs(
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=experts.num_device_sms,
            recv_num_sms=experts.num_device_sms if round_id == 0 else experts.num_deepep_sms,
            hook_use_comm_stream=False,
        )
        state = experts.dispatcher.dispatch_a_peo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            peo_overlap_args=peo_overlap_args,
        )
        states.append(state)

    # dispatch recv and GEMM
    for round_id in range(experts.num_rounds):
        dispatch_output = experts.dispatcher.dispatch_b_peo(
            inner_state=states[round_id],
        )
        GEMM_STREAM.wait_stream(current_stream)
        with torch.cuda.stream(GEMM_STREAM):
            # moe_hidden_state = torch.rand(20, 256 * experts.num_ranks, experts.hidden_size, dtype=torch.bfloat16,
            #                               device=hidden_states.device)
            # moe_hidden_states.append((moe_hidden_state, dispatch_output.topk_ids, dispatch_output.topk_weights))
            moe_hidden_state = experts.run_moe_core(gen_new_dispatch_output(experts, dispatch_output, round_id, False))
            moe_hidden_states.append(moe_hidden_state)
            gemm_done_event = torch.cuda.Event()
            GEMM_STREAM.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    for round_id in range(experts.num_rounds):
        peo_overlap_args = PeoOverlapArgs(
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms,
            recv_num_sms=experts.num_device_sms,
            hook_use_comm_stream=False,
        )
        current_stream.wait_event(gemm_done_events[round_id])
        combine_state = experts.dispatcher.combine_a_peo(
            combine_input=moe_hidden_states[round_id],
            peo_overlap_args=peo_overlap_args,
        )

    # combine recv
    combined_x = experts.dispatcher.combine_b_peo(inner_state=combine_state)
    current_stream.wait_stream(GEMM_STREAM)
    return combined_x


def forward_overlap_2_3(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
):
    states = list()
    gemm_done_events = list()
    moe_hidden_states = list()

    global dispatch_output, combine_state
    hook_use_default_stream = experts.overlap_method == 2
    # dispatch and GEMM
    for round_id in range(experts.num_rounds):
        # dispatch send
        num_experts_per_round = experts.num_experts // experts.num_ranks // experts.num_rounds
        start_idx = num_experts_per_round * round_id
        end_idx = start_idx + num_experts_per_round
        peo_overlap_args = PeoOverlapArgs(
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=experts.num_device_sms,
            recv_num_sms=experts.num_device_sms if round_id == 0 else experts.num_deepep_sms,
            hook_use_comm_stream=not hook_use_default_stream,
        )
        # logging.info(f"round_id: {round_id}, "
        #              f"peo_overlap_args: {peo_overlap_args}, "
        #              f"hidden_states: {hidden_states}, "
        #              f"topk_output: {topk_output}")
        state = experts.dispatcher.dispatch_a_peo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            peo_overlap_args=peo_overlap_args,
        )

        # dispatch recv
        if hook_use_default_stream:
            dispatch_output = experts.dispatcher.dispatch_b_peo(
                inner_state=state,
            )
        else:
            experts.comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(experts.comm_stream):
                dispatch_output = experts.dispatcher.dispatch_b_peo(
                    inner_state=state,
                )
            GEMM_STREAM.wait_stream(experts.comm_stream)

        # GEMM
        GEMM_STREAM.wait_stream(current_stream)
        with torch.cuda.stream(GEMM_STREAM):
            moe_hidden_state = experts.run_moe_core(gen_new_dispatch_output(experts, dispatch_output, round_id, False))
            moe_hidden_states.append(moe_hidden_state)
            gemm_done_event = torch.cuda.Event()
            GEMM_STREAM.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    for round_id in range(experts.num_rounds):
        peo_overlap_args = PeoOverlapArgs(
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms,
            recv_num_sms=experts.num_device_sms,
            hook_use_comm_stream=False,
            is_x_in_round=True,
        )
        current_stream.wait_event(gemm_done_events[round_id])
        if not hook_use_default_stream:
            current_stream.wait_stream(experts.comm_stream)
        combine_state = experts.dispatcher.combine_a_peo(
            combine_input=moe_hidden_states[round_id],
            peo_overlap_args=peo_overlap_args,
        )

    # combine recv
    if hook_use_default_stream:
        combined_x = experts.dispatcher.combine_b_peo(inner_state=combine_state)
    else:
        experts.comm_stream.wait_stream(current_stream)
        with torch.cuda.stream(experts.comm_stream):
            combined_x = experts.dispatcher.combine_b_peo(inner_state=combine_state)
        current_stream.wait_stream(experts.comm_stream)

    current_stream.wait_stream(GEMM_STREAM)
    return combined_x

def forward_overlap_4(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
):
    gemm_done_events = list()
    moe_hidden_states = list()
    # dispatch
    experts.comm_stream.wait_stream(current_stream)
    with torch.cuda.stream(experts.comm_stream):
        dispatch_output = experts.dispatch(hidden_states, topk_output)

        # current_stream.wait_stream(comm_stream)
        for round_id in range(experts.num_rounds):
            moe_hidden_state = experts.run_moe_core(gen_new_dispatch_output(experts, dispatch_output, round_id, round_id != (experts.num_rounds - 1)))
            moe_hidden_states.append(moe_hidden_state)
            gemm_done_event = torch.cuda.Event()
            experts.comm_stream.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    with torch.cuda.stream(current_stream):
        for round_id in range(experts.num_rounds):
            peo_overlap_args = PeoOverlapArgs(
                use_expert_overlap=True,
                num_rounds=experts.num_rounds,
                round_id=round_id,
                send_num_sms=experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms,
                recv_num_sms=experts.num_device_sms,
                hook_use_comm_stream=False,
            )
            combine_state = experts.dispatcher.combine_a_peo(
                combine_input=moe_hidden_states[round_id],
                peo_overlap_args=peo_overlap_args,
            )

        # combine recv
        combined_x = experts.dispatcher.combine_b_peo(inner_state=combine_state)

    current_stream.wait_stream(experts.comm_stream)
    return combined_x


def execute_peo(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
):
    current_stream = torch.cuda.current_stream()

    with torch.cuda.stream(current_stream):
        if experts.overlap_method == 1:
            return forward_overlap_1(
                experts,
                hidden_states,
                topk_output,
                current_stream,
            )
        elif experts.overlap_method == 2 or experts.overlap_method == 3:
            return forward_overlap_2_3(
                experts,
                hidden_states,
                topk_output,
                current_stream,
            )
        elif experts.overlap_method == 4:
            return forward_overlap_4(
                experts,
                hidden_states,
                topk_output,
                current_stream,
            )
        else:
            raise ValueError(f"Invalid overlap_method: {experts.overlap_method}")


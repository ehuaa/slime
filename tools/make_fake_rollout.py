"""Synthesise a rollout .pt for --load-debug-rollout-data.

Benchmarks Megatron parallelism (TP/PP/CP, recompute, max-tokens-per-gpu) without
paying ~51 min for a real sglang rollout. Only token *shapes* matter for that, so ids
and rewards are random -- meaningless for learning, but the memory/compute profile
matches a real batch.

Three contracts a real rollout satisfies implicitly, learned the hard way:
  * group_id must be None -- default rollouts emit ONE sample per training group
    (rollout.py:673 falls back to sample.index). build_dp_schedule counts GROUPS, so
    grouping siblings makes num_groups < global_batch_size and it asserts.
  * rollout_log_probs must be a per-token list -- loss.py:616 does
    `xs = log_probs or rollout_log_probs or values` then iterates it; None crashes
    compute_advantages_and_returns when kl_coef == 0.
  * len(rollout_log_probs) must equal response_length.

Default lengths match dapo-17k + Qwen3.8-27B at a 65536 cap: median 5552, mean 12966,
3.9% truncated.
"""

import argparse
import random

import torch


def build(args, rng):
    n_trunc = int(args.num_samples * args.truncated_ratio)
    samples = []
    for i in range(args.num_samples):
        if i < n_trunc:
            resp_len, status = args.max_response_len, "truncated"
        else:
            resp_len = int(rng.lognormvariate(args.log_mu, args.log_sigma))
            resp_len = max(args.min_response_len, min(resp_len, args.max_response_len - 1))
            status = "completed"
        prompt_len = rng.randint(args.prompt_len_min, args.prompt_len_max)
        samples.append({
            "group_index": i // args.n_samples_per_prompt,
            "index": i,
            "group_id": None,
            "prompt": "",
            "tokens": [rng.randrange(args.vocab_size) for _ in range(prompt_len + resp_len)],
            "multimodal_inputs": None,
            "multimodal_train_inputs": None,
            "response": "",
            "response_length": resp_len,
            "label": "0",
            "reward": float(rng.random() < args.positive_rate),
            "loss_mask": None,
            "weight_versions": [],
            "rollout_log_probs": [round(rng.gauss(-0.66, 0.5), 4) for _ in range(resp_len)],
            "rollout_routed_experts": None,
            "remove_sample": False,
            "teacher_log_probs": None,
            "status": status,
            "metadata": {},
            "generate_function_path": None,
            "train_metadata": None,
            "session_id": None,
            "non_generation_time": 0.0,
            "spec_info": {"spec_accept_token_num": 0, "spec_draft_token_num": 0,
                          "spec_verify_ct": 0, "completion_token_num": resp_len},
            "prefix_cache_info": {"cached_tokens": 0, "total_prompt_tokens": prompt_len},
        })
    rng.shuffle(samples)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=2048)
    p.add_argument("--n-samples-per-prompt", type=int, default=8)
    p.add_argument("--max-response-len", type=int, default=65536)
    p.add_argument("--min-response-len", type=int, default=313)
    p.add_argument("--truncated-ratio", type=float, default=0.039)
    p.add_argument("--prompt-len-min", type=int, default=180)
    p.add_argument("--prompt-len-max", type=int, default=320)
    p.add_argument("--vocab-size", type=int, default=248320)
    p.add_argument("--positive-rate", type=float, default=0.72)
    p.add_argument("--log-mu", type=float, default=8.622)     # exp(mu) = median ~5552
    p.add_argument("--log-sigma", type=float, default=1.303)  # -> mean ~12966
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    samples = build(args, random.Random(args.seed))
    torch.save({"samples": samples}, args.out)

    lens = sorted(s["response_length"] for s in samples)
    n = len(lens)
    n_tr = sum(1 for s in samples if s["status"] == "truncated")
    print(f"wrote {args.out}")
    print(f"  samples       {n}   groups {n} (group_id=None -> one per sample)")
    print(f"  response_len  mean={sum(lens)/n:.0f} median={lens[n//2]} min={lens[0]} max={lens[-1]}")
    print(f"  truncated     {n_tr}/{n} = {n_tr/n:.1%}")
    print(f"  total tokens  {sum(len(s['tokens']) for s in samples):,}")


if __name__ == "__main__":
    main()

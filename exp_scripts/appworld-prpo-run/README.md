# AppWorld on rLLM: synchronous PRPO / GRPO with Qwen3.5-4B

Experiment-layer code only: rLLM, verl and vLLM are used unmodified through rLLM's public `AgentTrainer` / `@rllm.rollout` / `@rllm.evaluator` extension points. Training is synchronous on-policy (`rllm.async_training.enable=false`); rLLM's fully async trainer is not used.

## Files

| File | Role |
|---|---|
| `appworld_react/` | The AppWorld react_code environment, byte for byte the same as Reef's SAO line (`reef exp_scripts/appworld/appworld_react`). Check with `sha256sum -c SHA256SUMS` inside it; `train_appworld.py` refuses to start otherwise. |
| `appworld_flow.py` | `@rllm.rollout(name="appworld-react")` flow and `appworld_evaluator`. Wraps `run_episode` with a gateway-backed `ChatModel` (`GatewayChat`), one AppWorld server per episode on an flock-leased port, and the world output deleted after grading. |
| `register_appworld.py` | Registers dataset `appworld` (splits train 90 / dev 57 / test_normal 168 / test_challenge 417) in the run's `RLLM_HOME`. Rows carry only the task id (+ instruction for logs): the prompt comes from the AppWorld server. |
| `train_appworld.py` | Hydra entry: loads train/dev, runs the preflight, hands everything to `AgentTrainer`. |
| `train_appworld.sh` | Launcher, `smoke` or `train`, flags listed in its header. Refuses a Qwen3.5 model without `--gdn-varlen`. |
| `appworld_gdn_varlen.py` | Qwen3.5 Gated DeltaNet packing fix, loaded in the verl workers by `--gdn-varlen` (see below). |
| `launch_grpo_dot16.sh` | One-command GRPO run on .16 (2 GPUs, fix on): host/GPU/port checks, memory-cap watchdog, detached start. `--check` only checks. |
| `adapter_follow.sh` | Extracts each new checkpoint's LoRA adapter as the run writes it (CPU only), for evaluation elsewhere. |
| `eval_appworld.py`, `eval_lane.sh`, `serve_eval_vllm.sh`, `eval_queue_plan.py` | Greedy evaluation lanes with the LoRA mounted unmerged (.16/.29). |
| `eval_gpu_pool.py`, `eval_follow.sh` | 8-GPU data-parallel evaluation on .36 (shared task claims), and its follower that keeps adding new checkpoints. |
| `summarize_eval.py` | Average score, TGC and SGC per model. |
| `appworld_mem_cap.sh` | Caps every AppWorld server at 16 GB address space (a runaway server once took the trainer down). |
| `make_eosfix_model_dir.sh` | Builds the Qwen3.5-4B model dir with `generation_config.json` naming `<\|im_end\|>` (248046). |
| `test_appworld_flow.py` | CPU unit tests (fake client, fake server): `../../.venv/bin/python -m unittest -v test_appworld_flow`. |

## Protocol (fixed by `appworld_react`, shared with Reef SAO)

- Prompt: grpo_vanilla `SYSTEM_PROMPT` + task instruction + AppWorld's official react_code prompt; one tool, `execute_code`.
- Budgets: 50 executed tool calls, 24576 generated tokens per episode (per call `max_completion_tokens` = what is left, clamped to the context room).
- Sampling T=1.0, top_p=1.0, no top_k; thinking off per request (`chat_template_kwargs.enable_thinking=false`), no reasoning parser, tool parser `qwen3_coder`.
- Reward: the corrected partial score (passed tests / tests a do-nothing agent would fail). `is_correct` = the official TGC (all tests pass). Signals `tgc` and `partial_score`.
- Context: 32768 by default, the same as Reef's SAO AppWorld driver (`SAO_CONTEXT_TOKENS`). A conversation that outgrows it ends as `context_overflow` and is graded as it stands, as in Reef.
- A grader failure (`reward is None`) is never scored: the flow raises, rLLM retries the rollout, and repeated failure stops the run. None of the 90 + 135 + 90 Qwen3.5 episodes measured on 2026-09-28 was ungradable.

## Training defaults

| Setting | PRPO (default) | GRPO (`--algorithm grpo`) |
|---|---|---|
| tasks x rollouts per step | 15 x 1 | 9 x 8 |
| advantage | batch-normalized (`prpo`) | group mean/std (`grpo`) |
| steps per epoch (90 train tasks) | 6 | 10 |

Shared: LoRA r32/a32 (vision tower excluded), merge into vLLM, lr 1e-5, KL 0.001 low_var_kl, entropy 0.002, clip 0.2/0.28, fsdp2 with param+optimizer offload, fused kernels, gradient checkpointing, one vLLM engine per GPU (TP 1, `max_num_seqs` 64), `max_prompt_length` = context and `max_response_length` = context - 3584 (every train/dev first prompt is 3682-3832 tokens), token-packing ceilings = context. The prompt bound is the full context because a turn that does not extend the previous token prefix makes rLLM open a new row whose prompt is the whole conversation so far, and `transform.py` left-truncates prompts past `max_prompt_length` silently; padding is stripped before compute, so the wide bound costs nothing. A checkpoint every epoch; no in-loop validation (evaluate checkpoints offline).

Measured Qwen3.5-4B trajectory length on train90 at T=1.0 (prompt + all turns + tool outputs, last call): p50 14532, p90 26905, p95 30088, max 39732 tokens; 86/90 fit 32768, 90/90 fit 40960. `--context 40960` keeps every episode whole if memory allows.

## Qwen3.5 packing fix (`--gdn-varlen`, required)

With `use_remove_padding` verl packs each micro-batch into one (1, total_nnz) row. The full-attention layers stay per sequence, but HF's `Qwen3_5GatedDeltaNet` (transformers 5.5.4) has no sequence boundaries: its causal conv1d and recurrent state run across rows, so every row after the first in a micro-batch gets its log-probs and gradient from the previous row's state. Measured 2026-10-09 on Qwen3.5-4B: assistant-token log-probs off by 0.08 nats on average (bf16 noise floor 0.007); rollout/actor `offpolicy/kl` 0.015 from step 1, against 0.0004-0.0008 with the fix. The PRPO run of 2026-10-08 and the .16 GRPO run of 2026-10-09 trained under the leak; their checkpoints are pre-fix. `appworld_gdn_varlen.py` splits each GDN call where the text position ids restart and runs the unchanged layer per sequence (packed vs unpacked agree to 1e-5; a single row is untouched). Dense models (Qwen3, FinQA's 2507) and Reef's slime/Megatron Qwen3.5 (which passes `cu_seqlens`) are not affected.

## Measured before training (2026-10-08, .29, plain vLLM 0.22.1)

16 train episodes through `GatewayChat` (12:41-12:56 EDT): every call returned content with thinking off (0 `<think>`), tool calls parsed with `qwen3_coder`, one Step per call, every episode graded (0 grader failures), no world output left behind. Token-prefix growth held on ~96% of turn transitions; 10 breaks in ~370 transitions, 6 of 16 episodes with at least one. In 8 of the 10 the text is identical and only the token ids differ: at T=1.0 the model sometimes samples a non-canonical BPE split, and the next prompt re-tokenizes the same text canonically. The other 2 are a tool call written inside long prose that the template re-renders. Each break gives that episode one more row (handled correctly by the full-context prompt bound above); a broken episode therefore weighs a little more under `seq-mean-token-mean`. Watch rLLM's merge metrics (rows per trajectory) in training.

## Running

```bash
# once per host: model dir on the host's local disk (.29 already has /mnt/disk1t/models/...-eosfix)
./make_eosfix_model_dir.sh /works/yanan/models/qwen35-4b-851bf6e8-eosfix        # .16

# CPU tests
../../.venv/bin/python -m unittest -v test_appworld_flow

# two training steps, then the real run (Qwen3.5 needs --gdn-varlen)
./train_appworld.sh smoke --gpus 0,1 --gdn-varlen
./train_appworld.sh train --gpus 0,1 --gdn-varlen [--algorithm grpo] [--context 40960] [--disable-nccl-p2p]

# .16 GRPO in one command (checks first with --check)
ssh 10.225.68.16 /home/yanan/agents/rllm/exp_scripts/appworld-prpo-run/launch_grpo_dot16.sh
```

Run state goes to `runs/<experiment>/` (launch record, `rollouts-<stamp>/episodes.jsonl` with one line per episode, AppWorld server logs); `runs/<experiment>/checkpoints` is a symlink to the host's local disk (`/works/yanan/...` on .16, `/mnt/disk1t/...` on .29). Logs in `logs/`.

## Environment requirements

- rllm venv (`/home/yanan/agents/rllm/.venv`): transformers 5.5.4, vLLM 0.22.1, verl 0.8.0. It has no `appworld`: the AppWorld servers run from `APPWORLD_VENV` (`gitlab/tail/appworld/.venv`).
- `fla` (flash-linear-attention 0.5.2) at `/home/yanan/.fla-kernels` on the training host, put on `PYTHONPATH` by the launcher. Without it Qwen3.5's Gated DeltaNet falls back to pure torch and step 2 OOMs. Present on .16 and .29.
- `causal_conv1d` is not installed; transformers uses its torch path for the short convolution (inference A/B: 0-2% difference).
- AppWorld data: `/home/yanan/appworld-root` when present (.16, local), else the NFS install `gitlab/tail/appworld`.
- RAM: one `appworld serve environment` process per concurrent episode (`--max-episodes`, default tasks x rollouts per step, at most 64), on top of verl's offloaded optimizer state.

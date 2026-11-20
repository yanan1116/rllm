#!/usr/bin/env bash
# AppWorld react_code on rLLM: synchronous PRPO (default) or GRPO, Qwen3.5-4B, 2 GPUs per run.
#
#   train_appworld.sh {smoke|train} [flags] [-- hydra overrides...]
#
# Flags (all optional):
#   --algorithm prpo|grpo   prpo: 15 tasks x 1 rollout, batch-normalized advantage (default)
#                           grpo: 9 tasks x 8 rollouts, group-normalized advantage
#                           Both cover the 90 train tasks exactly once per epoch.
#   --gpus 0,1              CUDA_VISIBLE_DEVICES; the run uses all listed GPUs
#   --context N             serving context = longest token row (default 32768, as Reef's SAO AppWorld driver)
#   --epochs N              train mode only (default 10); a checkpoint every epoch
#   --batch-size N          tasks per step (default by algorithm)
#   --gpu-mem-util F        vLLM share of each GPU (default 0.35)
#   --max-episodes N        concurrent AppWorld episodes / servers (default tasks x rollouts per step, max 64)
#   --max-num-seqs N        vLLM max_num_seqs per engine (default --max-episodes: one decode per episode;
#                           must fit the Mamba cache blocks gpu-mem-util leaves, which vLLM checks at start)
#   --model-path DIR        default <host-local>/models/qwen35-4b-851bf6e8-eosfix (make_eosfix_model_dir.sh)
#   --appworld-root DIR     default /home/yanan/appworld-root, else the NFS install
#   --ckpt-root DIR         host-local checkpoint root (default <host-local>/rllm-appworld-checkpoints)
#   --experiment-name NAME
#   --disable-nccl-p2p      NCCL_P2P_DISABLE=1 (RTX 6000 Ada pairs whose PCIe P2P hangs the first collective)
#   --gdn-varlen            run Qwen3.5's Gated DeltaNet per packed sequence in the actor/ref workers
#                           (appworld_gdn_varlen.py via actor_rollout_ref.model.external_lib). verl packs
#                           a micro-batch into one row and HF's GDN layer has no sequence boundaries, so
#                           without it every row after the first is computed from the previous row's
#                           state (measured 2026-10-09). Required for a Qwen3.5 (model_type qwen3_5) model.
#   --lr F                  actor learning rate (default 1e-5)
#   --save-freq N           train mode: a full checkpoint every N steps (default: every epoch);
#                           ~9 GB per save for Qwen3.5-4B, on the host's local disk
#
# Fixed protocol (appworld_react, shared with Reef SAO): 50 tool calls, 24576 generated tokens per
# episode, T=1.0 / top_p=1.0, thinking off, reward = corrected partial score, TGC as is_correct.
# rLLM's fully async trainer is never used: rllm.async_training.enable=false.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
RLLM_DIR="$(cd "$RUN_DIR/../.." && pwd)"
VENV="$RLLM_DIR/.venv"

die() { echo "[appworld] FATAL: $*" >&2; exit 1; }

MODE="${1:-}"; shift || true
[[ "$MODE" == smoke || "$MODE" == train ]] || die "usage: $0 {smoke|train} [flags] [-- hydra overrides]"

ALGORITHM=prpo; GPUS=0,1; CONTEXT=32768; EPOCHS=10; BATCH_SIZE=""; GPU_MEM_UTIL=0.35; MAX_EPISODES=""
MAX_NUM_SEQS=""; MODEL_PATH=""; APPWORLD_ROOT_ARG=""; CKPT_ROOT=""; EXPERIMENT_NAME=""; DISABLE_NCCL_P2P=0
LR=1e-5; SAVE_FREQ_ARG=""; GDN_VARLEN=0
FLA_DIR=/home/yanan/.fla-kernels
HYDRA_EXTRA=()
while (($#)); do
  case "$1" in
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --appworld-root) APPWORLD_ROOT_ARG="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --disable-nccl-p2p) DISABLE_NCCL_P2P=1; shift ;;
    --gdn-varlen) GDN_VARLEN=1; shift ;;
    --lr) LR="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ_ARG="$2"; shift 2 ;;
    --) shift; HYDRA_EXTRA=("$@"); break ;;
    *) die "unknown flag $1 (hydra overrides go after --)" ;;
  esac
done

# ---- host-local disk: checkpoints and the model dir never live on NFS ~/agents -------------------
HOST_LOCAL=""
for candidate in /works/yanan /mnt/disk1t; do
  [[ -d "$candidate" && -w "$candidate" ]] && { HOST_LOCAL="$candidate"; break; }
done
[[ -n "$HOST_LOCAL" || ( -n "$MODEL_PATH" && -n "$CKPT_ROOT" ) ]] \
  || die "no host-local disk found (/works/yanan, /mnt/disk1t); pass --model-path and --ckpt-root"
MODEL_PATH="${MODEL_PATH:-$HOST_LOCAL/models/qwen35-4b-851bf6e8-eosfix}"
CKPT_ROOT="${CKPT_ROOT:-$HOST_LOCAL/rllm-appworld-checkpoints}"
case "$(realpath -m "$CKPT_ROOT")" in /home/yanan/agents/*) die "--ckpt-root $CKPT_ROOT is on NFS ~/agents; use the host's local disk" ;; esac

if [[ -z "$APPWORLD_ROOT_ARG" ]]; then
  if [[ -d /home/yanan/appworld-root/data/tasks ]]; then APPWORLD_ROOT_ARG=/home/yanan/appworld-root
  else APPWORLD_ROOT_ARG=/home/yanan/agents/gitlab/tail/appworld; fi
fi

# ---- algorithm --------------------------------------------------------------------------------
case "$ALGORITHM" in
  prpo)
    BATCH_SIZE="${BATCH_SIZE:-15}"; ROLLOUT_N=1
    ALGORITHM_OVERRIDES=(
      algorithm.adv_estimator=prpo
      rllm.algorithm.adv_estimator=prpo
      rllm.algorithm.rollout_correction.tis_mode=null
      rllm.rejection_sample.min_trajs_per_group=1
    ) ;;
  grpo)
    BATCH_SIZE="${BATCH_SIZE:-9}"; ROLLOUT_N=8
    ALGORITHM_OVERRIDES=(
      algorithm.adv_estimator=grpo
      rllm.algorithm.adv_estimator=grpo
      algorithm.norm_adv_by_std_in_grpo=true
    ) ;;
  *) die "--algorithm must be prpo or grpo, got $ALGORITHM" ;;
esac
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -gt 4096 ]] || die "--context must be an integer > 4096"
# A row's prompt is its segment's first prompt: normally the task prompt (3682-3832 tokens), but when
# a turn does not extend the previous token prefix rLLM starts a new row whose prompt is the whole
# conversation so far, and transform.py left-truncates prompts past max_prompt_length without a word.
# So the prompt bound is the full context, and the response bound is what the shortest first prompt
# leaves. Padding is stripped before compute (use_remove_padding), so the wide bound costs nothing.
MAX_PROMPT=$CONTEXT
MAX_RESPONSE=$((CONTEXT - 3584))

case "$MODE" in
  smoke)
    [[ "$ALGORITHM" == prpo ]] && BATCH_SIZE=4 || BATCH_SIZE=2
    [[ "$ALGORITHM" == grpo ]] && ROLLOUT_N=2
    TOTAL_EPOCHS=1; TOTAL_STEPS=2; SAVE_FREQ=-1; RESUME_MODE=disable
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-appworld-${ALGORITHM}-smoke-$(hostname -s)}" ;;
  train)
    (( 90 % BATCH_SIZE == 0 )) || die "--batch-size $BATCH_SIZE does not divide the 90 train tasks; a ragged epoch drops tasks"
    [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
    STEPS_PER_EPOCH=$((90 / BATCH_SIZE))
    TOTAL_EPOCHS=$EPOCHS; TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS)); SAVE_FREQ=$STEPS_PER_EPOCH; RESUME_MODE=auto
    if [[ -n "$SAVE_FREQ_ARG" ]]; then
      [[ "$SAVE_FREQ_ARG" =~ ^[1-9][0-9]*$ ]] || die "--save-freq must be a positive integer"
      SAVE_FREQ=$SAVE_FREQ_ARG
    fi
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-appworld-${ALGORITHM}-qwen35-4b-b${BATCH_SIZE}n${ROLLOUT_N}-c${CONTEXT}}" ;;
esac
EPISODES_PER_STEP=$((BATCH_SIZE * ROLLOUT_N))
MAX_EPISODES="${MAX_EPISODES:-$(( EPISODES_PER_STEP < 64 ? EPISODES_PER_STEP : 64 ))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$MAX_EPISODES}"

# ---- preflight: fail before Ray starts ----------------------------------------------------------
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || die "model dir $MODEL_PATH missing; create it with $RUN_DIR/make_eosfix_model_dir.sh $MODEL_PATH"
grep -q 248046 "$MODEL_PATH/generation_config.json" 2>/dev/null \
  || die "$MODEL_PATH/generation_config.json lacks eos 248046 (<|im_end|>): Qwen3.5-4B would never stop a turn. Use make_eosfix_model_dir.sh"
MODEL_TYPE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_type", ""))' "$MODEL_PATH/config.json")
if [[ "$MODEL_TYPE" == qwen3_5 ]] && (( ! GDN_VARLEN )); then
  die "model_type qwen3_5 without --gdn-varlen: verl packs each micro-batch into one row and HF's Gated DeltaNet carries state across rows (assistant-token logprobs off by 0.08 nats vs a 0.007 noise floor; rollout/actor offpolicy KL 0.015 vs 0.0004 fixed). Add --gdn-varlen"
fi
(( ! GDN_VARLEN )) || [[ -f "$RUN_DIR/appworld_gdn_varlen.py" ]] || die "--gdn-varlen: $RUN_DIR/appworld_gdn_varlen.py missing"
[[ -d "$FLA_DIR/fla" ]] || die "$FLA_DIR/fla missing: Qwen3.5's Gated DeltaNet falls back to pure torch and OOMs at step 2. Copy it from .16:/home/yanan/.fla-kernels"
[[ -x "$APPWORLD_ROOT_ARG/.venv/bin/appworld" || -x /home/yanan/agents/gitlab/tail/appworld/.venv/bin/appworld ]] || die "no appworld CLI"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

STAMP=$(date +%Y%m%dT%H%M%S)
STATE_DIR="$RUN_DIR/runs/$EXPERIMENT_NAME"
mkdir -p "$STATE_DIR" "$RUN_DIR/logs" "$CKPT_ROOT/$EXPERIMENT_NAME"
ln -sfn "$CKPT_ROOT/$EXPERIMENT_NAME" "$STATE_DIR/checkpoints"

# ---- environment --------------------------------------------------------------------------------
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export RLLM_HOME="$RUN_DIR/.rllm_home"
export PYTHONPATH="$FLA_DIR:$RUN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# vllm._C links libcudart.so.13, which the venv bundles off the loader path (as finqa-grpo-run/env.sh).
export LD_LIBRARY_PATH="$VENV/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 RAY_DEDUP_LOGS=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export APPWORLD_ROOT="$APPWORLD_ROOT_ARG"
export APPWORLD_VENV=/home/yanan/agents/gitlab/tail/appworld/.venv
export APPWORLD_RUN_DIR="$STATE_DIR/rollouts-$STAMP"
export APPWORLD_MAX_EPISODES="$MAX_EPISODES"
export APPWORLD_CONTEXT_TOKENS="$CONTEXT"
export APPWORLD_TOKENIZER="$MODEL_PATH"
if (( DISABLE_NCCL_P2P )); then export NCCL_P2P_DISABLE=1; fi

[[ -f "$RLLM_HOME/datasets/appworld/train.parquet" && -f "$RLLM_HOME/datasets/appworld/dev.parquet" ]] \
  || python "$RUN_DIR/register_appworld.py" --root "$APPWORLD_ROOT" --splits train dev

{
  echo "started_at=$(date -Is) host=$(hostname -s) mode=$MODE algorithm=$ALGORITHM gpus=$GPUS"
  echo "batch=$BATCH_SIZE rollout_n=$ROLLOUT_N steps=$TOTAL_STEPS context=$CONTEXT max_episodes=$MAX_EPISODES max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEM_UTIL lr=$LR save_freq=$SAVE_FREQ"
  echo "model=$MODEL_PATH model_type=$MODEL_TYPE gdn_varlen=$GDN_VARLEN appworld_root=$APPWORLD_ROOT ckpt=$CKPT_ROOT/$EXPERIMENT_NAME"
  (cd "$RUN_DIR" && sha256sum appworld_flow.py train_appworld.py train_appworld.sh register_appworld.py appworld_gdn_varlen.py)
  (cd "$RUN_DIR/appworld_react" && sha256sum -c SHA256SUMS --quiet && echo "appworld_react=SHA256SUMS ok")
} | tee "$STATE_DIR/launch-$STAMP.txt"

# Overrides as an array: a '#' inside a backslash-continued command would silently drop the rest.
ARGS=(
  rllm/backend=verl
  "${ALGORITHM_OVERRIDES[@]}"
  rllm.async_training.enable=false
  rllm.algorithm.use_rllm=true
  +model.name="$MODEL_PATH"
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.lora.merge=True
  # LoRA on the language model only; the vision tower never sees an AppWorld token.
  "actor_rollout_ref.model.exclude_modules='.*visual.*'"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  # Qwen3.5's 248320-entry vocabulary makes materialised fp32 logits the OOM at these lengths.
  actor_rollout_ref.model.use_fused_kernels=True
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.hybrid_engine=True
  actor_rollout_ref.actor.optim.lr="$LR"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.002
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.entropy_checkpointing=True
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE"
  # Packing ceilings: verl never splits a sequence, so each must hold the longest token row.
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$CONTEXT"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$CONTEXT"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$CONTEXT"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
  actor_rollout_ref.rollout.enforce_eager=False
  # One engine per GPU: a 4B model needs no tensor parallel, and TP would take the PCIe P2P path.
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  actor_rollout_ref.rollout.max_model_len="$CONTEXT"
  # Gated DeltaNet: one Mamba cache block per running sequence; verl's 1024 exceeds the pool. An
  # episode has at most one call in flight, so the episode count bounds what one engine can receive.
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
  # qwen3_coder is Qwen3.5's tool format (hermes silently drops every call). No reasoning parser:
  # thinking is off per request, and the two together return content=None.
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder
  rllm.rollout.n="$ROLLOUT_N"
  rllm.rollout.n_val=1
  rllm.rollout.train.temperature=1.0
  rllm.rollout.train.top_p=1.0
  rllm.rollout.val.temperature=1.0
  rllm.rollout.val.top_p=1.0
  rllm.data.train_batch_size="$BATCH_SIZE"
  data.val_batch_size=19
  rllm.data.max_prompt_length="$MAX_PROMPT"
  rllm.data.max_response_length="$MAX_RESPONSE"
  rllm.trainer.logger="['console']"
  rllm.trainer.project_name=appworld-rllm
  rllm.trainer.experiment_name="$EXPERIMENT_NAME"
  rllm.trainer.total_epochs="$TOTAL_EPOCHS"
  rllm.trainer.total_batches="$TOTAL_STEPS"
  rllm.trainer.test_freq=-1
  rllm.trainer.save_freq="$SAVE_FREQ"
  rllm.trainer.val_before_train=false
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.nnodes=1
  trainer.n_gpus_per_node="$NGPU"
  trainer.resume_mode="$RESUME_MODE"
  trainer.default_hdfs_dir=null
  trainer.default_local_dir="$STATE_DIR/checkpoints"
)
(( ${#ARGS[@]} >= 60 )) || die "only ${#ARGS[@]} overrides built"
# Imported by verl inside every actor/ref worker (HFModelConfig); the run dir is on PYTHONPATH.
(( ! GDN_VARLEN )) || ARGS+=(actor_rollout_ref.model.external_lib=appworld_gdn_varlen)

cd "$RUN_DIR"
LOG="$RUN_DIR/logs/$EXPERIMENT_NAME-$STAMP.log"
python -u "$RUN_DIR/train_appworld.py" "${ARGS[@]}" "${HYDRA_EXTRA[@]}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"

#!/usr/bin/env bash
# Fan out the 12 GS-control folds (4 inputs x seeds 42/1/7) across free lambda1 V100s.
# One GPU per worker, held via gpu_reserve acquire/release (long-worker pattern); the gate
# is respected, so setting CUDA_VISIBLE_DEVICES to the acquired index is correct here.
PROJ=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
OUT=$PROJ/results/control_gs_native/matrix
STAGE=$OUT/stage; LOG=$OUT/logs; mkdir -p "$STAGE" "$LOG"
source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2

JOBS=()
for name in P78061_free P78061_put Q9KZC7_free Q9KZC7_put; do
  for seed in 42 1 7; do JOBS+=("$name $seed"); done
done
mapfile -t IDXS < <("$GR" list 2>/dev/null | awk '/^lambda1 /{print $2}')
N=${#IDXS[@]}
if [ "$N" -lt 1 ]; then echo "no free lambda1 GPUs"; exit 2; fi
echo "free lambda1 GPUs: $N (${IDXS[*]}); ${#JOBS[@]} jobs"

worker(){
  local idx=$1; shift
  "$GR" acquire "$idx" --pid $$ || { echo "[worker $idx] acquire failed"; return 1; }
  trap "$GR release $idx" EXIT
  for j in "$@"; do
    set -- $j; local name=$1 seed=$2
    CUDA_VISIBLE_DEVICES=$idx OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      boltz predict "$OUT/inputs/$name.yaml" --out_dir "$STAGE/${name}_s${seed}" \
      --diffusion_samples 50 --max_parallel_samples 5 --seed "$seed" \
      --recycling_steps 3 --sampling_steps 200 --output_format pdb \
      --accelerator gpu --devices 1 --num_workers 2 --override \
      > "$LOG/${name}_s${seed}.log" 2>&1
    echo "[done] $name s$seed rc=$? (idx $idx)"
  done
  "$GR" release "$idx"; trap - EXIT
}

# round-robin assign jobs to N workers
declare -a BUCKET
for i in "${!JOBS[@]}"; do
  w=$(( i % N ))
  BUCKET[$w]="${BUCKET[$w]}|${JOBS[$i]}"
done
for w in $(seq 0 $((N-1))); do
  IFS='|' read -ra mine <<< "${BUCKET[$w]}"
  jobs_clean=(); for x in "${mine[@]}"; do [ -n "$x" ] && jobs_clean+=("$x"); done
  [ ${#jobs_clean[@]} -eq 0 ] && continue
  worker "${IDXS[$w]}" "${jobs_clean[@]}" &
done
wait
echo "[matrix] ALL DONE"

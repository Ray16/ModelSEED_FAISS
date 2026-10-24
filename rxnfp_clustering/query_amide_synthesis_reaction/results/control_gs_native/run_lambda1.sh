#!/usr/bin/env bash
# Run the two GS native-substrate controls on lambda1 V100s with stock Boltz-2.
# One task per reserved GPU (explicit indices), threads capped for shared-node courtesy.
PROJ=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
IN=$PROJ/results/control_gs_native/inputs
STAGE=$PROJ/results/control_gs_native/stage
LOG=$PROJ/results/control_gs_native/logs
mkdir -p "$STAGE" "$LOG"

source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2

run_one(){
  local name=$1 idx=$2
  "$GR" run "$idx" -- env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
     boltz predict "$IN/$name.yaml" --out_dir "$STAGE/$name" \
     --diffusion_samples 50 --max_parallel_samples 5 --seed 42 \
     --recycling_steps 3 --sampling_steps 200 \
     --output_format pdb --accelerator gpu --devices 1 --num_workers 2 --override \
     > "$LOG/${name}_boltz.log" 2>&1
  echo "[run] $name (idx $idx) rc=$?"
}

mapfile -t IDXS < <("$GR" list 2>/dev/null | awk '/^lambda1 /{print $2}')
if [ "${#IDXS[@]}" -lt 2 ]; then echo "[run] fewer than 2 free lambda1 GPUs: ${IDXS[*]:-none}"; exit 2; fi
I1=${IDXS[0]}; I2=${IDXS[1]}
echo "[run] P78061 -> lambda1:$I1 ; Q9KZC7 -> lambda1:$I2"
run_one P78061_2mg_glu "$I1" &
P1=$!
run_one Q9KZC7_2mg_glu "$I2" &
P2=$!
wait $P1; wait $P2
echo "[run] ALL DONE"

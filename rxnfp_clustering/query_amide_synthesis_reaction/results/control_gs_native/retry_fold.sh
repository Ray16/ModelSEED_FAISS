#!/usr/bin/env bash
set -u
cd /nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction
INDIR=$(pwd)/results/control_gs_native/inputs
OUTDIR=$(pwd)/results/control_gs_native/outputs
LOG=results/control_gs_native/logs
done_count(){ local n=$1; ls "$OUTDIR/by_seed/$n/s42/${n}_model_"*.cif 2>/dev/null | wc -l; }
for round in $(seq 1 8); do
  a=$(done_count P78061_2mg_glu); b=$(done_count Q9KZC7_2mg_glu)
  echo "[retry] round $round start: P78061=$a Q9KZC7=$b"
  if [ "$a" -ge 50 ] && [ "$b" -ge 50 ]; then echo "[retry] both complete"; exit 0; fi
  bash /nfs/lambda_stor_01/homes/rzhu/Tool/uplifting-biomolecular-modeling/boltz2/fold.sh \
       --input "$INDIR" --mode exact --out "$OUTDIR" -- --diffusion_samples 50 --seed 42 \
       > "$LOG/fold_round${round}.log" 2>&1
  echo "[retry] round $round fold.sh rc=$? : $(tail -1 "$LOG/fold_round${round}.log")"
done
a=$(done_count P78061_2mg_glu); b=$(done_count Q9KZC7_2mg_glu)
echo "[retry] FINAL: P78061=$a Q9KZC7=$b"
[ "$a" -ge 50 ] && [ "$b" -ge 50 ]

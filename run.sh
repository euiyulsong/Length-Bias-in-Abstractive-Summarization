python3 summary_drift_qwen3.py all \
  --train-n 1000 \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 2 \
  --gradient-checkpointing \
  --drift-chains 100 \
  --drift-steps 8

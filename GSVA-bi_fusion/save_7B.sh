#!/bin/bash
# Convert DeepSpeed ZeRO-2 sharded checkpoints to single .bin files.
# Run from /gemini/space/zhaozy/libingyu/GSVA-main

GSVA_ROOT=/gemini/space/zhaozy/libingyu/GSVA-main

cd ${GSVA_ROOT}/run_7B/gsva-bitemporal-run-train-val/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_7B_train_val.bin --max_shard_size 100GB
cd ${GSVA_ROOT}

cd ${GSVA_ROOT}/run_7B/gsva-bitemporal-run-NS-RS/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_7B_NS_RS.bin --max_shard_size 100GB
cd ${GSVA_ROOT}

cd ${GSVA_ROOT}/run_7B/gsva-bitemporal-run-RS-NS/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_7B_RS_NS.bin --max_shard_size 100GB
cd ${GSVA_ROOT}

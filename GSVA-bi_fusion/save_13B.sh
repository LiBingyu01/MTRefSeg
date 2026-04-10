# cd ./run_13B/gsva-bitemporal-run-train_val/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_13B_train_val.bin --max_shard_size 500GB
# cd /gemini/space/zhaozy/libingyu/GSVA-main


# cd ./run_13B/gsva-bitemporal-run-NS-RS/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_13B_NS_RS.bin --max_shard_size 500GB
# cd /gemini/space/zhaozy/libingyu/GSVA-main


cd ./run_13B/gsva-bitemporal-run-RS-NS/ckpt_model && python zero_to_fp32.py . ../GSVA_bi_fusion_13B_RS_NS.bin --max_shard_size 500GB
cd /gemini/space/zhaozy/libingyu/GSVA-main

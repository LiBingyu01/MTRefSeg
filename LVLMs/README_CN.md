# LVLM 双时间输入统一说明

本目录汇总了多套遥感多模态大模型代码，并统一改造成了**双时间图像输入**版本，用于变化理解、变化指代分割和变化定位等任务。

当前已经适配为双时间输入的仓库包括：

- `GeoPixel-main`
- `groundingLMM-bi_fusion`
- `GSVA-bi_fusion`
- `LISA-bi_fusion`
- `SegEarth-R1-main-down`
- `UniChange-main`
- `UniGeoSeg-main`

## 1. 改动说明

这些仓库原始版本大多以单张图像或单时相视觉输入为主。当前目录中的版本统一改为接收双时间输入，即：

- 输入图像由 `A` / `B` 两个时相组成
- 模型前向中同时编码两张图像
- 在视觉主干或多模态融合层中加入了双时间特征交互 / bi-fusion / temporal fusion
- 输出任务仍保持各自原始范式，例如 referring segmentation、mask prediction、change grounding、change localization

常见数据组织形式为：

```text
dataset_root/
├── A/
├── B/
├── masks/
│   └── <sample_name>/
└── referring_expression/
    └── *.json
```

不同仓库的脚本名不同，但整体都围绕同一件事：读取 `A/B` 双时相图像，并结合 mask / expression 完成变化理解。

## 2. 统一环境

所有仓库建议共用一套环境，基线依赖来自：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion/requirements.txt
```

推荐安装方式：

```bash
python -m pip install -r /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion/requirements.txt
```

其中 `mmcv` 建议单独重装，使用下面这组命令：

```bash
pip uninstall -y mmcv mmcv-full mmcv-lite openmim

python -m pip install -U pip wheel ninja
python -m pip install "setuptools<82" "packaging<25"

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=8

python -m pip install --no-build-isolation -v "mmcv==2.2.0"
```

补充说明：

- 主要环境依赖基于 `torch==2.4.0`、`transformers==4.31.0`、`deepspeed==0.9.5`、`peft==0.4.0`
- 下面所有命令都默认在**各自仓库根目录**执行
- 很多脚本中写了本地绝对路径，运行前需要先改脚本顶部的数据路径、权重路径和输出路径

## 3. 统一权重导出说明

这些仓库里一部分采用 LoRA + DeepSpeed ZeRO 训练，保存时通常分两步：

1. 先进入训练输出目录中的 `ckpt_model` 或 `checkpoint-*`，执行 `zero_to_fp32.py`，把分片权重合并成单个 `pytorch_model.bin`
2. 再执行对应仓库的 merge 脚本，把 LoRA 权重合并回基础模型，导出成可直接推理的 HuggingFace 格式

如果某个仓库本身不是 LoRA 训练，通常训练输出就会直接保存在 `--output_dir`，不需要额外 merge。

## 4. 各仓库使用方式

### 4.1 `GeoPixel-main`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
```

需要准备的权重：

- `pretrain_weights/GeoPixel-7B-RES`
- `pretrain_weights/sam2_hiera_large.pt`

说明：

- 脚本里通常把视觉权重写成 `pretrain_weights/sam2-hiera-large`
- 代码内部会把它映射到实际文件 `pretrain_weights/sam2_hiera_large.pt`

训练：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
bash train.sh
```

如果想直接跑完整 train + eval 流程，也可以使用：

```bash
bash run_train_and_eval.sh
```

保存 / 导出权重：

```bash
cd output/checkpoint-xxxx
python zero_to_fp32.py . ../pytorch_model.bin

cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
python merge_lora_weights_and_save_hf_model.py \
  --version pretrain_weights/GeoPixel-7B-RES \
  --vision_pretrained pretrain_weights/sam2-hiera-large \
  --weight output/pytorch_model.bin \
  --save_path output/GeoPixel_bitemporal_merged
```

推理 / 测评：

```bash
bash run_eval_zero_shot_all.sh
```

也可以直接参考：

- `evaluate_changeref.py`
- `docs/inference.md`

### 4.2 `groundingLMM-bi_fusion`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/groundingLMM-bi_fusion
```

需要准备的权重：

- `pretrained_weights/GLaMM-GranD-Pretrained`
- `pretrained_weights/sam_vit_h_4b8939.pth`
- `pretrained_weights/clip-vit-large-patch14`

训练：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/groundingLMM-bi_fusion
bash scripts/train_change_ref.sh
```

保存 / 导出权重：

```bash
bash save.sh
```

这个脚本内部已经包含：

- `zero_to_fp32.py` 合并 DeepSpeed 分片
- `scripts/merge_lora_weights.py` 合并 LoRA 回基础模型

推理 / 测评：

```bash
bash scripts/eval_change_ref.sh
```

可视化：

```bash
bash scripts/vis_change_ref.sh
```

### 4.3 `GSVA-bi_fusion`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GSVA-bi_fusion
```

需要准备的权重：

- `pretrained_weights/weights`（7B）
- `pretrained_weights/weights_13b`（13B）
- `pretrained_weights/gsva-7b-pt.bin`
- `pretrained_weights/gsva-13b-pt.bin`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/sam_vit_h_4b8939.pth`

训练：

7B：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GSVA-bi_fusion
bash scripts/train_demo_7B.sh
```

13B：

```bash
bash scripts/train_demo_13B.sh
```

保存 / 导出权重：

7B：

```bash
bash save_7B.sh
```

13B：

```bash
bash save_13B.sh
```

如果需要导出为 HuggingFace 格式，也可以继续使用：

```bash
python merge_lora_weights_and_save_hf_model.py
```

推理 / 测评：

```bash
bash scripts/eval_demo.sh
```

可视化：

```bash
bash vis_7B.sh
```

### 4.4 `LISA-bi_fusion`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion
```

需要准备的权重：

- `pretrained_weights/LISA-7B-v1`
- `pretrained_weights/LISA-13B-llama2-v1`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/sam_vit_h_4b8939.pth`

训练：

7B：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion
bash run_7B.sh
```

13B：

```bash
bash run_13B.sh
```

保存 / 导出权重：

7B：

```bash
bash save_7B.sh
```

13B 也遵循同样流程：

- 先进入对应 `ckpt_model` 目录执行 `zero_to_fp32.py`
- 再运行 `merge_lora_weights_and_save_hf_model.py` 合并 LoRA

推理 / 测评：

7B：

```bash
bash infer_7B.sh
```

13B：

```bash
bash infer_13B.sh
```

### 4.5 `SegEarth-R1-main-down`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/SegEarth-R1-main-down
```

需要准备的权重：

- `pretrained_weights/phi-1_5_dev`
- `pretrained_weights/model_final_9d7f02.pkl`

可选的已导出权重：

- `pretrained_weights/SegEarth-R1-RRSIS-D`

训练：

train/val 主版本：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/SegEarth-R1-main-down
bash scripts/train_change_train_val.sh
```

NS：

```bash
bash scripts/train_change_ns.sh
```

RS：

```bash
bash scripts/train_change_rs.sh
```

保存 / 导出权重：

- 该仓库主训练流程默认直接把 checkpoint 保存到脚本指定的 `--output_dir`
- 一般不需要额外做 LoRA merge
- 如果你要发布最终推理权重，直接整理 `output_dir` 下最佳 checkpoint 即可

推理 / 测评：

```bash
bash scripts/eval_change.sh
```

批量评测：

```bash
bash scripts/eval_all.sh
```

### 4.6 `UniChange-main`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniChange-main
```

需要准备的权重：

- `pretrained_weights/LLaVA-Lightning-7B-delta-v1-1`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/ViT_L.pth`

训练：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniChange-main
bash train_UniChange.sh
```

保存 / 导出权重：

```bash
bash save.sh
```

该脚本内部已经包含：

- `zero_to_fp32.py` 合并分片
- `merge_lora_weights_and_save_hf_model.py` 合并 LoRA

推理 / 测评：

```bash
bash validation_UniChange.sh
```

可视化：

```bash
bash vis_7B.sh
```

### 4.7 `UniGeoSeg-main`

仓库路径：

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniGeoSeg-main
```

需要准备的权重：

- `pretrained_weights/UniGeoSeg`

当前说明：

- 当前这个双时间改造目录里，已经整理好了**推理 / 测评**入口
- 但没有像其它仓库那样单独暴露出完整的训练脚本和导出脚本
- 因此这里建议把它视为一个**已整理好的双时间评测版本**

推理 / 测评：

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniGeoSeg-main
bash scripts/eval_change.sh
```

批量评测：

```bash
bash scripts/eval_all.sh
```

## 5. 建议使用顺序

如果你是第一次在这套目录上复现实验，建议按照下面顺序来：

1. 先创建统一环境，并按上面的方式安装 `mmcv==2.2.0`
2. 准备每个仓库各自的基础权重
3. 打开对应训练脚本，先检查数据路径、输出路径、GPU 数量、checkpoint 路径
4. 训练结束后，如果该仓库是 LoRA + DeepSpeed 方案，就先执行 `zero_to_fp32.py`，再执行 merge 脚本
5. 最后使用各自的 `eval` / `infer` / `vis` 脚本做推理和结果检查

## 6. 备注

- `LISA-bi_fusion` 的 `requirements.txt` 被作为这一整套工程的统一环境基线
- `SegEarth-R1-main-down` 更偏向直接保存训练输出，不是默认 LoRA merge 流程
- `UniGeoSeg-main` 当前主要提供双时间推理 / 测评能力
- `GeoPixel-main` 的视觉权重命名需要注意：脚本参数是 `pretrain_weights/sam2-hiera-large`，实际文件是 `pretrain_weights/sam2_hiera_large.pt`


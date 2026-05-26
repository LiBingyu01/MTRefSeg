# Unified Guide for Bi-Temporal LVM Repositories

This directory collects several conventional vision-language segmentation repositories and adapts them to **bi-temporal image input** for change referring segmentation and change understanding.

The following repositories have been adapted:

- `CRIS.pytorch-master-down`
- `FIANet-master-down`
- `LAVT-RIS-main-down`
- `RMSIN-main-down`
- `robust-ref-seg-main-down-RefSegformer`
- `rrsis-main-down-LGCE`
- `RSRefSeg-release-down`

All of them now operate on paired `A/B` temporal images together with a referring expression, and output the corresponding change mask.

## 1. Common Data Layout

These adapted repositories usually read data from:

```bash
/lby_data01/zhaozy/lby/ChangeRef_Clear
```

Common split names include:

```text
ChangeRef_Clear/
├── train_FINAL_CLEAN/
├── val_FINAL_CLEAN/
├── NS_FINAL_CLEAN_train/
├── NS_FINAL_CLEAN_val/
├── RS_FINAL_CLEAN_train/
├── RS_FINAL_CLEAN_val/
```

Each split typically contains:

```text
split_root/
├── A/
├── B/
├── masks/
└── referring_expression/
```

The script names vary across repositories, but the overall logic is the same: load dual-time `A/B` images and predict the referred change region.

## 2. Shared Environment

The recommended base environment is:

```bash
LVM/FIANet-master-down/requirements_for_VLM.txt
```

Base installation:

```bash
python -m pip install -r LVM/FIANet-master-down/requirements_for_VLM.txt
```

For `punkt_tab`, include the following commands exactly:

```bash
cd LVM/FIANet-master-down
mkdir -p /root/nltk_data/tokenizers
unzip punkt_tab.zip -d /root/nltk_data/tokenizers/
```

Additional notes:

- use `requirements_for_VLM.txt` as the shared base environment
- if a specific repository still misses some dependencies, install its own `requirements.txt`
- the simplest fallback is:

```bash
python -m pip install -r /path/to/repo/requirements.txt
```

- `RSRefSeg-release-down` belongs to the newer `mmseg/mmengine` stack, so if `mmcv/mmengine/mmsegmentation` related imports are missing, install the remaining dependencies required by that repository

## 3. Unified Checkpoint Notes

Unlike the multimodal LLM repositories, most repositories in this `LVM` directory save standard `.pth` checkpoints directly after training. No LoRA merge is needed in the common workflow.

Exception:

- `RSRefSeg-release-down` uses a `DeepSpeed/mmengine` style workflow
- before testing, you typically export `epoch_xx.pth` into `pytorch_model.bin` with `zero_to_fp32.py`

For the other repositories, the common pattern is:

- training directly saves `model_best_xxx.pth`, `best_model.pth`, or `ckpt.pth`
- inference uses the saved `.pth` directly via `--resume`

## 4. Repository-Wise Usage

### 4.1 `CRIS.pytorch-master-down`

Repository path:

```bash
LVM/CRIS.pytorch-master-down
```

Required pretrained weights:

- `pretrain/RN50.pt`
- `pretrain/RN101.pt`

Notes:

- the initialization weights are CLIP ResNet checkpoints
- the bi-temporal configs are mainly under `config/changeref_r50*.yaml` and `config/changeref_r101*.yaml`

Training:

ResNet-50:

```bash
cd LVM/CRIS.pytorch-master-down
WANDB_MODE=offline python -u train.py --config config/changeref_r50.yaml
```

ResNet-101:

```bash
WANDB_MODE=offline python -u train.py --config config/changeref_r101.yaml
```

For NS / RS settings, switch to:

- `config/changeref_r50_NS.yaml`
- `config/changeref_r50_RS.yaml`
- `config/changeref_r101_NS.yaml`
- `config/changeref_r101_RS.yaml`

Saving / exporting checkpoints:

- training saves directly to `{output_folder}/{exp_name}/` defined in the config
- common files include:
  - `best_model.pth`
  - `last_model.pth`

Inference / evaluation:

ResNet-50:

```bash
python -u test.py --config config/changeref_r50.yaml
```

ResNet-101:

```bash
python -u test101.py --config config/changeref_r101.yaml
```

Visualization:

```bash
python -u vis_result.py --config config/changeref_r101.yaml
```

Additional note:

- `run.sh` and `run101.sh` already keep common train / test / visualization templates

### 4.2 `FIANet-master-down`

Repository path:

```bash
LVM/FIANet-master-down
```

Required pretrained weights:

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

Training:

train-val:

```bash
cd LVM/FIANet-master-down
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py \
  --dataset change_ref \
  --batch-size 16 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --epochs 20 \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `train_NS.py`
- `train_RS.py`

Saving / exporting checkpoints:

- checkpoints are saved directly under `output/.../model_save/`
- the best checkpoint is typically:

```text
model_best_FIANet.pth
```

Inference / evaluation:

train-val:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 test.py \
  --dataset change_ref \
  --batch-size 16 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --resume ./output/change_ref/model_save/model_best_FIANet.pth \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `test_NS.py`
- `test_RS.py`

Visualization:

```bash
bash vis.sh
```

Additional notes:

- `run.sh` is currently evaluation-oriented, while the training commands are kept in comments
- `vis.sh` contains hardcoded dataset/checkpoint paths, so update them before running

### 4.3 `LAVT-RIS-main-down`

Repository path:

```bash
LVM/LAVT-RIS-main-down
```

Required pretrained weights:

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

Training:

train-val:

```bash
cd LVM/LAVT-RIS-main-down
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py \
  --model lavt_change \
  --dataset change_ref \
  --model_id lavt_change \
  --batch-size 16 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --epochs 20 \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `train_NS.py`
- `train_RS.py`

Saving / exporting checkpoints:

- the common best checkpoint path is:

```text
output/change_ref/model_save/model_best_lavt_change.pth
```

Inference / evaluation:

train-val:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 test.py \
  --model lavt_change \
  --dataset change_ref \
  --model_id lavt_change \
  --batch-size 16 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --resume ./output/change_ref/model_save/model_best_lavt_change.pth \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `test_NS.py`
- `test_RS.py`

Visualization:

```bash
bash vis.sh
```

Additional note:

- update the hardcoded dataset and checkpoint paths in `vis.sh` before running it

### 4.4 `RMSIN-main-down`

Repository path:

```bash
LVM/RMSIN-main-down
```

Required pretrained weights:

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

Training:

train-val:

```bash
cd LVM/RMSIN-main-down
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py \
  --model lavt_one \
  --dataset change_ref \
  --model_id lavt_one \
  --batch-size 16 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --epochs 20 \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `train_NS.py`
- `train_RS.py`

Saving / exporting checkpoints:

- checkpoints are saved directly under `output/.../model_save/`
- the best checkpoint is commonly:

```text
model_best_lavt_one.pth
```

Inference / evaluation:

train-val:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 test.py \
  --model lavt_one \
  --dataset change_ref \
  --model_id lavt_one \
  --batch-size 16 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --resume ./output/change_ref/model_save/model_best_lavt_one.pth \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `test_NS.py`
- `test_RS.py`

Visualization:

```bash
bash run_visualize.sh
```

Additional note:

- `run_visualize.sh` already provides a relatively complete visualization entry

### 4.5 `rrsis-main-down-LGCE`

Repository path:

```bash
LVM/rrsis-main-down-LGCE
```

Required pretrained weights:

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

Training:

train-val:

```bash
cd LVM/rrsis-main-down-LGCE
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py \
  --dataset change_ref \
  --batch-size 16 \
  --lr 0.00005 \
  --wd 1e-2 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --epochs 20 \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `train_NS.py`
- `train_RS.py`

Saving / exporting checkpoints:

- checkpoints are saved directly under `output/.../model_save/`
- the common best checkpoint name is:

```text
model_best_lavt.pth
```

Inference / evaluation:

train-val:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 test.py \
  --dataset change_ref \
  --batch-size 16 \
  --swin_type base \
  --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
  --resume ./output/change_ref/model_save/model_best_lavt.pth \
  --output_dir ./output/change_ref/model_save \
  --img_size 480
```

NS / RS:

- `test_NS.py`
- `test_RS.py`

Visualization:

```bash
bash vis.sh
```

### 4.6 `robust-ref-seg-main-down-RefSegformer`

Repository path:

```bash
LVM/robust-ref-seg-main-down-RefSegformer
```

Required pretrained weights:

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

Notes:

- `configs/swin_base_patch4_window12_480.yaml` uses Swin-B pretrained weights
- the default BERT tokenizer / BERT weights both point to `pretrained_weights/bert-base-uncased`

Training:

train-val:

```bash
cd LVM/robust-ref-seg-main-down-RefSegformer
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
  --exp MTRefSeg \
  --dataset MTRefSeg \
  --batch_size 16 \
  --use_mask \
  --use_exist \
  --use_pixel_decoder \
  --output ./logs \
  --epoch 20
```

NS / RS:

- `main_ns.py`
- `main_rs.py`

Saving / exporting checkpoints:

- checkpoints are saved to:

```text
./logs/<exp>/ckpt.pth
```

- examples:
  - `./logs/MTRefSeg/ckpt.pth`
  - `./logs/MTRefSeg_ns/ckpt.pth`
  - `./logs/MTRefSeg_rs/ckpt.pth`

Inference / evaluation:

train-val:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 test.py \
  --exp MTRefSeg \
  --dataset MTRefSeg \
  --batch_size 16 \
  --use_mask \
  --use_exist \
  --use_pixel_decoder \
  --output ./logs \
  --epoch 20 \
  --eval
```

NS / RS:

- `test_ns.py`
- `test_rs.py`

Additional notes:

- `run.sh` currently uses `./logs_eval` in the evaluation examples
- if your trained checkpoint is actually stored under `./logs/<exp>/ckpt.pth`, change `--output` accordingly during testing
- this repository does not currently provide a dedicated visualization wrapper

### 4.7 `RSRefSeg-release-down`

Repository path:

```bash
LVM/RSRefSeg-release-down
```

Required pretrained weights:

- `KyanChen/sam-vit-base`
- `thomas/siglip-so400m-patch14-384`

Notes:

- these models are referenced via `model_name_or_path` in the config files
- if network access is available, they are usually downloaded automatically on first use
- for offline usage, prepare them in the local cache / mirrored path in advance

Training:

train-val:

```bash
cd LVM/RSRefSeg-release-down
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-train-val.py 1
```

NS:

```bash
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-NS.py 1
```

RS:

```bash
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-RS.py 1
```

Saving / exporting checkpoints:

- raw training checkpoints are saved under:

```text
work_dirs/RSRefSeg-b-train-val/epoch_*.pth
work_dirs/RSRefSeg-b-NS/epoch_*.pth
work_dirs/RSRefSeg-b-RS/epoch_*.pth
```

- to export them in the format used by the current testing scripts:

train-val:

```bash
cd work_dirs/RSRefSeg-b-train-val
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

NS:

```bash
cd work_dirs/RSRefSeg-b-NS
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

RS:

```bash
cd work_dirs/RSRefSeg-b-RS
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

Inference / evaluation:

train-val:

```bash
cd LVM/RSRefSeg-release-down
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-train-val.py \
  work_dirs/RSRefSeg-b-train-val/exported_weights/pytorch_model.bin \
  1
```

NS:

```bash
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-NS.py \
  work_dirs/RSRefSeg-b-NS/exported_weights/pytorch_model.bin \
  1
```

RS:

```bash
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-RS.py \
  work_dirs/RSRefSeg-b-RS/exported_weights/pytorch_model.bin \
  1
```

Additional note:

- `run.sh` already contains complete examples for train / test / export
- among these repositories, this is the one closest to a modern `mmseg + mmengine + DeepSpeed` workflow

## 5. Recommended Usage Order

For first-time reproduction, the recommended order is:

1. install `FIANet-master-down/requirements_for_VLM.txt`
2. unpack `punkt_tab` with the commands shown above
3. if a repository still misses some dependencies, install its own `requirements.txt`
4. prepare the required pretrained weights for each repository
5. adjust dataset paths, checkpoint paths, and GPU ids in the scripts
6. after training, use the saved `.pth` directly for evaluation in the standard repositories
7. for `RSRefSeg-release-down`, run `zero_to_fp32.py` first, then `dist_test.sh`

## 6. Final Notes

- these `LVM` repositories are conventional vision-language segmentation models, not multimodal LLMs
- they have all been adapted to bi-temporal `A/B` image input
- `FIANet`, `LAVT`, `RMSIN`, `LGCE`, and `RefSegformer` are essentially in the `Swin + BERT` family
- `CRIS` is based on `CLIP RN50/RN101`
- `RSRefSeg` is based on `SAM + SigLIP`


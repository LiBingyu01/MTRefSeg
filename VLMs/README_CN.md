# LVM 双时间输入统一说明

本目录收集了一组传统视觉-语言分割模型（LVM），并统一改造成了**双时间图像输入**版本，用于变化指代分割 / 变化理解任务。

当前已经适配为双时间输入的仓库包括：

- `CRIS.pytorch-master-down`
- `FIANet-master-down`
- `LAVT-RIS-main-down`
- `RMSIN-main-down`
- `robust-ref-seg-main-down-RefSegformer`
- `rrsis-main-down-LGCE`
- `RSRefSeg-release-down`

这些仓库当前都围绕 `A/B` 双时相图像进行输入，目标仍然是根据文本表达式输出对应的变化区域 mask。

## 1. 常见数据形式

这些适配版通常默认从 `/lby_data01/zhaozy/lby/ChangeRef_Clear` 读取数据，常见结构为：

```text
ChangeRef_Clear/
├── train_FINAL_CLEAN/
├── val_FINAL_CLEAN/
├── NS_FINAL_CLEAN_train/
├── NS_FINAL_CLEAN_val/
├── RS_FINAL_CLEAN_train/
├── RS_FINAL_CLEAN_val/
```

每个 split 内部一般对应：

```text
split_root/
├── A/
├── B/
├── masks/
└── referring_expression/
```

不同仓库脚本名不同，但本质上都是读取双时相 `A/B` 图像和对应表达式，预测变化区域。

## 2. 统一环境

建议优先使用下面这个基础环境文件：

```bash
LVM/FIANet-master-down/requirements_for_VLM.txt
```

基础安装方式：

```bash
python -m pip install -r LVM/FIANet-master-down/requirements_for_VLM.txt
```

`punkt_tab` 也建议手动准备。按你的要求，保留下面这组命令：

```bash
cd LVM/FIANet-master-down
mkdir -p /root/nltk_data/tokenizers
unzip punkt_tab.zip -d /root/nltk_data/tokenizers/
```

补充说明：

- 这套 `requirements_for_VLM.txt` 可以作为公共基础环境
- 其他仓库如果缺少依赖，直接再安装各自的 `requirements.txt` 即可
- 最简单的补依赖方式是：

```bash
python -m pip install -r /path/to/repo/requirements.txt
```

- `RSRefSeg-release-down` 属于较新的 `mmseg/mmengine` 体系，如果运行时还缺 `mmcv/mmengine/mmsegmentation` 相关依赖，就按它自己的依赖继续补

## 3. 统一保存说明

这组 LVM 仓库和前面的多模态大模型不同，绝大多数训练后直接保存 `.pth` 权重，不需要做 LoRA merge。

例外：

- `RSRefSeg-release-down` 使用了 `DeepSpeed/mmengine` 风格训练
- 测试前通常需要先把 `epoch_xx.pth` 通过 `zero_to_fp32.py` 导出成单独的 `pytorch_model.bin`

其余仓库通常都是：

- 训练直接输出 `model_best_xxx.pth` / `best_model.pth` / `ckpt.pth`
- 推理阶段直接 `--resume` 这个 `.pth` 即可

## 4. 各仓库使用方式

### 4.1 `CRIS.pytorch-master-down`

仓库路径：

```bash
LVM/CRIS.pytorch-master-down
```

需要准备的权重：

- `pretrain/RN50.pt`
- `pretrain/RN101.pt`

说明：

- 这里的初始化权重是 CLIP ResNet 版本
- 当前双时间改造配置主要放在 `config/changeref_r50*.yaml` 和 `config/changeref_r101*.yaml`

训练：

ResNet-50：

```bash
cd LVM/CRIS.pytorch-master-down
WANDB_MODE=offline python -u train.py --config config/changeref_r50.yaml
```

ResNet-101：

```bash
WANDB_MODE=offline python -u train.py --config config/changeref_r101.yaml
```

NS / RS 版本直接换对应配置文件：

- `config/changeref_r50_NS.yaml`
- `config/changeref_r50_RS.yaml`
- `config/changeref_r101_NS.yaml`
- `config/changeref_r101_RS.yaml`

保存 / 导出权重：

- 训练脚本会直接保存到配置文件定义的 `{output_folder}/{exp_name}/`
- 典型文件包括：
  - `best_model.pth`
  - `last_model.pth`

推理 / 测评：

ResNet-50：

```bash
python -u test.py --config config/changeref_r50.yaml
```

ResNet-101：

```bash
python -u test101.py --config config/changeref_r101.yaml
```

可视化：

```bash
python -u vis_result.py --config config/changeref_r101.yaml
```

补充：

- `run.sh` 和 `run101.sh` 里保留了常用 train / test / vis 模板

### 4.2 `FIANet-master-down`

仓库路径：

```bash
LVM/FIANet-master-down
```

需要准备的权重：

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

训练：

train-val：

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

NS / RS：

- `train_NS.py`
- `train_RS.py`

保存 / 导出权重：

- 训练后直接保存在 `output/.../model_save/`
- 最优权重文件名通常为：

```text
model_best_FIANet.pth
```

推理 / 测评：

train-val：

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

NS / RS：

- `test_NS.py`
- `test_RS.py`

可视化：

```bash
bash vis.sh
```

补充：

- `run.sh` 当前默认更偏向测试，训练命令也保留在注释里
- `vis.sh` 中的数据路径和 checkpoint 路径是硬编码的，运行前先改

### 4.3 `LAVT-RIS-main-down`

仓库路径：

```bash
LVM/LAVT-RIS-main-down
```

需要准备的权重：

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

训练：

train-val：

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

NS / RS：

- `train_NS.py`
- `train_RS.py`

保存 / 导出权重：

- 训练后直接保存为：

```text
output/change_ref/model_save/model_best_lavt_change.pth
```

推理 / 测评：

train-val：

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

NS / RS：

- `test_NS.py`
- `test_RS.py`

可视化：

```bash
bash vis.sh
```

补充：

- `vis.sh` 里的数据路径和 checkpoint 路径需要按本机环境调整

### 4.4 `RMSIN-main-down`

仓库路径：

```bash
LVM/RMSIN-main-down
```

需要准备的权重：

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

训练：

train-val：

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

NS / RS：

- `train_NS.py`
- `train_RS.py`

保存 / 导出权重：

- 训练后直接保存到 `output/.../model_save/`
- 对应最优权重通常为：

```text
model_best_lavt_one.pth
```

推理 / 测评：

train-val：

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

NS / RS：

- `test_NS.py`
- `test_RS.py`

可视化：

```bash
bash run_visualize.sh
```

补充：

- `run_visualize.sh` 已经给了一个比较完整的可视化入口

### 4.5 `rrsis-main-down-LGCE`

仓库路径：

```bash
LVM/rrsis-main-down-LGCE
```

需要准备的权重：

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

训练：

train-val：

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

NS / RS：

- `train_NS.py`
- `train_RS.py`

保存 / 导出权重：

- 训练后直接保存到 `output/.../model_save/`
- 常见最优权重名为：

```text
model_best_lavt.pth
```

推理 / 测评：

train-val：

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

NS / RS：

- `test_NS.py`
- `test_RS.py`

可视化：

```bash
bash vis.sh
```

### 4.6 `robust-ref-seg-main-down-RefSegformer`

仓库路径：

```bash
LVM/robust-ref-seg-main-down-RefSegformer
```

需要准备的权重：

- `pretrained_weights/swin_base_patch4_window12_384_22k.pth`
- `pretrained_weights/bert-base-uncased`

说明：

- 配置文件 `configs/swin_base_patch4_window12_480.yaml` 中使用的是 Swin-B 预训练权重
- BERT tokenizer / BERT 权重默认都指向 `pretrained_weights/bert-base-uncased`

训练：

train-val：

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

NS / RS：

- `main_ns.py`
- `main_rs.py`

保存 / 导出权重：

- 训练阶段会保存到：

```text
./logs/<exp>/ckpt.pth
```

- 例如：
  - `./logs/MTRefSeg/ckpt.pth`
  - `./logs/MTRefSeg_ns/ckpt.pth`
  - `./logs/MTRefSeg_rs/ckpt.pth`

推理 / 测评：

train-val：

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

NS / RS：

- `test_ns.py`
- `test_rs.py`

补充：

- `run.sh` 里的测试命令当前写的是 `./logs_eval`
- 如果你的训练 checkpoint 实际保存在 `./logs/<exp>/ckpt.pth`，测试时要把 `--output` 改成对应训练目录
- 这个仓库当前没有单独整理好的可视化脚本

### 4.7 `RSRefSeg-release-down`

仓库路径：

```bash
LVM/RSRefSeg-release-down
```

需要准备的权重：

- `KyanChen/sam-vit-base`
- `thomas/siglip-so400m-patch14-384`

说明：

- 这两个基础模型在配置中以 `model_name_or_path` 形式指定
- 如果联网正常，首次运行时通常会自动下载到缓存目录
- 如果离线运行，需要你提前把这些模型准备到本地缓存 / 镜像路径

训练：

train-val：

```bash
cd LVM/RSRefSeg-release-down
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-train-val.py 1
```

NS：

```bash
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-NS.py 1
```

RS：

```bash
bash tools/dist_train.sh configs_RSRefSeg/RSRefSeg-b-RS.py 1
```

保存 / 导出权重：

- 原始训练权重保存在：

```text
work_dirs/RSRefSeg-b-train-val/epoch_*.pth
work_dirs/RSRefSeg-b-NS/epoch_*.pth
work_dirs/RSRefSeg-b-RS/epoch_*.pth
```

- 如果要按当前仓库脚本方式测试，先导出：

train-val：

```bash
cd work_dirs/RSRefSeg-b-train-val
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

NS：

```bash
cd work_dirs/RSRefSeg-b-NS
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

RS：

```bash
cd work_dirs/RSRefSeg-b-RS
python zero_to_fp32.py . exported_weights --tag epoch_20.pth
```

推理 / 测评：

train-val：

```bash
cd LVM/RSRefSeg-release-down
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-train-val.py \
  work_dirs/RSRefSeg-b-train-val/exported_weights/pytorch_model.bin \
  1
```

NS：

```bash
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-NS.py \
  work_dirs/RSRefSeg-b-NS/exported_weights/pytorch_model.bin \
  1
```

RS：

```bash
bash tools/dist_test.sh \
  configs_RSRefSeg/RSRefSeg-b-RS.py \
  work_dirs/RSRefSeg-b-RS/exported_weights/pytorch_model.bin \
  1
```

补充：

- `run.sh` 里已经整理好了 train / test / export 的完整示例
- 这是这组仓库里最接近现代 `mmseg + mmengine + DeepSpeed` 工作流的一套

## 5. 建议使用顺序

建议第一次复现实验时按下面顺序做：

1. 先安装 `FIANet-master-down/requirements_for_VLM.txt`
2. 再执行 `punkt_tab` 的解压命令
3. 如果某个仓库仍然缺依赖，再补它自己的 `requirements.txt`
4. 准备每个仓库需要的基础权重
5. 修改脚本中的数据路径、checkpoint 路径、GPU 编号
6. 训练完成后，普通仓库直接使用 `.pth` 测试
7. `RSRefSeg-release-down` 先执行 `zero_to_fp32.py`，再做 `dist_test.sh`

## 6. 备注

- 这组 `LVM` 仓库属于传统 vision-language segmentation 模型，不是多模态大模型
- 它们现在都已经被改造成双时间 `A/B` 输入版本
- `FIANet` / `LAVT` / `RMSIN` / `LGCE` / `RefSegformer` 这一组本质上都依赖 `Swin + BERT`
- `CRIS` 使用的是 `CLIP RN50/RN101`
- `RSRefSeg` 使用的是 `SAM + SigLIP`


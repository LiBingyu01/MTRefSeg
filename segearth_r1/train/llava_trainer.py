import os
import torch
import shutil
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available, is_torch_tpu_available,is_accelerate_available
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



import torch

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class ChangeDetectionEvalCallback(TrainerCallback):
    """
    每轮训练结束后，在验证集上评测变化检测分割性能。
    直接使用内存中的模型，无需重新加载 checkpoint。

    空 mask 处理规则：
      GT 非空，Pred 非空：正常计算 IoU
      GT 非空，Pred 为空：IoU = 0
      GT 为空，Pred 非空：IoU = 0
      GT 为空，Pred 为空：IoU = 1
    """

    def __init__(self, val_dataset, val_collator, eval_batch_size=1,
                 dataloader_num_workers=2, max_eval_samples=None):
        self.val_dataset = val_dataset
        self.val_collator = val_collator
        self.eval_batch_size = eval_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.max_eval_samples = max_eval_samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        # distributed 信息
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0

        # 构建 DataLoader
        eval_dataset = self.val_dataset
        if self.max_eval_samples is not None and len(eval_dataset) > self.max_eval_samples:
            eval_dataset = torch.utils.data.Subset(eval_dataset, range(self.max_eval_samples))

        if distributed:
            sampler = DistributedSampler(eval_dataset, shuffle=False)
            dataloader = DataLoader(
                eval_dataset,
                batch_size=self.eval_batch_size,
                sampler=sampler,
                collate_fn=self.val_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
            )
        else:
            dataloader = DataLoader(
                eval_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self.val_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
            )

        # 切换到 eval 模式
        was_training = model.training
        model.eval()

        device = next(model.parameters()).device

        eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]

        seg_correct = torch.zeros(
            len(eval_seg_iou_list), device=device, dtype=torch.float64
        )
        seg_total = torch.zeros(1, device=device, dtype=torch.float64)

        sum_iou = torch.zeros(1, device=device, dtype=torch.float64)
        cum_I = torch.zeros(1, device=device, dtype=torch.float64)
        cum_U = torch.zeros(1, device=device, dtype=torch.float64)

        # 额外统计空 mask 情况
        empty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
        empty_pred_total = torch.zeros(1, device=device, dtype=torch.float64)
        empty_both_total = torch.zeros(1, device=device, dtype=torch.float64)
        nonempty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
        false_positive_empty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
        false_negative_nonempty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)

        with torch.no_grad():
            for inputs in dataloader:
                gt = inputs.pop("masks")

                for k in list(inputs.keys()):
                    if torch.is_tensor(inputs[k]):
                        inputs[k] = inputs[k].to(device)

                gt = gt.to(device)

                if "token_refer_id" in inputs:
                    inputs["token_refer_id"] = [
                        ids.to(device) for ids in inputs["token_refer_id"]
                    ]

                # 使用 unwrap_model 兼容 DeepSpeed / DDP 包装
                raw_model = unwrap_model(model)
                model_dtype = next(raw_model.parameters()).dtype
                if model_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    model_dtype = torch.float32

                outputs = raw_model.eval_seg(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    images=inputs["images"].to(dtype=model_dtype),
                    images_t2=inputs["images_t2"].to(dtype=model_dtype),
                    masks=gt,
                    token_refer_id=inputs.get("token_refer_id"),
                    refer_embedding_indices=inputs.get("refer_embedding_indices"),
                    labels=inputs.get("labels"),
                    token_answer_id=None,
                    answer_embedding_indices=None,
                )

                # pred masks -> [B, H, W]
                pred_masks = []
                for j in range(len(outputs)):
                    pred_mask = outputs[j]["pred_masks"]
                    if pred_mask.dim() > 2:
                        pred_mask = pred_mask[0]
                    pred_masks.append(pred_mask)

                pred_masks = torch.stack(pred_masks, dim=0).float()

                # gt -> [B, H, W]
                if gt.dim() == 4 and gt.shape[1] == 1:
                    gt_masks = gt.squeeze(1).float()
                elif gt.dim() == 3:
                    gt_masks = gt.float()
                else:
                    raise ValueError(f"Unexpected gt shape: {gt.shape}")

                # 如果 pred 已经是概率图 [0,1]，则直接用；否则 sigmoid
                if pred_masks.min() >= 0 and pred_masks.max() <= 1:
                    pred_prob = pred_masks
                else:
                    pred_prob = torch.sigmoid(pred_masks)

                pred_bin = (pred_prob > 0.5).float()
                gt_bin = (gt_masks > 0.5).float()

                batch_size = pred_bin.shape[0]

                pred_flat = pred_bin.reshape(batch_size, -1)
                gt_flat = gt_bin.reshape(batch_size, -1)

                # foreground intersection / union
                inter = (pred_flat * gt_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - inter

                gt_area = gt_flat.sum(dim=1)
                pred_area = pred_flat.sum(dim=1)

                empty_gt = gt_area == 0
                empty_pred = pred_area == 0
                empty_both = empty_gt & empty_pred

                nonempty_gt = gt_area > 0
                false_positive_empty_gt = empty_gt & (~empty_pred)
                false_negative_nonempty_gt = nonempty_gt & empty_pred

                # 常规 IoU
                iou = inter / (union + 1e-6)

                # 关键修正：
                # GT 全黑且 Pred 全黑，说明模型正确预测“无目标”，IoU 设为 1
                # GT 全黑但 Pred 非空，仍然自然为 0
                iou[empty_both] = 1.0

                # 统计空 mask 情况
                empty_gt_total += empty_gt.sum().double()
                empty_pred_total += empty_pred.sum().double()
                empty_both_total += empty_both.sum().double()
                nonempty_gt_total += nonempty_gt.sum().double()
                false_positive_empty_gt_total += false_positive_empty_gt.sum().double()
                false_negative_nonempty_gt_total += false_negative_nonempty_gt.sum().double()

                # 累计指标
                # empty_both 样本 inter=0, union=0，不影响 oIoU；
                # 但通过 iou=1 影响 mIoU 和 Precision@k。
                cum_I += inter.sum().double()
                cum_U += union.sum().double()
                sum_iou += iou.sum().double()
                seg_total += batch_size

                for idx, thr in enumerate(eval_seg_iou_list):
                    seg_correct[idx] += (iou >= thr).sum().double()

        # 多卡汇总
        if distributed:
            dist.all_reduce(seg_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(seg_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_iou, op=dist.ReduceOp.SUM)
            dist.all_reduce(cum_I, op=dist.ReduceOp.SUM)
            dist.all_reduce(cum_U, op=dist.ReduceOp.SUM)

            dist.all_reduce(empty_gt_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(empty_pred_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(empty_both_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(nonempty_gt_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(false_positive_empty_gt_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(false_negative_nonempty_gt_total, op=dist.ReduceOp.SUM)

        # 打印 & 记录，仅 rank 0
        if rank == 0:
            epoch = int(state.epoch) if state.epoch is not None else "?"
            total = int(seg_total.item())

            if total > 0:
                miou = (sum_iou / seg_total).item()
            else:
                miou = 0.0

            if cum_U.item() > 0:
                oiou = (cum_I / (cum_U + 1e-6)).item()
            else:
                # 如果所有样本都是 GT 空且 Pred 空，则整体也视为完全正确
                oiou = 1.0 if int(empty_both_total.item()) == total and total > 0 else 0.0

            print(f"\n{'=' * 45}")
            print(f"  [Val] Epoch {epoch}  |  samples={total}")
            print(f"  mIoU : {miou * 100:.2f}%   oIoU : {oiou * 100:.2f}%")
            print(f"{'-' * 45}")

            for idx, thr in enumerate(eval_seg_iou_list):
                prec = (seg_correct[idx] / seg_total).item() if total > 0 else 0.0
                print(f"  Prec@{thr:.1f}: {prec * 100:.2f}%")

            print(f"{'-' * 45}")
            print(f"  Non-empty GT samples          : {int(nonempty_gt_total.item())}")
            print(f"  Empty GT samples              : {int(empty_gt_total.item())}")
            print(f"  Empty Pred samples            : {int(empty_pred_total.item())}")
            print(f"  Empty GT & Empty Pred samples : {int(empty_both_total.item())}")
            print(f"  Empty GT but Non-empty Pred FP: {int(false_positive_empty_gt_total.item())}")
            print(f"  Non-empty GT but Empty Pred FN: {int(false_negative_nonempty_gt_total.item())}")
            print(f"{'=' * 45}\n")

            # 写入 Trainer 日志
            if state.log_history is not None:
                log_entry = {
                    "epoch": state.epoch,
                    "val_mIoU": round(miou * 100, 4),
                    "val_oIoU": round(oiou * 100, 4),
                    "val_total_samples": total,
                    "val_nonempty_gt": int(nonempty_gt_total.item()),
                    "val_empty_gt": int(empty_gt_total.item()),
                    "val_empty_pred": int(empty_pred_total.item()),
                    "val_empty_both": int(empty_both_total.item()),
                    "val_empty_gt_nonempty_pred_fp": int(false_positive_empty_gt_total.item()),
                    "val_nonempty_gt_empty_pred_fn": int(false_negative_nonempty_gt_total.item()),
                }

                for idx, thr in enumerate(eval_seg_iou_list):
                    prec = (seg_correct[idx] / seg_total).item() if total > 0 else 0.0
                    log_entry[f"val_Prec@{thr}"] = round(prec * 100, 4)

                state.log_history.append(log_entry)

        # 恢复训练模式
        if was_training:
            model.train()


class LLaVATrainer(Trainer):
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     if dist.is_available():
    #         dist.barrier()
    #     import ipdb;ipdb.set_trace()
    #     if hasattr(self.train_dataset,'cur_dataset_index'):
    #         self.train_dataset.update_dataset_index()
    #     print(self.train_dataset.cur_dataset_index)
    #
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()


    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs.pop("image_name", None)
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'loss_dice' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                # loss_mask = outputs["loss_mask"].item() if isinstance(outputs, dict) else 0
                # loss_dice = outputs["loss_dice"].item() if isinstance(outputs, dict) else 0
                # loss_SEG_class = outputs["loss_SEG_class"].item() if isinstance(outputs, dict) else 0
                # loss_class_name_class = outputs["loss_class_name_class"].item() if isinstance(outputs, dict) else 0
                # loss_dict = {
                #     'loss_mask':loss_mask,
                #     'loss_dice': loss_dice,
                #     'loss_SEG_class':loss_SEG_class,
                #     'loss_class_name_class': loss_class_name_class
                # }
                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss

    # def training_step(self, model, inputs) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps

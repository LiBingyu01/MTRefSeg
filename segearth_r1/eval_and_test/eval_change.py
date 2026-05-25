"""
Evaluation script for bi-temporal referring change detection with SegEarth-R1.

Single-GPU usage:
  python segearth_r1/eval_and_test/eval_change.py \
      --model_path /path/to/checkpoint \
      --base_data_path /path/to/change_dataset \
      --data_split val \
      --mask_config segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
      --version llava_phi

Multi-GPU usage:
  torchrun --nproc_per_node=4 segearth_r1/eval_and_test/eval_change.py \
      --model_path /path/to/checkpoint \
      --base_data_path /path/to/change_dataset \
      --data_split val \
      --mask_config segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
      --version llava_phi
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import transformers
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from eval_dataset.change_val_dataset import ChangeValDataset, ChangeValDataCollector
from segearth_r1.model.builder import load_pretrained_model
from segearth_r1.utils import disable_torch_init
from segearth_r1.mm_utils import get_model_name_from_path
from segearth_r1 import conversation as conversation_lib


@dataclass
class EvalArgs:
    model_path: str = field(default="/path/to/model")
    base_data_path: str = field(default="/path/to/change_dataset")
    mask_config: str = field(
        default="./segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    version: str = field(default="llava_phi")
    data_split: str = field(default="val")
    image_size: int = field(default=1024)
    eval_batch_size: int = field(default=4)
    dataloader_num_workers: int = field(default=4)
    use_seg_query: bool = field(default=False)
    vis_path: Optional[str] = field(default=None)
    model_map_name: str = field(default="segearth_r1")


def evaluation():
    parser = transformers.HfArgumentParser(EvalArgs)
    args = parser.parse_args_into_dataclasses()[0]

    # distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if rank == 0:
        print(f"Model: {args.model_path}")

    device = f"cuda:{local_rank}" if distributed else "cuda"

    tokenizer, model, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        model_args=args,
        mask_config=args.mask_config,
        use_seg_query=args.use_seg_query,
        device=device,
    )

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]

    dataset = ChangeValDataset(
        base_data_path=args.base_data_path,
        tokenizer=tokenizer,
        split=args.data_split,
        image_size=args.image_size,
    )
    collator = ChangeValDataCollector(tokenizer=tokenizer)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

    model.to(device=device, dtype=torch.float).eval()

    # Metrics
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]

    seg_correct = torch.zeros(len(eval_seg_iou_list), device=device, dtype=torch.float64)
    seg_total = torch.zeros(1, device=device, dtype=torch.float64)

    sum_iou = torch.zeros(1, device=device, dtype=torch.float64)
    cum_I = torch.zeros(1, device=device, dtype=torch.float64)
    cum_U = torch.zeros(1, device=device, dtype=torch.float64)

    # Extra statistics for empty-mask analysis
    empty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
    empty_pred_total = torch.zeros(1, device=device, dtype=torch.float64)
    empty_both_total = torch.zeros(1, device=device, dtype=torch.float64)
    nonempty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
    false_positive_empty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)
    false_negative_nonempty_gt_total = torch.zeros(1, device=device, dtype=torch.float64)

    if rank == 0 and args.vis_path is not None:
        os.makedirs(args.vis_path, exist_ok=True)

    with torch.no_grad():
        for inputs in tqdm(
            dataloader,
            desc=f"Eval [{args.data_split}]",
            disable=(rank != 0),
        ):
            gt = inputs.pop("masks")

            for k in list(inputs.keys()):
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].to(device)

            gt = gt.to(device)

            if "token_refer_id" in inputs:
                inputs["token_refer_id"] = [
                    ids.to(device) for ids in inputs["token_refer_id"]
                ]

            model_dtype = next(model.parameters()).dtype

            outputs = model.eval_seg(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"].to(dtype=model_dtype),
                images_t2=inputs["images_t2"].to(dtype=model_dtype),
                masks=gt,
                token_refer_id=inputs.get("token_refer_id"),
                refer_embedding_indices=inputs.get("refer_embedding_indices"),
                labels=inputs.get("labels"),
                token_answer_id=None,
                answer_embedding_indices=None,
            )

            # Collect prediction masks to [B, H, W]
            pred_masks = []
            for j in range(len(outputs)):
                pred_mask = outputs[j]["pred_masks"]

                # common shapes: [1,H,W] or [H,W]
                if pred_mask.dim() > 2:
                    pred_mask = pred_mask[0]

                pred_masks.append(pred_mask)

            pred_masks = torch.stack(pred_masks, dim=0).float()

            # GT -> [B, H, W]
            if gt.dim() == 4 and gt.shape[1] == 1:
                gt_masks = gt.squeeze(1).float()
            elif gt.dim() == 3:
                gt_masks = gt.float()
            else:
                raise ValueError(f"Unexpected gt shape: {gt.shape}")

            # If pred is already probability map [0,1], use it directly.
            # Otherwise apply sigmoid.
            if pred_masks.min() >= 0 and pred_masks.max() <= 1:
                pred_prob = pred_masks
            else:
                pred_prob = torch.sigmoid(pred_masks)

            pred_bin = (pred_prob > 0.5).float()
            gt_bin = (gt_masks > 0.5).float()

            batch_size = pred_bin.shape[0]
            pred_flat = pred_bin.reshape(batch_size, -1)
            gt_flat = gt_bin.reshape(batch_size, -1)

            # Foreground intersection and union for each sample
            intersection_tensor = (pred_flat * gt_flat).sum(dim=1)
            union_tensor = (
                pred_flat.sum(dim=1)
                + gt_flat.sum(dim=1)
                - intersection_tensor
            )

            gt_area = gt_flat.sum(dim=1)
            pred_area = pred_flat.sum(dim=1)

            empty_gt = gt_area == 0
            empty_pred = pred_area == 0
            empty_both = empty_gt & empty_pred

            nonempty_gt = gt_area > 0
            false_positive_empty_gt = empty_gt & (~empty_pred)
            false_negative_nonempty_gt = nonempty_gt & empty_pred

            # Normal IoU
            iou_tensor = intersection_tensor / (union_tensor + 1e-6)

            # Important fix:
            # GT empty + Pred empty should be counted as correct.
            # GT empty + Pred non-empty remains IoU=0 naturally.
            iou_tensor[empty_both] = 1.0

            # Accumulate empty-mask statistics
            empty_gt_total += empty_gt.sum().double()
            empty_pred_total += empty_pred.sum().double()
            empty_both_total += empty_both.sum().double()
            nonempty_gt_total += nonempty_gt.sum().double()
            false_positive_empty_gt_total += false_positive_empty_gt.sum().double()
            false_negative_nonempty_gt_total += false_negative_nonempty_gt.sum().double()

            # Accumulate mIoU and oIoU
            # For empty-both samples, I=0 and U=0, so they do not affect oIoU.
            # They affect mIoU through iou_tensor=1.
            cum_I += intersection_tensor.sum().double()
            cum_U += union_tensor.sum().double()
            sum_iou += iou_tensor.sum().double()
            seg_total += batch_size

            for idx, thr in enumerate(eval_seg_iou_list):
                seg_correct[idx] += (iou_tensor >= thr).sum().double()

            # Optional visualization
            if rank == 0 and args.vis_path is not None:
                for j, img_name in enumerate(inputs.get("image_name", [])):
                    gt_mask_vis = (
                        gt_bin[j] * 255
                    ).detach().cpu().numpy().astype("uint8")
                    pred_mask_vis = (
                        pred_bin[j] * 255
                    ).detach().cpu().numpy().astype("uint8")

                    cv2.imwrite(
                        os.path.join(args.vis_path, f"{img_name}_gt.png"),
                        gt_mask_vis,
                    )
                    cv2.imwrite(
                        os.path.join(args.vis_path, f"{img_name}_pred.png"),
                        pred_mask_vis,
                    )

    # Distributed aggregation
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

    if rank == 0:
        total_samples = int(seg_total.item())

        if total_samples == 0:
            print(
                f"\n[WARNING] No samples were evaluated for split '{args.data_split}'. "
                "Please check --base_data_path and --data_split."
            )
            if distributed:
                dist.destroy_process_group()
            return 0.0, 0.0

        mIoU = (sum_iou / seg_total).item()

        # oIoU ignores empty-both naturally because I=0,U=0.
        if cum_U.item() > 0:
            overall_IoU = (cum_I / (cum_U + 1e-6)).item()
        else:
            # If all samples are empty-both, regard overall IoU as 1.
            # If there are empty GT + non-empty pred samples, cum_U would be > 0.
            overall_IoU = 1.0 if int(empty_both_total.item()) == total_samples else 0.0

        print("\n" + "=" * 40)
        print(f"Evaluation Results [{args.data_split}]")
        print("=" * 40)
        print(f"Mean IoU (mIoU):    {mIoU * 100:.2f}")
        print(f"Overall IoU (oIoU): {overall_IoU * 100:.2f}")
        print("-" * 40)

        for idx, thr in enumerate(eval_seg_iou_list):
            precision_k = (seg_correct[idx] / seg_total).item()
            print(f"Precision @ {thr:.1f}: {precision_k * 100:.2f}%")

        print("-" * 40)
        print(f"Total samples:                  {total_samples}")
        print(f"Non-empty GT samples:           {int(nonempty_gt_total.item())}")
        print(f"Empty GT samples:               {int(empty_gt_total.item())}")
        print(f"Empty Pred samples:             {int(empty_pred_total.item())}")
        print(f"Empty GT & Empty Pred samples:  {int(empty_both_total.item())}")
        print(f"Empty GT but Non-empty Pred FP: {int(false_positive_empty_gt_total.item())}")
        print(f"Non-empty GT but Empty Pred FN: {int(false_negative_nonempty_gt_total.item())}")
        print("=" * 40 + "\n")

    if distributed:
        dist.destroy_process_group()

    return (mIoU * 100.0, overall_IoU * 100.0) if rank == 0 else (0.0, 0.0)


if __name__ == "__main__":
    ss = evaluation()
    print(f"result_{ss}")
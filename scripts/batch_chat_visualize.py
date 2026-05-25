import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segearth_r1 import conversation as conversation_lib
from segearth_r1.eval_and_test.eval_dataset.change_val_dataset import ChangeValDataCollector, ChangeValDataset
from segearth_r1.mm_utils import get_model_name_from_path
from segearth_r1.model.builder import load_pretrained_model
from segearth_r1.utils import disable_torch_init


def parse_args(args):
    parser = argparse.ArgumentParser(description="Batch visualization for MTRefSeg-R1")
    parser.add_argument("--model_path", required=True, help="Path to trained checkpoint")
    parser.add_argument(
        "--mask_config",
        default="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
        help="Mask2Former config path",
    )
    parser.add_argument("--version", default="llava_phi")
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_seg_query", action="store_true", default=False)
    parser.add_argument("--model_map_name", default="segearth_r1")

    parser.add_argument("--json_dir", required=True, help="Directory of referring_expression/*.json")
    parser.add_argument("--image_dir_A", required=True, help="Directory of T1 images")
    parser.add_argument("--image_dir_B", required=True, help="Directory of T2 images")
    parser.add_argument("--mask_root", required=True, help="Masks root directory")
    parser.add_argument("--output_dir", default="./vis_output_batch")

    parser.add_argument("--max_samples", default=-1, type=int)
    parser.add_argument("--max_expressions", default=-1, type=int)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_canvas", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only_json", default=None)
    return parser.parse_args(args)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def maybe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff._-]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "expr"
    return text[:max_len]


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5):
    out = image_rgb.copy().astype(np.float32)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return out.clip(0, 255).astype(np.uint8)


def put_caption(image_rgb: np.ndarray, text: str):
    canvas = image_rgb.copy()
    words = text.split(" ")
    lines = []
    cur = ""
    for word in words:
        nxt = word if not cur else cur + " " + word
        if len(nxt) > 42:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        lines.append(cur)

    y = 24
    for line in lines[:4]:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return canvas


def save_canvas_dynamic(items, save_path: str):
    panels = [put_caption(img, cap) for img, cap in items]
    canvas = np.concatenate(panels, axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def normalize_expr(expr: str) -> str:
    return expr.strip().lower()


def infer_vis_mode(expr: str) -> str:
    e = normalize_expr(expr)

    disappear_keywords = [
        "disappear", "disappeared", "disappearing", "vanish", "vanished",
        "removed", "demolished", "gone", "lost", "deleted", "missing",
        "消失", "拆除", "去掉", "移除", "不见",
    ]
    appear_keywords = [
        "appear", "appeared", "appearing", "added", "new", "newly",
        "emerged", "built", "constructed", "出现", "新增", "新建",
    ]
    change_keywords = [
        "change", "changed", "changing", "became", "become", "turned",
        "repaved", "repainted", "converted", "transformed", "modified",
        "increased", "decreased", "expanded", "shrunk", "color", "colour",
        "状态改变", "变成", "变为", "改变", "变化", "颜色",
    ]

    has_disappear = any(k in e for k in disappear_keywords)
    has_appear = any(k in e for k in appear_keywords)
    has_change = any(k in e for k in change_keywords)

    if has_disappear and not has_appear and not has_change:
        return "A"
    if has_appear and not has_disappear and not has_change:
        return "B"
    if has_change or (has_disappear and has_appear):
        return "AB"
    return "AB"


def build_overlay_items(vis_mode: str, image_np_t1: np.ndarray, image_np_t2: np.ndarray, mask: np.ndarray, color=(255, 0, 0)):
    items = []
    if vis_mode in ("A", "AB"):
        items.append(("_A", overlay_mask(image_np_t1, mask, color=color, alpha=0.5), "on A/T1"))
    if vis_mode in ("B", "AB"):
        items.append(("_B", overlay_mask(image_np_t2, mask, color=color, alpha=0.5), "on B/T2"))
    return items


def align_ms_deform_attn_bias_dtype(model: torch.nn.Module) -> int:
    fixed = 0
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ != "MSDeformAttn":
                continue
            sampling_offsets = getattr(module, "sampling_offsets", None)
            if sampling_offsets is None or sampling_offsets.bias is None:
                continue
            weight = sampling_offsets.weight
            bias = sampling_offsets.bias
            if bias.dtype == weight.dtype and bias.device == weight.device:
                continue
            sampling_offsets.bias.data = bias.data.to(device=weight.device, dtype=weight.dtype)
            fixed += 1
    return fixed


def load_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def collect_jsons(args):
    if args.only_json is not None:
        return [args.only_json]
    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if args.max_samples > 0:
        json_paths = json_paths[: args.max_samples]
    return json_paths


def build_dataset(args, tokenizer):
    split_dir = os.path.dirname(args.json_dir)
    dataset = ChangeValDataset(
        base_data_path=split_dir,
        tokenizer=tokenizer,
        split="val",
        image_size=args.image_size,
    )
    collator = ChangeValDataCollector(tokenizer=tokenizer)
    sample_to_index = {}
    for idx, sample in enumerate(dataset.samples):
        key = sample["img_name"]
        sample_to_index[key] = idx
    return dataset, collator, sample_to_index


def make_single_batch(dataset, collator, sample_index: int):
    return collator([dataset[sample_index]])


def sigmoid_if_needed(pred_mask: torch.Tensor):
    pred = pred_mask.detach().cpu().float()
    if pred.min() >= 0 and pred.max() <= 1:
        return pred.numpy()
    return torch.sigmoid(pred).numpy()


def resize_binary_mask(mask: np.ndarray, out_hw):
    out_h, out_w = out_hw
    resized = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def load_model_for_vis(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    class BuilderArgs:
        def __init__(self, cli_args):
            self.seg_task = "referring"
            self.model_map_name = cli_args.model_map_name

    builder_args = BuilderArgs(args)
    tokenizer, model, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        model_args=builder_args,
        mask_config=args.mask_config,
        use_seg_query=args.use_seg_query,
        device=args.device,
    )
    model = model.to(device=args.device)
    fixed = align_ms_deform_attn_bias_dtype(model)
    if fixed > 0:
        print(f"[MTRefSegVis] aligned sampling_offsets.bias for {fixed} MSDeformAttn module(s).")
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    return tokenizer, model


def main(cli_args):
    args = parse_args(cli_args)
    maybe_mkdir(args.output_dir)

    tokenizer, model = load_model_for_vis(args)
    dataset, collator, sample_to_index = build_dataset(args, tokenizer)
    json_paths = collect_jsons(args)

    if len(json_paths) == 0:
        raise RuntimeError(f"No json files found in {args.json_dir}")

    print(f"Found {len(json_paths)} json files")

    model_dtype = next(model.parameters()).dtype

    for sample_idx, json_path in enumerate(json_paths):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_id = os.path.splitext(os.path.basename(json_path))[0]
        expressions = data.get("referring_expressions", [])
        if args.max_expressions > 0:
            expressions = expressions[: args.max_expressions]

        print(f"\n[{sample_idx + 1}/{len(json_paths)}] {os.path.basename(json_path)}")
        print(f"  num expressions: {len(expressions)}")

        sample_out_dir = os.path.join(args.output_dir, file_id)
        maybe_mkdir(sample_out_dir)

        image_path_t1 = os.path.join(args.image_dir_A, f"{file_id}.jpg")
        image_path_t2 = os.path.join(args.image_dir_B, f"{file_id}.jpg")
        if not os.path.exists(image_path_t1):
            matches = glob.glob(os.path.join(args.image_dir_A, file_id + ".*"))
            if matches:
                image_path_t1 = sorted(matches, key=natural_key)[0]
        if not os.path.exists(image_path_t2):
            matches = glob.glob(os.path.join(args.image_dir_B, file_id + ".*"))
            if matches:
                image_path_t2 = sorted(matches, key=natural_key)[0]

        try:
            image_np_t1 = load_rgb(image_path_t1)
            image_np_t2 = load_rgb(image_path_t2)
        except Exception as e:
            print(f"  image loading error: {e}")
            continue

        for expr_idx, expr in enumerate(expressions):
            sample_name = f"{file_id}_{expr_idx}"
            if sample_name not in sample_to_index:
                print(f"  missing dataset sample: {sample_name}")
                continue

            expr_slug = safe_name(expr)
            expr_prefix = os.path.join(sample_out_dir, f"{expr_idx:02d}_{expr_slug}")
            pred_mask_path = expr_prefix + "_pred_mask.png"

            if (not args.overwrite) and os.path.exists(pred_mask_path):
                print(f"  skip existing: {pred_mask_path}")
                continue

            batch = make_single_batch(dataset, collator, sample_to_index[sample_name])
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            images = batch["images"].to(args.device, dtype=model_dtype)
            images_t2 = batch["images_t2"].to(args.device, dtype=model_dtype)
            gt_masks = batch["masks"].to(args.device)
            refer_embedding_indices = batch.get("refer_embedding_indices")
            if refer_embedding_indices is not None:
                refer_embedding_indices = refer_embedding_indices.to(args.device)
            token_refer_id = batch.get("token_refer_id")
            if token_refer_id is not None:
                token_refer_id = [ids.to(args.device) for ids in token_refer_id]

            try:
                with torch.no_grad():
                    outputs = model.eval_seg(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        images_t2=images_t2,
                        masks=gt_masks,
                        token_refer_id=token_refer_id,
                        refer_embedding_indices=refer_embedding_indices,
                        labels=labels,
                        token_answer_id=None,
                        answer_embedding_indices=None,
                    )
            except Exception as e:
                print(f"  expression failed [{expr_idx}]: {e}")
                continue

            if len(outputs) == 0:
                print(f"  empty prediction [{expr_idx}] {expr}")
                continue

            pred_mask = outputs[0]["pred_masks"]
            if pred_mask.dim() > 2:
                pred_mask = pred_mask[0]
            pred_prob = sigmoid_if_needed(pred_mask)
            pred_bin = (pred_prob > 0.5).astype(np.uint8)
            pred_bin = resize_binary_mask(pred_bin, image_np_t1.shape[:2])
            cv2.imwrite(pred_mask_path, pred_bin * 255)

            vis_mode = infer_vis_mode(expr)
            pred_overlay_items = build_overlay_items(vis_mode, image_np_t1, image_np_t2, pred_bin, color=(255, 0, 0))
            for suffix, overlay_rgb, _ in pred_overlay_items:
                cv2.imwrite(expr_prefix + f"_pred_overlay{suffix}.png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            gt_overlay_items = []
            gt_mask = None
            if args.save_gt:
                gt_mask_tensor = batch["masks"][0]
                if gt_mask_tensor.dim() == 3 and gt_mask_tensor.shape[0] == 1:
                    gt_mask_tensor = gt_mask_tensor[0]
                gt_mask = (gt_mask_tensor.detach().cpu().numpy() > 0).astype(np.uint8)
                gt_mask = resize_binary_mask(gt_mask, image_np_t1.shape[:2])
                cv2.imwrite(expr_prefix + "_gt_mask.png", gt_mask * 255)
                gt_overlay_items = build_overlay_items(vis_mode, image_np_t1, image_np_t2, gt_mask, color=(0, 255, 0))
                for suffix, overlay_rgb, _ in gt_overlay_items:
                    cv2.imwrite(expr_prefix + f"_gt_overlay{suffix}.png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            with open(expr_prefix + "_meta.txt", "w", encoding="utf-8") as fw:
                fw.write(f"json_path: {json_path}\n")
                fw.write(f"img_A: {image_path_t1}\n")
                fw.write(f"img_B: {image_path_t2}\n")
                fw.write(f"expression: {expr}\n")
                fw.write("model_text_output: [eval_seg mode - no text generation]\n")
                fw.write(f"visualization_mode: {vis_mode}\n")
                fw.write(f"dataset_sample_name: {sample_name}\n")

            if args.save_canvas:
                canvas_items = [(image_np_t1, "T1 / earlier"), (image_np_t2, "T2 / later")]
                for _, overlay_rgb, title in pred_overlay_items:
                    canvas_items.append((overlay_rgb, f"Pred {title} | {expr}"))
                for _, overlay_rgb, title in gt_overlay_items:
                    canvas_items.append((overlay_rgb, f"GT {title}"))
                save_canvas_dynamic(canvas_items, expr_prefix + "_canvas.jpg")

            print(f"  saved [{expr_idx}] {expr_prefix} | vis_mode={vis_mode}")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])

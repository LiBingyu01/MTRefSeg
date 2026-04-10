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
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model import LisaGSVAForCausalLM, add_task_tokens, init_vision_seg_for_model
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from model.segment_anything.utils.transforms import ResizeLongestSide


def parse_args(args):
    parser = argparse.ArgumentParser(description="Batch visualization for dual-temporal LISA")

    parser.add_argument("--mllm_model_path", required=True, help="Path or HF id of checkpoint")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--log_base_dir", default="./outputs", type=str)
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=20, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument("--weight", default=None, type=str, help="Path to a trained weight .bin file")
    parser.add_argument("--segmentation_model_path", default=None, type=str, help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")

    parser.add_argument("--json_dir", required=True, help="Directory of referring_expression/*.json")
    parser.add_argument("--image_dir_A", required=True, help="Directory of T1 images")
    parser.add_argument("--image_dir_B", required=True, help="Directory of T2 images")
    parser.add_argument("--mask_root", default=None, help="Optional masks root directory")
    parser.add_argument("--output_dir", default="./vis_output_batch", help="Where to save results")

    parser.add_argument("--max_samples", default=-1, type=int, help="How many json files to process, -1 means all")
    parser.add_argument("--max_expressions", default=-1, type=int, help="How many expressions per json, -1 means all")
    parser.add_argument("--save_gt", action="store_true", help="Save GT mask/overlay when json contains mask info")
    parser.add_argument("--save_canvas", action="store_true", help="Save stitched canvas")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--only_json", default=None, help="Process only one json file")

    return parser.parse_args(args)



def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



def cast_tensor_precision(x, precision):
    if precision == "bf16":
        return x.bfloat16()
    elif precision == "fp16":
        return x.half()
    else:
        return x.float()



def prepare_single_image(image_path, clip_image_processor, transform, args):
    if not os.path.exists(image_path):
        raise FileNotFoundError("File not found: {}".format(image_path))

    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError("Failed to read image: {}".format(image_path))

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = clip_image_processor.preprocess(
        image_np, return_tensors="pt"
    )["pixel_values"][0].unsqueeze(0).cuda()
    image_clip = cast_tensor_precision(image_clip, args.precision)

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = preprocess(
        torch.from_numpy(image).permute(2, 0, 1).contiguous(), img_size=args.image_size
    ).unsqueeze(0).cuda()
    image = cast_tensor_precision(image, args.precision)

    return image_np, image_clip, image, resize_list, original_size_list



def build_prompt(user_prompt, use_mm_start_end):
    prompt = (
        DEFAULT_IMAGE_TOKEN
        + " is the earlier image, and \n"
        + DEFAULT_IMAGE_TOKEN
        + " is the later image.\n"
        + f"Compare the two images and segment the {user_prompt}."
    )
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return prompt



def evaluate_dual_time(
    model,
    input_ids,
    image_clip_t1,
    image_t1,
    resize_list_t1,
    original_size_list_t1,
    image_clip_t2,
    image_t2,
    resize_list_t2,
    original_size_list_t2,
):
    h, w = original_size_list_t2[0]
    dummy_label = torch.zeros(h, w, dtype=torch.float32, device=image_t2.device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    offset = torch.tensor([0, 1], dtype=torch.long, device=input_ids.device)

    outputs = model.model_forward(
        images_t1=image_t1,
        images_t2=image_t2,
        images_clip_t1=image_clip_t1,
        images_clip_t2=image_clip_t2,
        input_ids=input_ids,
        labels=input_ids,
        attention_masks=attention_mask,
        offset=offset,
        masks_list=[torch.zeros(1, h, w, dtype=torch.float32, device=image_t2.device)],
        label_list=[dummy_label],
        resize_list=[resize_list_t2[0]],
        inference=True,
    )
    return outputs["pred_masks"]



def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.mllm_model_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer, args = add_task_tokens(tokenizer, args)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16
    args.torch_dtype = torch_dtype

    # batch_chat_visualize is always single-GPU eval, no LoRA at inference
    args.eval_only = True

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.float16,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "seg_token_idx": args.seg_token_idx,
        "rej_token_idx": args.rej_token_idx,
        "segmentation_model_path": getattr(args, "segmentation_model_path", None),
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "tokenizer": tokenizer,
    }

    model = LisaGSVAForCausalLM.from_pretrained(
        args.mllm_model_path,
        low_cpu_mem_usage=True,
        **model_args,
        **kwargs,
    )

    # Use the same init path as main.py: initializes vision tower, SAM, LoRA,
    # resizes token embeddings, and freezes/unfreezes parameters consistently.
    model = init_vision_seg_for_model(model, tokenizer, args)

    # Load fine-tuned weight if specified (same as main.py --weight logic)
    if hasattr(args, "weight") and args.weight is not None:
        state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded trained weights from:", args.weight)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.float16,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    return model, tokenizer, clip_image_processor, transform



def safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff._-]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "expr"
    return text[:max_len]



def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5):
    out = image_rgb.copy()
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out_f = out.astype(np.float32)
    out_f[mask] = out_f[mask] * (1 - alpha) + color_arr * alpha
    return out_f.astype(np.uint8)



def put_caption(image_rgb: np.ndarray, text: str):
    canvas = image_rgb.copy()
    lines = []
    words = text.split(" ")
    cur = ""
    for w in words:
        nxt = w if not cur else cur + " " + w
        if len(nxt) > 42:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = nxt
    if cur:
        lines.append(cur)

    y = 24
    for line in lines[:4]:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return canvas



def resolve_mask(mask_info: Dict[str, Any], json_dir: str, mask_root: Optional[str], h: int, w: int):
    mask = np.zeros((h, w), dtype=np.uint8)
    loaded = False

    if isinstance(mask_info, dict) and "path" in mask_info:
        candidates = []
        raw_path = mask_info["path"]
        if os.path.isabs(raw_path):
            candidates.append(raw_path)
        else:
            candidates.append(os.path.join(json_dir, raw_path))
            if mask_root is not None:
                candidates.append(os.path.join(mask_root, raw_path))
                candidates.append(os.path.join(mask_root, os.path.basename(raw_path)))

        for p in candidates:
            if os.path.exists(p):
                m = cv2.imread(p, 0)
                if m is not None:
                    mask = (m > 0).astype(np.uint8)
                    loaded = True
                    break

    if (not loaded) and isinstance(mask_info, dict) and "polygons" in mask_info and mask_info["polygons"]:
        for poly in mask_info["polygons"]:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [pts], 1)
        loaded = True

    return mask if loaded else None



def normalize_expr(expr: str) -> str:
    return expr.strip().lower()



def infer_vis_mode(expr: str) -> str:
    e = normalize_expr(expr)

    disappear_keywords = [
        "disappear", "disappeared", "disappearing", "vanish", "vanished",
        "removed", "demolished", "gone", "lost", "deleted", "missing",
        "消失", "拆除", "去掉", "移除", "不见"
    ]
    appear_keywords = [
        "appear", "appeared", "appearing", "added", "new", "newly",
        "emerged", "built", "constructed", "出现", "新增", "新建"
    ]
    change_keywords = [
        "change", "changed", "changing", "became", "become", "turned",
        "repaved", "repainted", "converted", "transformed", "modified",
        "increased", "decreased", "expanded", "shrunk", "color", "colour",
        "状态改变", "变成", "变为", "改变", "变化", "颜色"
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



def save_canvas_dynamic(items, save_path):
    panels = [put_caption(img, cap) for img, cap in items]
    canvas = np.concatenate(panels, axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))



def run_one_expression(
    model,
    tokenizer,
    args,
    image_clip_t1,
    image_t1,
    resize_list_t1,
    original_size_list_t1,
    image_clip_t2,
    image_t2,
    resize_list_t2,
    original_size_list_t2,
    expr: str,
):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    prompt = build_prompt(expr, args.use_mm_start_end)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "Sure, it is [SEG]")
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    with torch.no_grad():
        pred_masks = evaluate_dual_time(
            model=model,
            input_ids=input_ids,
            image_clip_t1=image_clip_t1,
            image_t1=image_t1,
            resize_list_t1=resize_list_t1,
            original_size_list_t1=original_size_list_t1,
            image_clip_t2=image_clip_t2,
            image_t2=image_t2,
            resize_list_t2=resize_list_t2,
            original_size_list_t2=original_size_list_t2,
        )

    text_output = "[inference mode - no text generation]"
    return text_output, pred_masks



def collect_jsons(args):
    if args.only_json is not None:
        return [args.only_json]
    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if args.max_samples > 0:
        json_paths = json_paths[: args.max_samples]
    return json_paths



def maybe_mkdir(path):
    os.makedirs(path, exist_ok=True)



def main(cli_args):
    args = parse_args(cli_args)
    maybe_mkdir(args.output_dir)

    model, tokenizer, clip_image_processor, transform = load_model(args)
    json_paths = collect_jsons(args)

    if len(json_paths) == 0:
        raise RuntimeError("No json files found in {}".format(args.json_dir))

    print("Found {} json files".format(len(json_paths)))

    for sample_idx, json_path in enumerate(json_paths):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name_A = os.path.splitext(data["img_A"])[0] + ".jpg"
        img_name_B = os.path.splitext(data["img_B"])[0] + ".jpg"
        ref_expressions = data.get("referring_expressions", [])
        mask_infos = data.get("mask", None)

        if args.max_expressions > 0:
            ref_expressions = ref_expressions[: args.max_expressions]
            if isinstance(mask_infos, list):
                mask_infos = mask_infos[: args.max_expressions]

        image_path_t1 = os.path.join(args.image_dir_A, img_name_A)
        image_path_t2 = os.path.join(args.image_dir_B, img_name_B)

        print("\n[{}/{}] {}".format(sample_idx + 1, len(json_paths), os.path.basename(json_path)))
        print("  T1:", image_path_t1)
        print("  T2:", image_path_t2)
        print("  num expressions:", len(ref_expressions))

        try:
            image_np_t1, image_clip_t1, image_t1, resize_list_t1, original_size_list_t1 = prepare_single_image(image_path_t1, clip_image_processor, transform, args)
            image_np_t2, image_clip_t2, image_t2, resize_list_t2, original_size_list_t2 = prepare_single_image(image_path_t2, clip_image_processor, transform, args)
        except Exception as e:
            print("  image loading error:", e)
            continue

        sample_stem = os.path.splitext(os.path.basename(json_path))[0]
        sample_out_dir = os.path.join(args.output_dir, sample_stem)
        maybe_mkdir(sample_out_dir)

        for expr_idx, expr in enumerate(ref_expressions):
            expr_slug = safe_name(expr)
            expr_prefix = os.path.join(sample_out_dir, "{:02d}_{}".format(expr_idx, expr_slug))
            pred_mask_path = expr_prefix + "_pred_mask.png"

            if (not args.overwrite) and os.path.exists(pred_mask_path):
                print("  skip existing:", pred_mask_path)
                continue

            try:
                text_output, pred_masks = run_one_expression(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    image_clip_t1=image_clip_t1,
                    image_t1=image_t1,
                    resize_list_t1=resize_list_t1,
                    original_size_list_t1=original_size_list_t1,
                    image_clip_t2=image_clip_t2,
                    image_t2=image_t2,
                    resize_list_t2=resize_list_t2,
                    original_size_list_t2=original_size_list_t2,
                    expr=expr,
                )
            except Exception as e:
                print("  expression failed [{}]: {}".format(expr_idx, e))
                continue

            vis_mode = infer_vis_mode(expr)

            with open(expr_prefix + "_meta.txt", "w", encoding="utf-8") as fw:
                fw.write("json_path: {}\n".format(json_path))
                fw.write("img_A: {}\n".format(image_path_t1))
                fw.write("img_B: {}\n".format(image_path_t2))
                fw.write("expression: {}\n".format(expr))
                fw.write("model_text_output: {}\n".format(text_output))
                fw.write("visualization_mode: {}\n".format(vis_mode))

            if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
                print("  empty prediction [{}] {}".format(expr_idx, expr))
                continue

            pred_mask = pred_masks[0].detach().cpu().numpy()[0]
            pred_mask = (pred_mask > 0).astype(np.uint8)
            cv2.imwrite(pred_mask_path, pred_mask * 255)

            pred_overlay_items = build_overlay_items(vis_mode, image_np_t1, image_np_t2, pred_mask, color=(255, 0, 0))
            for suffix, overlay_rgb, _ in pred_overlay_items:
                cv2.imwrite(expr_prefix + "_pred_overlay{}.png".format(suffix), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            gt_overlay_items = []
            if args.save_gt and isinstance(mask_infos, list) and expr_idx < len(mask_infos):
                gt_mask = resolve_mask(
                    mask_infos[expr_idx],
                    json_dir=os.path.dirname(json_path),
                    mask_root=args.mask_root,
                    h=image_np_t2.shape[0],
                    w=image_np_t2.shape[1],
                )
                if gt_mask is not None:
                    cv2.imwrite(expr_prefix + "_gt_mask.png", gt_mask.astype(np.uint8) * 255)
                    gt_overlay_items = build_overlay_items(vis_mode, image_np_t1, image_np_t2, gt_mask, color=(0, 255, 0))
                    for suffix, overlay_rgb, _ in gt_overlay_items:
                        cv2.imwrite(expr_prefix + "_gt_overlay{}.png".format(suffix), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            if args.save_canvas:
                canvas_items = [(image_np_t1, "T1 / earlier"), (image_np_t2, "T2 / later")]
                for _, overlay_rgb, title in pred_overlay_items:
                    canvas_items.append((overlay_rgb, "Pred {} | {}".format(title, expr)))
                for _, overlay_rgb, title in gt_overlay_items:
                    canvas_items.append((overlay_rgb, "GT {}".format(title)))
                save_canvas_dynamic(canvas_items, expr_prefix + "_canvas.jpg")

            print("  saved [{}] {} | vis_mode={}".format(expr_idx, expr_prefix, vis_mode))

    print("\nDone. Results saved to {}".format(args.output_dir))



if __name__ == "__main__":
    main(sys.argv[1:])

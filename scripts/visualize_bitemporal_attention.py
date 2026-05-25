#!/usr/bin/env python3
"""
Standalone bi-temporal attention / fusion visualization for SegEarth-R1.

This script is intentionally non-invasive:
  - it does not modify training or evaluation entrypoints
  - it does not change any default forward behavior
  - it only loads the model, runs one sample in eval mode, and writes figures

Outputs:
  1. LLM self-attention maps from referring-expression tokens to T1 / T2 image tokens
  2. Temporal fusion response maps from the change-aware fusion module
  3. Optional predicted / GT mask overlays
  4. Metadata JSON and raw numpy arrays for later analysis

Example:
  python scripts/visualize_bitemporal_attention.py \
    --model-path /path/to/checkpoint \
    --base-data-path /path/to/change_dataset \
    --split val \
    --sample-index 0 \
    --output-dir ./attention_vis/sample0 \
    --mask-config segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --version llava_phi \
    --layer-indices mid,last
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segearth_r1 import conversation as conversation_lib
from segearth_r1.constants import (  # noqa: E402
    ANSWER_TOKEN_INDEX,
    IMAGE_TOKEN_INDEX,
    REFER_TOKEN_INDEX,
    SEG_TOKEN_INDEX,
)
from segearth_r1.eval_and_test.eval_dataset.change_val_dataset import (  # noqa: E402
    ChangeValDataCollector,
    ChangeValDataset,
)
from segearth_r1.mm_utils import get_model_name_from_path  # noqa: E402
from segearth_r1.model.builder import load_pretrained_model  # noqa: E402
from segearth_r1.utils import disable_torch_init  # noqa: E402


PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize bi-temporal multimodal attention for SegEarth-R1.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--base-data-path", required=True, help="Dataset root or split directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualization outputs.")
    parser.add_argument(
        "--mask-config",
        default="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
        help="Mask2Former config.",
    )
    parser.add_argument("--version", default="llava_phi", help="Conversation template name.")
    parser.add_argument("--split", default="val", help="Dataset split name if base-data-path is a dataset root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in ChangeValDataset.")
    parser.add_argument("--sample-name", default="", help="Optional image_name to select instead of sample-index.")
    parser.add_argument("--image-size", type=int, default=1024, help="Preprocess image size.")
    parser.add_argument("--use-seg-query", action="store_true", help="Use seg query path if your checkpoint requires it.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0 or cpu.")
    parser.add_argument(
        "--cast-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float32",
        help="Optional dtype cast after loading the model.",
    )
    parser.add_argument(
        "--layer-indices",
        default="mid,last",
        help="Comma-separated layer ids. Supports ints plus first/mid/last/all.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for heatmaps.")
    parser.add_argument("--model-map-name", default="segearth_r1", help="Model map name used by builder.")
    parser.add_argument("--save-layerwise", action="store_true", help="Save per-layer attention overlays in addition to mean maps.")
    parser.add_argument("--logit-threshold", type=float, default=0.5, help="Threshold for predicted mask visualization.")
    return parser.parse_args()


def natural_key(text: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def resolve_dtype(name: str, device: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def align_ms_deform_attn_bias_dtype(model: torch.nn.Module) -> int:
    # Visualization can keep the checkpoint's mixed dtypes, but MSDeformAttn.forward
    # assumes sampling_offsets.weight/bias already match and reassigns the parameter.
    # PyTorch 2.4 rejects that reassignment, so align the bias once in this process.
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


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * PIXEL_STD + PIXEL_MEAN
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def mask_tensor_to_gray(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().float().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def resize_map(attn_map: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(attn_map.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def overlay_heatmap(base_rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = normalize_map(heatmap)
    colored = cv2.applyColorMap(np.uint8(heatmap * 255.0), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out = ((1.0 - alpha) * base_rgb.astype(np.float32) + alpha * colored.astype(np.float32)).clip(0, 255)
    return out.astype(np.uint8)


def overlay_binary_mask(base_rgb: np.ndarray, mask_gray: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    mask = (mask_gray > 127).astype(np.uint8)
    color_arr = np.zeros_like(base_rgb, dtype=np.uint8)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    out = base_rgb.copy().astype(np.float32)
    out[mask > 0] = (1.0 - alpha) * out[mask > 0] + alpha * color_arr[mask > 0].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def add_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    canvas = image_rgb.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def infer_grid_shape(num_tokens: int) -> Tuple[int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, side

    best_h, best_w = 1, num_tokens
    best_gap = num_tokens
    for h in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        gap = abs(w - h)
        if gap < best_gap:
            best_h, best_w = h, w
            best_gap = gap
    return best_h, best_w


def resolve_layer_indices(spec: str, num_layers: int) -> List[int]:
    items = [item.strip() for item in spec.split(",") if item.strip()]
    if not items:
        return [num_layers - 1]
    if len(items) == 1 and items[0] == "all":
        return list(range(num_layers))

    resolved: List[int] = []
    for item in items:
        if item == "first":
            idx = 0
        elif item == "mid":
            idx = num_layers // 2
        elif item == "last":
            idx = num_layers - 1
        else:
            idx = int(item)
            if idx < 0:
                idx += num_layers
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index out of range: {item} for {num_layers} layers.")
        if idx not in resolved:
            resolved.append(idx)
    return resolved


def select_sample_index(dataset: ChangeValDataset, sample_name: str, sample_index: int) -> int:
    if sample_name:
        for idx, sample in enumerate(dataset.samples):
            if sample["img_name"] == sample_name:
                return idx
        raise ValueError(f"sample-name={sample_name} not found in dataset.")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample-index={sample_index} out of range for dataset size={len(dataset)}.")
    return sample_index


def make_single_batch(dataset: ChangeValDataset, sample_index: int, tokenizer) -> Dict[str, object]:
    collator = ChangeValDataCollector(tokenizer=tokenizer)
    return collator([dataset[sample_index]])


def chunk_original_ids(input_ids_1d: torch.Tensor) -> List[Tuple[str, int]]:
    chunks: List[Tuple[str, int]] = []
    current_tokens: List[int] = []
    for token in input_ids_1d.tolist():
        if token >= 0:
            current_tokens.append(token)
            continue
        if current_tokens:
            chunks.append(("text", len(current_tokens)))
            current_tokens = []
        chunks.append(("special", token))
    if current_tokens:
        chunks.append(("text", len(current_tokens)))
    return chunks


def compute_expanded_spans(
    original_input_ids: torch.Tensor,
    image_lengths: Sequence[int],
    refer_length: int,
    answer_length: int = 0,
    seg_query_length: int = 0,
    use_seg_query: bool = False,
) -> Dict[str, Tuple[int, int]]:
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    image_idx = 0

    for chunk_type, value in chunk_original_ids(original_input_ids):
        if chunk_type == "text":
            cursor += value
            continue

        token = value
        if token == IMAGE_TOKEN_INDEX:
            if image_idx >= len(image_lengths):
                raise RuntimeError("Encountered more <image> tokens than provided image lengths.")
            start = cursor
            end = cursor + image_lengths[image_idx]
            key = "image_t1" if image_idx == 0 else "image_t2" if image_idx == 1 else f"image_{image_idx}"
            spans[key] = (start, end)
            cursor = end
            image_idx += 1
        elif token == REFER_TOKEN_INDEX:
            spans["refer"] = (cursor, cursor + refer_length)
            cursor += refer_length
        elif token == ANSWER_TOKEN_INDEX:
            spans["answer"] = (cursor, cursor + answer_length)
            cursor += answer_length
        elif token == SEG_TOKEN_INDEX:
            if use_seg_query and seg_query_length > 0:
                spans["seg_query"] = (cursor, cursor + seg_query_length)
                cursor += seg_query_length
        else:
            raise RuntimeError(f"Unsupported special token id in replay: {token}")

    spans["sequence"] = (0, cursor)
    return spans


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def build_summary_grid(images: Sequence[np.ndarray], cols: int = 4, tile_size: int = 512) -> np.ndarray:
    tiles = []
    for image in images:
        tile = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), cols):
        row = tiles[start:start + cols]
        if len(row) < cols:
            blank = np.zeros_like(tiles[0])
            while len(row) < cols:
                row.append(blank)
        rows.append(cv2.hconcat([cv2.cvtColor(tile, cv2.COLOR_RGB2BGR) for tile in row]))
    grid_bgr = cv2.vconcat(rows)
    return cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity_map(prototype: torch.Tensor, feature_map: torch.Tensor) -> np.ndarray:
    proto = F.normalize(prototype.detach().float(), dim=0)
    feat = F.normalize(feature_map.detach().float(), dim=0)
    sim = (feat * proto[:, None, None]).sum(dim=0)
    return sim.cpu().numpy().astype(np.float32)


def postprocess_mask_outputs(model, mask_outputs: Dict[str, torch.Tensor | None], images: torch.Tensor) -> List[Dict[str, torch.Tensor | None]]:
    seg_cls_results = mask_outputs["pred_SEG_logits"]
    mask_pred_results = F.interpolate(
        mask_outputs["pred_masks"],
        size=(images.shape[-2], images.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    processed_results = []
    if seg_cls_results is None:
        seg_cls_results = [None] * mask_pred_results.shape[0]

    for seg_cls_result, mask_pred_result in zip(seg_cls_results, mask_pred_results):
        if seg_cls_result is not None:
            seg_cls_result = seg_cls_result.to(mask_pred_result)
            result = model.SEG_instance_inference(seg_cls_result.float(), mask_pred_result.float())
        else:
            result = model.SEG_instance_inference(None, mask_pred_result.float())
        processed_results.append(result)

    return processed_results


def select_query_index(final_mask_logits: torch.Tensor, final_seg_logits: torch.Tensor | None) -> Tuple[int, Dict[str, List[float]]]:
    mask_logits = final_mask_logits.detach().float()
    mask_probs = mask_logits.sigmoid()
    pred_masks = (mask_logits > 0).float()
    mask_scores = (
        (mask_probs.flatten(1) * pred_masks.flatten(1)).sum(1)
        / (pred_masks.flatten(1).sum(1) + 1e-6)
    )

    if final_seg_logits is not None:
        seg_scores = final_seg_logits.detach().float().sigmoid().flatten()
        combined_scores = seg_scores * mask_scores
    else:
        seg_scores = torch.ones_like(mask_scores)
        combined_scores = mask_scores

    query_index = int(combined_scores.argmax().item())
    score_info = {
        "seg_scores": seg_scores.cpu().tolist(),
        "mask_scores": mask_scores.cpu().tolist(),
        "combined_scores": combined_scores.cpu().tolist(),
    }
    return query_index, score_info


def cross_attention_forward_with_weights(
    layer,
    tgt: torch.Tensor,
    memory: torch.Tensor,
    memory_mask: torch.Tensor | None = None,
    memory_key_padding_mask: torch.Tensor | None = None,
    pos: torch.Tensor | None = None,
    query_pos: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer.normalize_before:
        tgt2 = layer.norm(tgt)
        attn_out, attn_weights = layer.multihead_attn(
            query=layer.with_pos_embed(tgt2, query_pos),
            key=layer.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + layer.dropout(attn_out)
        return tgt, attn_weights

    attn_out, attn_weights = layer.multihead_attn(
        query=layer.with_pos_embed(tgt, query_pos),
        key=layer.with_pos_embed(memory, pos),
        value=memory,
        attn_mask=memory_mask,
        key_padding_mask=memory_key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    tgt = tgt + layer.dropout(attn_out)
    tgt = layer.norm(tgt)
    return tgt, attn_weights


def trace_mask_decoder(
    predictor,
    multi_scale_features: Sequence[torch.Tensor],
    mask_features: torch.Tensor,
    seg_query: torch.Tensor | None = None,
    seg_embedding: torch.Tensor | None = None,
) -> Dict[str, object]:
    if getattr(predictor, "seg_concat", False):
        raise NotImplementedError("Visualization trace currently supports the non-concat mask decoder path only.")

    src = []
    pos = []
    size_list = []
    for i in range(predictor.num_feature_levels):
        size_list.append(tuple(int(v) for v in multi_scale_features[i].shape[-2:]))
        pos_i = predictor.pe_layer(multi_scale_features[i], None).flatten(2).to(multi_scale_features[i].dtype)
        src_i = predictor.input_proj[i](multi_scale_features[i]).flatten(2) + predictor.level_embed.weight[i][None, :, None]
        pos.append(pos_i.permute(2, 0, 1))
        src.append(src_i.permute(2, 0, 1))

    _, batch_size, _ = src[0].shape
    conditioned_with_seg_logits = seg_query is not None
    if seg_query is not None:
        query_embed = predictor.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = seg_query.permute(1, 0, 2)
    elif seg_embedding is not None:
        query_embed = torch.zeros(
            predictor.new_query_embed.weight.shape[0],
            batch_size,
            predictor.new_query_embed.weight.shape[-1],
            device=seg_embedding.device,
            dtype=seg_embedding.dtype,
        )
        output = seg_embedding.permute(1, 0, 2)
    else:
        query_embed = predictor.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = predictor.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)

    stage_query_states = [output]
    stage_seg_logits = []
    stage_mask_logits = []
    stage_attn_masks = []
    cross_attn_weights = []
    cross_attn_spatial_shapes = []

    if conditioned_with_seg_logits:
        seg_cls, _, outputs_mask, attn_mask = predictor.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
            SEG_embedding=seg_embedding,
            class_name_embedding=None,
        )
    else:
        seg_cls, _, outputs_mask, attn_mask = predictor.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
            SEG_embedding=None,
            class_name_embedding=None,
        )
    stage_seg_logits.append(seg_cls)
    stage_mask_logits.append(outputs_mask)
    stage_attn_masks.append(attn_mask)

    for i in range(predictor.num_layers):
        level_index = i % predictor.num_feature_levels
        attn_mask = stage_attn_masks[-1].clone()
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        output, attn_weights = cross_attention_forward_with_weights(
            predictor.transformer_cross_attention_layers[i],
            output,
            src[level_index],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[level_index],
            query_pos=query_embed,
        )
        cross_attn_weights.append(attn_weights)
        cross_attn_spatial_shapes.append(size_list[level_index])

        output = predictor.transformer_self_attention_layers[i](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed,
        )
        output = predictor.transformer_ffn_layers[i](output)

        if conditioned_with_seg_logits:
            seg_cls, _, outputs_mask, attn_mask = predictor.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % predictor.num_feature_levels],
                SEG_embedding=seg_embedding,
                class_name_embedding=None,
            )
        else:
            seg_cls, _, outputs_mask, attn_mask = predictor.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % predictor.num_feature_levels],
                SEG_embedding=None,
                class_name_embedding=None,
            )

        stage_query_states.append(output)
        stage_seg_logits.append(seg_cls)
        stage_mask_logits.append(outputs_mask)
        stage_attn_masks.append(attn_mask)

    return {
        "pred_SEG_logits": stage_seg_logits[-1],
        "pred_masks": stage_mask_logits[-1],
        "stage_query_states": stage_query_states,
        "stage_seg_logits": stage_seg_logits,
        "stage_mask_logits": stage_mask_logits,
        "cross_attn_weights": cross_attn_weights,
        "cross_attn_spatial_shapes": cross_attn_spatial_shapes,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    class BuilderArgs:
        def __init__(self, args: argparse.Namespace):
            self.seg_task = "referring"
            self.model_map_name = args.model_map_name

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

    cast_dtype = resolve_dtype(args.cast_dtype, args.device)
    if cast_dtype is not None:
        model = model.to(device=args.device, dtype=cast_dtype)
    else:
        model = model.to(device=args.device)
    fixed_ms_deform_attn = align_ms_deform_attn_bias_dtype(model)
    if fixed_ms_deform_attn > 0:
        print(
            f"[AttentionVis] aligned sampling_offsets.bias for {fixed_ms_deform_attn} "
            "MSDeformAttn module(s)."
        )
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    dataset = ChangeValDataset(
        base_data_path=args.base_data_path,
        tokenizer=tokenizer,
        split=args.split,
        image_size=args.image_size,
    )

    sample_index = select_sample_index(dataset, args.sample_name, args.sample_index)
    sample_info = dataset.samples[sample_index]
    batch = make_single_batch(dataset, sample_index, tokenizer=tokenizer)

    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    labels = batch["labels"].to(args.device)
    images = batch["images"].to(args.device)
    images_t2 = batch["images_t2"].to(args.device)
    refer_embedding_indices = batch.get("refer_embedding_indices")
    if refer_embedding_indices is not None:
        refer_embedding_indices = refer_embedding_indices.to(args.device)
    token_refer_id = batch.get("token_refer_id")
    if token_refer_id is not None:
        token_refer_id = [ids.to(args.device) for ids in token_refer_id]
    gt_mask = batch.get("masks")
    if gt_mask is not None:
        gt_mask = gt_mask.to(args.device)

    model_dtype = next(model.parameters()).dtype
    images_for_model = images.to(dtype=model_dtype)
    images_t2_for_model = images_t2.to(dtype=model_dtype)

    with torch.no_grad():
        image_features_t1 = model.encode_images(images_for_model)
        image_features_t2 = model.encode_images(images_t2_for_model)

        _, attn_mask_mm, pkv_mm, inputs_embeds, labels_mm, _, refer_indices_mm = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images_for_model,
            token_refer_id=token_refer_id,
            token_answer_id=None,
            refer_embedding_indices=refer_embedding_indices,
            answer_embedding_indices=None,
            use_seg_query=args.use_seg_query,
            images_t2=images_t2_for_model,
        )

        llm_outputs = model.model(
            input_ids=None,
            attention_mask=attn_mask_mm,
            past_key_values=pkv_mm,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

        pred_outputs = model.eval_seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images_for_model,
            images_t2=images_t2_for_model,
            masks=gt_mask,
            token_refer_id=token_refer_id,
            refer_embedding_indices=refer_embedding_indices,
            labels=labels,
            output_attentions=False,
        )

        feat_dict_t1 = model.get_vision_tower_feature(images_for_model)
        feat_dict_t2 = model.get_vision_tower_feature(images_t2_for_model)

    attentions = llm_outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions. Check whether the loaded checkpoint supports output_attentions.")

    num_layers = len(attentions)
    selected_layers = resolve_layer_indices(args.layer_indices, num_layers)

    image_token_lengths = [int(image_features_t1.shape[1]), int(image_features_t2.shape[1])]
    refer_length = int(token_refer_id[0].shape[0]) if token_refer_id is not None else 0
    seg_query_length = int(model.seg_query.shape[0]) if args.use_seg_query and hasattr(model, "seg_query") else 0
    spans = compute_expanded_spans(
        original_input_ids=batch["input_ids"][0],
        image_lengths=image_token_lengths,
        refer_length=refer_length,
        answer_length=0,
        seg_query_length=seg_query_length,
        use_seg_query=args.use_seg_query,
    )

    if refer_indices_mm is not None:
        actual_refer_positions = torch.where(refer_indices_mm[0].bool())[0]
        if actual_refer_positions.numel() > 0:
            spans["refer"] = (int(actual_refer_positions[0].item()), int(actual_refer_positions[-1].item()) + 1)

    seq_len_expected = spans["sequence"][1]
    seq_len_actual = int(inputs_embeds.shape[1])
    if seq_len_expected != seq_len_actual:
        raise RuntimeError(
            f"Expanded sequence length mismatch: replay={seq_len_expected}, actual={seq_len_actual}. "
            "The visualization script would misalign spans, so it stops here."
        )

    if "image_t1" not in spans or "image_t2" not in spans or "refer" not in spans:
        raise RuntimeError(f"Failed to recover required spans from the multimodal sequence: {spans}")

    t1_start, t1_end = spans["image_t1"]
    t2_start, t2_end = spans["image_t2"]
    refer_start, refer_end = spans["refer"]
    refer_positions = torch.arange(refer_start, refer_end, device=attentions[0].device)

    grid_t1 = infer_grid_shape(t1_end - t1_start)
    grid_t2 = infer_grid_shape(t2_end - t2_start)

    base_t1 = tensor_to_rgb(batch["images"][0])
    base_t2 = tensor_to_rgb(batch["images_t2"][0])
    base_pair = ((base_t1.astype(np.float32) + base_t2.astype(np.float32)) * 0.5).clip(0, 255).astype(np.uint8)

    raw_arrays: Dict[str, np.ndarray] = {}
    llm_maps_t1: List[np.ndarray] = []
    llm_maps_t2: List[np.ndarray] = []

    for layer_idx in selected_layers:
        layer_attn = attentions[layer_idx][0]
        layer_attn = layer_attn[:, refer_positions, :]

        map_t1 = layer_attn[:, :, t1_start:t1_end].mean(dim=(0, 1)).detach().cpu().float().numpy().reshape(grid_t1)
        map_t2 = layer_attn[:, :, t2_start:t2_end].mean(dim=(0, 1)).detach().cpu().float().numpy().reshape(grid_t2)

        llm_maps_t1.append(map_t1)
        llm_maps_t2.append(map_t2)
        raw_arrays[f"llm_layer_{layer_idx:02d}_t1"] = map_t1
        raw_arrays[f"llm_layer_{layer_idx:02d}_t2"] = map_t2

        if args.save_layerwise:
            up_t1 = resize_map(map_t1, base_t1.shape[:2])
            up_t2 = resize_map(map_t2, base_t2.shape[:2])
            write_rgb(
                output_dir / f"llm_attn_layer_{layer_idx:02d}_t1.png",
                add_label(overlay_heatmap(base_t1, up_t1, args.alpha), f"Layer {layer_idx} -> T1"),
            )
            write_rgb(
                output_dir / f"llm_attn_layer_{layer_idx:02d}_t2.png",
                add_label(overlay_heatmap(base_t2, up_t2, args.alpha), f"Layer {layer_idx} -> T2"),
            )

    mean_map_t1 = np.mean(np.stack(llm_maps_t1, axis=0), axis=0)
    mean_map_t2 = np.mean(np.stack(llm_maps_t2, axis=0), axis=0)
    raw_arrays["llm_mean_t1"] = mean_map_t1
    raw_arrays["llm_mean_t2"] = mean_map_t2

    llm_overlay_t1 = add_label(
        overlay_heatmap(base_t1, resize_map(mean_map_t1, base_t1.shape[:2]), args.alpha),
        f"LLM Attn Mean -> T1 ({selected_layers})",
    )
    llm_overlay_t2 = add_label(
        overlay_heatmap(base_t2, resize_map(mean_map_t2, base_t2.shape[:2]), args.alpha),
        f"LLM Attn Mean -> T2 ({selected_layers})",
    )
    write_rgb(output_dir / "llm_attn_mean_t1.png", llm_overlay_t1)
    write_rgb(output_dir / "llm_attn_mean_t2.png", llm_overlay_t2)

    fusion_summary: Dict[str, Dict[str, float]] = {}
    fusion_overlays: Dict[str, np.ndarray] = {}

    for level_name in sorted(model.change_feature_fusion.blocks.keys(), key=natural_key):
        block = model.change_feature_fusion.blocks[level_name]
        feat_t1 = feat_dict_t1[level_name]
        feat_t2 = feat_dict_t2[level_name]

        with torch.no_grad():
            t1_proj = block.t1_proj(feat_t1)
            t2_proj = block.t2_proj(feat_t2)
            signed_diff = t2_proj - t1_proj
            abs_diff = signed_diff.abs()
            diff_feat = block.diff_encoder(torch.cat([signed_diff, abs_diff], dim=1))
            fused = block.fuse(torch.cat([t1_proj, t2_proj, abs_diff, diff_feat], dim=1))
            gate = block.gate(torch.cat([t1_proj, t2_proj], dim=1))
            shortcut = 0.5 * (t1_proj + t2_proj)
            out = block.out_act(block.out_norm(fused + gate * diff_feat + shortcut))

        abs_diff_map = abs_diff[0].abs().mean(dim=0).detach().cpu().float().numpy()
        diff_feat_map = diff_feat[0].abs().mean(dim=0).detach().cpu().float().numpy()
        out_map = out[0].abs().mean(dim=0).detach().cpu().float().numpy()

        raw_arrays[f"fusion_{level_name}_abs_diff"] = abs_diff_map
        raw_arrays[f"fusion_{level_name}_diff_feat"] = diff_feat_map
        raw_arrays[f"fusion_{level_name}_out"] = out_map

        fusion_summary[level_name] = {
            "gate_mean": float(gate.mean().detach().cpu().item()),
            "gate_std": float(gate.std().detach().cpu().item()),
            "abs_diff_mean": float(abs_diff.mean().detach().cpu().item()),
            "out_mean": float(out.mean().detach().cpu().item()),
        }

        abs_diff_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(abs_diff_map, base_pair.shape[:2]), args.alpha),
            f"{level_name} abs-diff",
        )
        out_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(out_map, base_pair.shape[:2]), args.alpha),
            f"{level_name} fused-output",
        )
        write_rgb(output_dir / f"fusion_{level_name}_abs_diff.png", abs_diff_overlay)
        write_rgb(output_dir / f"fusion_{level_name}_out.png", out_overlay)
        fusion_overlays[f"{level_name}_abs_diff"] = abs_diff_overlay
        fusion_overlays[f"{level_name}_out"] = out_overlay

    with torch.no_grad():
        hidden_states = llm_outputs.last_hidden_state
        decoder_features = model.build_decoder_features(images_for_model, images_t2_for_model)
        seg_embedding = model.build_seg_embedding_from_hidden(
            hidden_states,
            refer_indices_mm,
            decoder_features,
        )
        if args.use_seg_query:
            raise NotImplementedError("Visualization trace currently supports the non-seg-query path only.")
        seg_query = None
        mask_features, _, multi_scale_features = model.pixel_decoder.forward_features(decoder_features)
        trace_outputs = trace_mask_decoder(
            model.predictor,
            multi_scale_features,
            mask_features,
            seg_query=seg_query,
            seg_embedding=seg_embedding,
        )
        trace_pred_outputs = postprocess_mask_outputs(
            model,
            {
                "pred_SEG_logits": trace_outputs["pred_SEG_logits"],
                "pred_masks": trace_outputs["pred_masks"],
            },
            images_for_model,
        )

    seg_embedding_vector = seg_embedding[0, 0]
    seg_embedding_mask_similarity = cosine_similarity_map(seg_embedding_vector, mask_features[0])
    raw_arrays["seg_embedding_mask_features_similarity"] = seg_embedding_mask_similarity
    seg_embedding_mask_similarity_overlay = add_label(
        overlay_heatmap(base_pair, resize_map(seg_embedding_mask_similarity, base_pair.shape[:2]), args.alpha),
        "SEG_embedding -> mask_features similarity",
    )
    write_rgb(output_dir / "seg_embedding_mask_features_similarity.png", seg_embedding_mask_similarity_overlay)

    seg_embedding_multiscale_vis: List[np.ndarray] = []
    seg_embedding_multiscale_stats: Dict[str, Dict[str, float]] = {}
    for level_idx, feature_map in enumerate(multi_scale_features):
        sim_map = cosine_similarity_map(seg_embedding_vector, feature_map[0])
        raw_arrays[f"seg_embedding_multiscale_{level_idx:02d}_similarity"] = sim_map
        sim_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(sim_map, base_pair.shape[:2]), args.alpha),
            f"SEG_embedding -> multiscale[{level_idx}] similarity",
        )
        write_rgb(output_dir / f"seg_embedding_multiscale_{level_idx:02d}_similarity.png", sim_overlay)
        seg_embedding_multiscale_vis.append(sim_overlay)
        seg_embedding_multiscale_stats[f"multiscale_{level_idx:02d}"] = {
            "mean": float(sim_map.mean()),
            "std": float(sim_map.std()),
            "min": float(sim_map.min()),
            "max": float(sim_map.max()),
        }

    stage0_mask_logits = trace_outputs["stage_mask_logits"][0][0, 0].detach().cpu().float().numpy()
    raw_arrays["seg_embedding_stage0_mask_logits"] = stage0_mask_logits.astype(np.float32)
    stage0_mask_prob = sigmoid_np(stage0_mask_logits)
    raw_arrays["seg_embedding_stage0_mask_prob"] = stage0_mask_prob.astype(np.float32)
    seg_embedding_stage0_overlay = add_label(
        overlay_heatmap(base_pair, resize_map(stage0_mask_prob, base_pair.shape[:2]), args.alpha),
        "SEG_embedding -> stage0 mask",
    )
    write_rgb(output_dir / "seg_embedding_stage0_mask.png", seg_embedding_stage0_overlay)

    final_mask_logits = trace_outputs["pred_masks"][0]
    final_seg_logits = trace_outputs["pred_SEG_logits"][0] if trace_outputs["pred_SEG_logits"] is not None else None
    selected_query_index, query_score_info = select_query_index(final_mask_logits, final_seg_logits)

    selected_query_mask_vis: List[np.ndarray] = []
    selected_query_cross_attn_vis: List[np.ndarray] = []
    stage_mask_stats: List[Dict[str, float]] = []
    cross_attn_stats: List[Dict[str, float]] = []

    for stage_idx, stage_mask_logits in enumerate(trace_outputs["stage_mask_logits"]):
        selected_mask_logits = stage_mask_logits[0, selected_query_index].detach().cpu().float().numpy()
        raw_arrays[f"decoder_stage_{stage_idx:02d}_query_mask_logits"] = selected_mask_logits.astype(np.float32)
        selected_mask_prob = sigmoid_np(selected_mask_logits)
        raw_arrays[f"decoder_stage_{stage_idx:02d}_query_mask_prob"] = selected_mask_prob.astype(np.float32)

        mask_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(selected_mask_prob, base_pair.shape[:2]), args.alpha),
            f"Decoder stage {stage_idx} mask q={selected_query_index}",
        )
        write_rgb(output_dir / f"decoder_stage_{stage_idx:02d}_query_mask.png", mask_overlay)
        selected_query_mask_vis.append(mask_overlay)
        stage_mask_stats.append(
            {
                "stage": float(stage_idx),
                "min": float(selected_mask_prob.min()),
                "max": float(selected_mask_prob.max()),
                "mean": float(selected_mask_prob.mean()),
                "std": float(selected_mask_prob.std()),
            }
        )

    for layer_idx, (attn_weights, spatial_shape) in enumerate(
        zip(trace_outputs["cross_attn_weights"], trace_outputs["cross_attn_spatial_shapes"])
    ):
        cur_attn = attn_weights[0, :, selected_query_index, :].mean(dim=0).detach().cpu().float().numpy()
        cur_map = cur_attn.reshape(spatial_shape)
        raw_arrays[f"decoder_cross_attn_{layer_idx:02d}"] = cur_map.astype(np.float32)
        cross_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(cur_map, base_pair.shape[:2]), args.alpha),
            f"Cross-attn {layer_idx} q={selected_query_index} {spatial_shape}",
        )
        write_rgb(output_dir / f"decoder_cross_attn_{layer_idx:02d}.png", cross_overlay)
        selected_query_cross_attn_vis.append(cross_overlay)
        cross_attn_stats.append(
            {
                "layer": float(layer_idx),
                "mean": float(cur_map.mean()),
                "std": float(cur_map.std()),
                "max": float(cur_map.max()),
            }
        )

    local_vision = decoder_features["res5"].flatten(2).permute(0, 2, 1)
    local_vision = model.local_project(local_vision)
    refer_hidden_list = model.get_SEG_embedding(hidden_states, refer_indices_mm, return_all=True)
    refer_hidden = refer_hidden_list[0][1:]
    refer_hidden_proj = model.text_projector(refer_hidden)
    refer_hidden_proj = F.normalize(refer_hidden_proj, dim=-1)
    local_vision_norm = F.normalize(local_vision[0], dim=-1)
    refer_token_similarity = torch.matmul(refer_hidden_proj, local_vision_norm.transpose(0, 1))
    refer_token_similarity = refer_token_similarity.detach().cpu().float().numpy()
    sim_h, sim_w = decoder_features["res5"].shape[-2:]

    refer_similarity_vis: List[np.ndarray] = []
    for token_idx, token_sim in enumerate(refer_token_similarity):
        sim_map = token_sim.reshape(sim_h, sim_w)
        raw_arrays[f"refer_token_{token_idx:02d}_res5_similarity"] = sim_map.astype(np.float32)
        sim_overlay = add_label(
            overlay_heatmap(base_pair, resize_map(sim_map, base_pair.shape[:2]), args.alpha),
            f"Refer token {token_idx} -> res5 local vision",
        )
        write_rgb(output_dir / f"refer_token_{token_idx:02d}_res5_similarity.png", sim_overlay)
        refer_similarity_vis.append(sim_overlay)

    mean_refer_similarity = refer_token_similarity.mean(axis=0).reshape(sim_h, sim_w)
    raw_arrays["refer_res5_similarity_mean"] = mean_refer_similarity.astype(np.float32)
    mean_refer_similarity_overlay = add_label(
        overlay_heatmap(base_pair, resize_map(mean_refer_similarity, base_pair.shape[:2]), args.alpha),
        "Refer tokens mean -> res5 local vision",
    )
    write_rgb(output_dir / "refer_res5_similarity_mean.png", mean_refer_similarity_overlay)

    pred_mask = pred_outputs[0]["pred_masks"]
    if pred_mask.dim() > 2:
        pred_mask = pred_mask[0]
    pred_prob = pred_mask.detach().cpu().float().numpy()
    if pred_prob.min() < 0.0 or pred_prob.max() > 1.0:
        pred_prob = 1.0 / (1.0 + np.exp(-pred_prob))
    pred_gray = (pred_prob > args.logit_threshold).astype(np.uint8) * 255
    pred_overlay = add_label(
        overlay_binary_mask(base_pair, pred_gray, color=(255, 64, 64), alpha=0.45),
        "Pred mask",
    )
    write_rgb(output_dir / "pred_mask_overlay.png", pred_overlay)
    raw_arrays["pred_mask_prob"] = pred_prob.astype(np.float32)

    trace_pred_mask = trace_pred_outputs[0]["pred_masks"]
    if trace_pred_mask.dim() > 2:
        trace_pred_mask = trace_pred_mask[0]
    trace_pred_prob = trace_pred_mask.detach().cpu().float().numpy()
    trace_pred_overlay = add_label(
        overlay_binary_mask(base_pair, (trace_pred_prob > 0.5).astype(np.uint8) * 255, color=(255, 196, 64), alpha=0.45),
        f"Trace pred mask q={selected_query_index}",
    )
    write_rgb(output_dir / "trace_pred_mask_overlay.png", trace_pred_overlay)
    raw_arrays["trace_pred_mask_prob"] = trace_pred_prob.astype(np.float32)

    if gt_mask is not None:
        gt_gray = mask_tensor_to_gray(batch["masks"][0])
        gt_overlay_t1 = add_label(
            overlay_binary_mask(base_t1, gt_gray, color=(64, 255, 64), alpha=0.45),
            "GT mask @ T1",
        )
        gt_overlay_t2 = add_label(
            overlay_binary_mask(base_t2, gt_gray, color=(64, 255, 64), alpha=0.45),
            "GT mask @ T2",
        )
        write_rgb(output_dir / "gt_mask_overlay_t1.png", gt_overlay_t1)
        write_rgb(output_dir / "gt_mask_overlay_t2.png", gt_overlay_t2)
        write_rgb(output_dir / "gt_mask_overlay.png", build_summary_grid([gt_overlay_t1, gt_overlay_t2], cols=2, tile_size=512))
        raw_arrays["gt_mask"] = (gt_gray > 127).astype(np.float32)
    else:
        gt_overlay_t1 = add_label(base_t1, "GT mask unavailable @ T1")
        gt_overlay_t2 = add_label(base_t2, "GT mask unavailable @ T2")
        write_rgb(output_dir / "gt_mask_overlay_t1.png", gt_overlay_t1)
        write_rgb(output_dir / "gt_mask_overlay_t2.png", gt_overlay_t2)
        write_rgb(output_dir / "gt_mask_overlay.png", build_summary_grid([gt_overlay_t1, gt_overlay_t2], cols=2, tile_size=512))

    t1_labeled = add_label(base_t1, "T1 image")
    t2_labeled = add_label(base_t2, "T2 image")

    preferred_fusion_key = "res5_out" if "res5_out" in fusion_overlays else sorted(fusion_overlays.keys(), key=natural_key)[-1]
    preferred_diff_key = "res5_abs_diff" if "res5_abs_diff" in fusion_overlays else sorted(fusion_overlays.keys(), key=natural_key)[0]

    summary_grid = build_summary_grid(
        [
            gt_overlay_t1,
            gt_overlay_t2,
            selected_query_cross_attn_vis[0] if selected_query_cross_attn_vis else fusion_overlays[preferred_fusion_key],
            selected_query_mask_vis[0] if selected_query_mask_vis else fusion_overlays[preferred_diff_key],
        ],
        cols=4,
        tile_size=512,
    )
    write_rgb(output_dir / "summary_grid.png", summary_grid)

    write_rgb(output_dir / "image_t1.png", t1_labeled)
    write_rgb(output_dir / "image_t2.png", t2_labeled)

    refer_text = tokenizer.decode(token_refer_id[0].detach().cpu().tolist(), skip_special_tokens=False) if token_refer_id is not None else ""
    meta = {
        "model_path": args.model_path,
        "base_data_path": args.base_data_path,
        "split": args.split,
        "sample_index": sample_index,
        "sample_name": sample_info["img_name"],
        "image_a_path": sample_info["img_A_path"],
        "image_b_path": sample_info["img_B_path"],
        "mask_path": sample_info["mask_path"],
        "expression": sample_info["expression"],
        "refer_text_decoded": refer_text,
        "selected_layers": selected_layers,
        "num_layers": num_layers,
        "image_token_lengths": image_token_lengths,
        "image_token_grid_t1": grid_t1,
        "image_token_grid_t2": grid_t2,
        "spans": {key: [int(v[0]), int(v[1])] for key, v in spans.items()},
        "fusion_summary": fusion_summary,
        "seg_embedding_multiscale_stats": seg_embedding_multiscale_stats,
        "selected_query_index": selected_query_index,
        "selected_query_scores": query_score_info,
        "decoder_stage_mask_stats": stage_mask_stats,
        "decoder_cross_attn_stats": cross_attn_stats,
    }

    expression_text = sample_info["expression"].strip()
    expression_lines = [
        f"expression: {expression_text}",
        f"refer_text_decoded: {refer_text}",
        f"sample_name: {sample_info['img_name']}",
    ]
    (output_dir / "expression.txt").write_text("\n".join(expression_lines) + "\n", encoding="utf-8")
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_npz(output_dir / "attention_arrays.npz", raw_arrays)
    print(f"[AttentionVis] saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

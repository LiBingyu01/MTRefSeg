# train_stage1.py
# Stage-1 training for MTRefSeg-R1:
# freeze LLM / no LoRA, train bi-temporal understanding modules only.

import os
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

# Reuse most components from the original train.py
import segearth_r1.train.train as base_train

warnings.filterwarnings("ignore")


@dataclass
class Stage1Arguments:
    """
    Extra arguments for Stage-1 training.
    Stage 1 is designed to learn bi-temporal/change-aware visual modeling
    with a pure visual Swin + MaskFormer objective.
    """
    stage1_train_mm_projector: bool = field(
        default=False,
        metadata={"help": "Train multimodal projector in Stage 1. Usually False for visual-only pretraining."},
    )
    stage1_train_mask_decoder: bool = field(
        default=True,
        metadata={"help": "Train pixel decoder and mask transformer decoder in Stage 1."},
    )
    stage1_train_vision_tower: bool = field(
        default=True,
        metadata={"help": "Train Swin backbone in Stage 1."},
    )
    stage1_train_temporal_modules: bool = field(
        default=True,
        metadata={"help": "Train temporal fusion/change-aware modules if they exist."},
    )
    stage1_train_seg_projectors: bool = field(
        default=False,
        metadata={"help": "Train SEG/referring projection modules in Stage 1. Usually False for visual-only pretraining."},
    )
    stage1_train_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to train token embeddings. Default False to keep LLM frozen."},
    )
    stage1_train_lm_head: bool = field(
        default=False,
        metadata={"help": "Whether to train lm_head. Default False to keep LLM frozen."},
    )
    stage1_verbose_trainable: bool = field(
        default=True,
        metadata={"help": "Print trainable parameters after applying Stage-1 freezing."},
    )


def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def enable_named_parameter(model, name: str):
    """
    Enable a standalone nn.Parameter by name, e.g. seg_query.
    """
    if hasattr(model, name):
        param = getattr(model, name)
        if isinstance(param, torch.nn.Parameter):
            param.requires_grad = True


def enable_module_if_exists(model, module_name: str):
    """
    Enable a child module if it exists.
    """
    if hasattr(model, module_name):
        module = getattr(model, module_name)
        if isinstance(module, torch.nn.Module):
            set_requires_grad(module, True)


def apply_stage1_freezing(model, model_args, training_args, stage1_args):
    """
    Stage 1 policy:
    - No LLM LoRA.
    - Freeze full model first.
    - Then unfreeze only modules related to bi-temporal visual adaptation,
      mask decoding, and multimodal projection.
    """

    # Force disable LoRA in Stage 1.
    if getattr(training_args, "lora_enable", False):
        print("[Stage1] WARNING: lora_enable=True was passed, but Stage 1 disables LoRA.")
    training_args.lora_enable = False

    # Freeze everything first: LLM, vision tower, projector, mask decoder, etc.
    model.requires_grad_(False)

    # 1. Train Swin backbone if requested.
    if stage1_args.stage1_train_vision_tower and hasattr(model.get_model(), "vision_tower"):
        set_requires_grad(model.get_model().vision_tower, True)
        print("[Stage1] Trainable: model.model.vision_tower")

    # 2. Train mm_projector if explicitly requested.
    if stage1_args.stage1_train_mm_projector:
        if hasattr(model.get_model(), "mm_projector"):
            set_requires_grad(model.get_model().mm_projector, True)
            print("[Stage1] Trainable: model.model.mm_projector")

    # 3. Train mask decoder modules.
    if stage1_args.stage1_train_mask_decoder:
        enable_module_if_exists(model, "pixel_decoder")
        enable_module_if_exists(model, "predictor")
        enable_named_parameter(model, "seg_query")

        if hasattr(model, "pixel_decoder"):
            print("[Stage1] Trainable: pixel_decoder")
        if hasattr(model, "predictor"):
            print("[Stage1] Trainable: predictor")
        if hasattr(model, "seg_query"):
            print("[Stage1] Trainable: seg_query")

    # 4. Train temporal / bi-temporal modules if you added them.
    # These names cover common implementations:
    #   bi_temporal_fusion, temporal_fusion, change_fusion, directional_fusion.
    if stage1_args.stage1_train_temporal_modules:
        temporal_module_names = [
            "bi_temporal_fusion",
            "temporal_fusion",
            "change_fusion",
            "directional_fusion",
            "language_guided_temporal_fusion",
            "change_aware_fusion",
            "change_feature_fusion",
        ]
        for name in temporal_module_names:
            if hasattr(model, name):
                enable_module_if_exists(model, name)
                print(f"[Stage1] Trainable: {name}")

    # 5. Train SEG/referring projection modules only if explicitly requested.
    if stage1_args.stage1_train_seg_projectors:
        seg_projector_names = [
            "seg_query_projector",
            "origin_SEG_token_projector",
            "local_project",
            "change_type_head",
            "change_head",
        ]
        for name in seg_projector_names:
            if hasattr(model, name):
                enable_module_if_exists(model, name)
                print(f"[Stage1] Trainable: {name}")

    # Keep heavier text/seg fusion projectors frozen in Stage 1.
    forced_frozen_names = [
        "text_projector",
        "SEG_token_projector",
        "d_layers",
    ]
    for name in forced_frozen_names:
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, torch.nn.Module):
                set_requires_grad(module, False)
                print(f"[Stage1] Frozen: {name}")

    # 6. Optional: train embeddings / lm_head.
    # Default is False to keep language model frozen.
    if stage1_args.stage1_train_embeddings:
        if hasattr(model.get_model(), "embed_tokens"):
            set_requires_grad(model.get_model().embed_tokens, True)
            print("[Stage1] Trainable: model.model.embed_tokens")

    if stage1_args.stage1_train_lm_head:
        if hasattr(model, "lm_head"):
            set_requires_grad(model.lm_head, True)
            print("[Stage1] Trainable: lm_head")

    if not stage1_args.stage1_train_vision_tower and hasattr(model.get_model(), "vision_tower"):
        set_requires_grad(model.get_model().vision_tower, False)
        print("[Stage1] Frozen: vision_tower")

    # 7. Ensure LLM backbone remains frozen.
    # Note: mm_projector / vision_tower are child modules under model.model,
    # so we already re-enabled mm_projector above after freezing all.
    if hasattr(model, "model"):
        for name, p in model.model.named_parameters():
            if "mm_projector" not in name and "vision_tower" not in name:
                p.requires_grad = False

    # Re-enable mm_projector again after LLM freeze loop.
    if stage1_args.stage1_train_mm_projector and hasattr(model.get_model(), "mm_projector"):
        set_requires_grad(model.get_model().mm_projector, True)

    print_trainable_summary(model)


def print_trainable_summary(model):
    trainable = 0
    total = 0

    print("\n================ Stage1 Trainable Parameters ================")
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print(f"[Trainable] {name} | shape={tuple(p.shape)} | numel={n}")
    print("-------------------------------------------------------------")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable ratio:  {100.0 * trainable / max(total, 1):.4f}%")
    print("=============================================================\n")


def train():
    parser = transformers.HfArgumentParser(
        (
            base_train.ModelArguments,
            base_train.DataArguments,
            base_train.TrainingArguments,
            Stage1Arguments,
        )
    )
    model_args, data_args, training_args, stage1_args = parser.parse_args_into_dataclasses()

    base_train.local_rank = training_args.local_rank

    if data_args.dataset_type != "change_detection_stage1":
        print(
            f"[Stage1] Overriding dataset_type from {data_args.dataset_type} to change_detection_stage1 "
            "for visual-only pretraining."
        )
        data_args.dataset_type = "change_detection_stage1"
    if data_args.val_data_path is not None:
        data_args.val_dataset_type = "change_detection_stage1"

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    mask_cfg = base_train.get_mask_config(config=model_args.mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task

    bnb_model_from_pretrained_args = {}

    print("using model segearth_r1")
    print("[Stage1] LLM LoRA is disabled. Training focuses on bi-temporal visual adaptation.")

    model = base_train.segearth_r1.from_pretrained(
        model_args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        cache_dir=training_args.cache_dir,
        use_seg_query=model_args.use_seg_query,
        **bnb_model_from_pretrained_args,
    )
    model.set_temporal_fusion_type(model_args.temporal_fusion_type)
    model.reset_change_module_parameters()

    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt)

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        base_train.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    if model_args.version in base_train.conversation_lib.conv_templates:
        base_train.conversation_lib.default_conversation = (
            base_train.conversation_lib.conv_templates[model_args.version]
        )
    else:
        base_train.conversation_lib.default_conversation = (
            base_train.conversation_lib.conv_templates["vicuna_v1"]
        )

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(
        SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
        EOS=tokenizer.eos_token_id,
    )

    # Apply Stage-1 freezing AFTER all modules and special tokens are initialized.
    apply_stage1_freezing(model, model_args, training_args, stage1_args)

    data_module = base_train.make_unify_datamodule(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    training_args.dataloader_drop_last = True

    callbacks = []
    if data_args.val_data_path is not None:
        if data_args.dataset_type == "change_detection_stage1":
            val_dataset = base_train.ChangeDetectionDataset(
                base_data_path=data_args.val_data_path,
                tokenizer=tokenizer,
                split="val",
                visual_only=True,
            )
            val_collator = base_train.ChangeDataCollector(tokenizer=tokenizer)
        else:
            val_dataset = base_train.ChangeValDataset(
                base_data_path=data_args.val_data_path,
                tokenizer=tokenizer,
                split="val",
            )
            val_collator = base_train.ChangeValDataCollector(tokenizer=tokenizer)
        callbacks.append(
            base_train.ChangeDetectionEvalCallback(
                val_dataset=val_dataset,
                val_collator=val_collator,
                eval_batch_size=1,
                dataloader_num_workers=2,
                max_eval_samples=(
                    data_args.max_eval_samples
                    if data_args.max_eval_samples is not None
                    else 5000
                ),
            )
        )
        print(f"[Stage1] ChangeDetectionEvalCallback registered, val samples={len(val_dataset)}")

    trainer = base_train.LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks if callbacks else None,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # Stage 1 should not save LoRA because LoRA is disabled.
    base_train.safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
    )


if __name__ == "__main__":
    train()

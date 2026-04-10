# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from .losses import dice_loss, sigmoid_ce_loss


class LisaGSVAMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
        else:
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
            self.init_seg_and_proj(self.config)

    def init_seg_and_proj(self, config):
        # SAM
        builder_sam = build_sam_vit_h if "sam_vit_h" in self.segmentation_model_path else \
            build_sam_vit_l if "sam_vit_l" in self.segmentation_model_path else build_sam_vit_b
        self.visual_model = builder_sam(self.segmentation_model_path)
        # Projection layer for SAM
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

class LisaGSVAModel(LisaGSVAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.seg_token_idx = kwargs.get("seg_token_idx", 0)

class BiTemporalFusion(nn.Module):
  def __init__(self, in_channels=256):
    super().__init__()
    # 降维
    self.project = nn.Sequential(
      nn.Linear(in_channels*3, in_channels),
      nn.ReLU(inplace=True)
    )
    # 融合后的处理
    self.fusion_conv = nn.Sequential(
      nn.Linear(in_channels, in_channels),
      nn.ReLU(inplace=True),
      nn.Linear(in_channels, in_channels)
    )

  def forward(self, emb1, emb2):
    # [B, C, H, W] -> [B, H, W, C]
    emb1 = emb1.permute(0, 2, 3, 1)
    emb2 = emb2.permute(0, 2, 3, 1)
    diff = torch.abs(emb1 - emb2)
    fused = self.project(torch.cat([emb1, emb2, diff], dim=-1))
    x = self.fusion_conv(fused)
    x = x.permute(0, 3, 1, 2)
    return x

class LisaGSVAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        use_mm_start_end = kwargs.pop("use_mm_start_end", True)
        vision_tower = kwargs.pop("vision_tower", "openai/clip-vit-large-patch14")
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        self.seg_token_idx = kwargs.pop("seg_token_idx", 0)
        self.rej_token_idx = kwargs.pop("rej_token_idx", 0)
        self.llm_tokenizer = kwargs.pop("tokenizer", None)
        
        train_mask_decoder = kwargs.pop("train_mask_decoder", None)
        out_dim = kwargs.pop("out_dim", None)
        segmentation_model_path = kwargs.pop("segmentation_model_path", None)

        # 2. Conditionally update the config (your original logic)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = use_mm_start_end
            config.mm_vision_tower = vision_tower

        # 3. Safe to call the base class now! kwargs is clean.
        super().__init__(config, **kwargs)

        # 4. Reassign and initialize your custom model
        if train_mask_decoder is not None:
            self.train_mask_decoder = train_mask_decoder
            
        # We pass the popped variables back in explicitly for LisaGSVAMetaModel
        self.model = LisaGSVAModel(
            config, 
            seg_token_idx=self.seg_token_idx,
            train_mask_decoder=train_mask_decoder,
            out_dim=out_dim,
            segmentation_model_path=segmentation_model_path,
            **kwargs
        )
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.Np = self.model.vision_tower.num_patches
        self.post_init()
        
        self.fusion_module = BiTemporalFusion(in_channels=256)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def pad_sequnce_and_stack(self, input_ids, attention_masks, labels):
        input_ids = nn.utils.rnn.pad_sequence(input_ids, True, 0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, True, False)
        labels = nn.utils.rnn.pad_sequence(labels, True, IGNORE_INDEX)
        return input_ids, attention_masks, labels
    def model_forward(
        self,
        images_t1: torch.FloatTensor,
        images_t2: torch.FloatTensor,
        images_clip_t1: torch.FloatTensor,
        images_clip_t2: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        do_segs: List[bool] = None,
        inference: bool = False,
        reeval: bool = False,
        **kwargs,
    ):
        device, dtype = images_clip_t1.device, images_clip_t1.dtype
        
        # 1. 提取双路特征并融合
        image_embeddings_t1 = self.get_visual_embs(images_t1)
        image_embeddings_t2 = self.get_visual_embs(images_t2)
        
        # for inference
        fused_image_embeddings = (image_embeddings_t1 + image_embeddings_t2) / 2.0
        
        # for training
        # fused_image_embeddings = self.fusion_module(image_embeddings_t1, image_embeddings_t2)
        
        batch_size = image_embeddings_t1.shape[0] # 图像的数量
        assert batch_size == len(offset) - 1
        
        if inference: # Segmentation Eval
            n_batch = 1
            length = input_ids.shape[0] # 问答序列的数量
            assert images_clip_t1.shape[0] == 1
            images_clip_t1_extend = images_clip_t1.expand(length, -1, -1, -1).contiguous()
            images_clip_t2_extend = images_clip_t2.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            output_ids = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images_t1=images_clip_t1_extend[: end_i - start_i],
                    images_t2=images_clip_t2_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True
                )
                output_hidden_states.append(output_i.hidden_states)
                for k in range(length):
                    pred_output_ids = output_i.logits[k].argmax(dim=1)
                    pred_ids = input_ids[k].clone()
                    
                    img_indices = (pred_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                    for img_idx in img_indices.flip(dims=[0]):
                        pred_ids = torch.cat([
                            pred_ids[0:img_idx], 
                            torch.zeros(self.Np, device=device, dtype=torch.int64), 
                            pred_ids[img_idx + 1:]
                        ], dim=0)
                        
                    seg_index_gt = (pred_ids == self.seg_token_idx).nonzero(as_tuple=True)[0]
                    seg_index_pred = seg_index_gt - 1
                    pred_seg_values = torch.where((pred_output_ids[seg_index_pred] != self.seg_token_idx), self.rej_token_idx, self.seg_token_idx)
                    
                    rej_index_gt = (pred_ids == self.rej_token_idx).nonzero(as_tuple=True)[0]
                    rej_index_pred = rej_index_gt - 1
                    pred_rej_values = torch.where((pred_output_ids[rej_index_pred] != self.rej_token_idx), self.seg_token_idx, self.rej_token_idx)
                    
                    pred_ids[seg_index_gt] = pred_seg_values
                    pred_ids[rej_index_gt] = pred_rej_values
                    output_ids.append(pred_ids)
                
                if reeval:
                    input_ids[input_ids == self.rej_token_idx] = self.seg_token_idx
                    output_i_reeval = super().forward(
                        images_t1=images_clip_t1_extend[: end_i - start_i],
                        images_t2=images_clip_t2_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True
                    )
                    output_hidden_states[-1] = output_i_reeval.hidden_states
                    torch.cuda.empty_cache()
                    
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states = [output_hidden_states_level]
            output = None
            
        else: # Training 
            images_clip_t1_list = []
            images_clip_t2_list = []
            for i in range(len(offset) - 1): 
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_t1_i = (images_clip_t1[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
                images_clip_t2_i = (images_clip_t2[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
                images_clip_t1_list.append(images_clip_t1_i)
                images_clip_t2_list.append(images_clip_t2_i)
                
            images_clip_t1_cat = torch.cat(images_clip_t1_list, dim=0)
            images_clip_t2_cat = torch.cat(images_clip_t2_list, dim=0)
            
            output = super().forward(
                images_t1=images_clip_t1_cat,
                images_t2=images_clip_t2_cat,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True
            )
            output_hidden_states = output.hidden_states

        # 2. 获取输出特征
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        num_queries = input_ids.shape[0]
        seq_len = last_hidden_state.shape[1]
        
        seg_token_mask = torch.zeros((num_queries, seq_len), dtype=torch.bool, device=device)
        rej_token_mask = torch.zeros((num_queries, seq_len), dtype=torch.bool, device=device)

        for q_idx in range(num_queries):
            cur_input_ids = input_ids[q_idx]
            image_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            # 映射 [SEG]
            seg_indices = torch.where(cur_input_ids == self.seg_token_idx)[0]
            for seg_idx in seg_indices:
                images_before = (image_indices < seg_idx).sum().item()
                physical_idx = seg_idx.item() + images_before * (self.Np - 1)
                if 0 <= physical_idx - 1 < seq_len:
                    seg_token_mask[q_idx, physical_idx - 1] = True
                    
            # 映射 [REJ]
            rej_indices = torch.where(cur_input_ids == self.rej_token_idx)[0]
            for rej_idx in rej_indices:
                images_before = (image_indices < rej_idx).sum().item()
                physical_idx = rej_idx.item() + images_before * (self.Np - 1)
                if 0 <= physical_idx - 1 < seq_len:
                    rej_token_mask[q_idx, physical_idx - 1] = True

        mask_list_comp = []
        for q_idx in range(num_queries):
            this_seg_token_m = seg_token_mask[q_idx].long() * 2
            this_rej_token_m = rej_token_mask[q_idx].long() * 1
            this_seg_rej = this_seg_token_m + this_rej_token_m
            gathered_idx = this_seg_rej.nonzero(as_tuple=True)[0]
            this_seg_rej = this_seg_rej[gathered_idx].eq(2).nonzero(as_tuple=True)[0]
            mask_list_comp.append(this_seg_rej)        
        
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset], dim=0
        )     
        
        num_pred_embs = len(seg_token_offset) - 1 
        
        pred_embeddings_ = []
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        
        # 建立 Query 到 Image 的精确映射字典
        mask_img_map = [(t >= offset).long().argmin().item() - 1 for t in range(num_pred_embs)]
        
        # 3. SAM 解码器预测 (严格按 Query 循环)
        pred_masks = []
        pred_ious = []
        
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(dtype)
            
            img_idx = mask_img_map[i]
            
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=fused_image_embeddings[img_idx].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[img_idx],
                original_size=label_list[img_idx].shape
            )
            pred_masks.append(pred_mask[:, 0])
            pred_ious.append(iou_predictions[:, 0])
            
        model_output = output
        gt_masks = masks_list
        
        # === 核心修复区：在返回前，将扁平的 Query 级别预测统一打包为 Image 级别 ===
        grouped_pred_masks = []
        grouped_mask_list_comp = []
        for k in range(len(offset) - 1):
            begin, end = offset[k], offset[k + 1]
            select_preds = pred_masks[begin:end]
            
            if len(select_preds) == 1:
                grouped_pred_masks.append(select_preds[0]) # [1, H, W]
            elif len(select_preds) > 1:
                grouped_pred_masks.append(torch.cat(select_preds, dim=0)) # [N, H, W]
            else:
                grouped_pred_masks.append(torch.empty((0, label_list[k].shape[0], label_list[k].shape[1]), device=device))
                
            if not inference:
                select_comps = mask_list_comp[begin:end]
                if len(select_comps) == 1:
                    grouped_mask_list_comp.extend(select_comps)
                else:
                    grouped_mask_list_comp.append(select_comps)

        pred_masks = grouped_pred_masks
        
        if not inference: 
            mask_list_comp = grouped_mask_list_comp
            pred_masks_= []
            for b_idx in range(batch_size):
                L, h, w = pred_masks[b_idx].shape
                if L == 0:
                    pred_masks_.append(pred_masks[b_idx])
                    continue
                this_pred_masks_ = torch.zeros_like(gt_masks[b_idx], dtype=torch.float32)
                if isinstance(mask_list_comp[b_idx], torch.Tensor):
                    this_pred_masks_[mask_list_comp[b_idx]] = pred_masks[b_idx]
                else:
                    assert isinstance(mask_list_comp[b_idx], list) and len(mask_list_comp[b_idx]) == L
                    for j in range(L):
                        this_pred_masks_[j] = pred_masks[b_idx][j:j + 1][mask_list_comp[b_idx][j]]
                pred_masks_.append(this_pred_masks_)
            pred_masks = pred_masks_
        
        for b in range(batch_size):
            if pred_masks[b].shape[0] > 0 and gt_masks[b].shape[0] > 0:
                assert pred_masks[b].shape[1:] == gt_masks[b].shape[1:], f"b_idx: {b}, pm.shape: {pred_masks[b].shape}, gm.shape: {gt_masks[b].shape}"
                
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "output_ids": output_ids
            }
            
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
            
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss = ce_loss + mask_loss
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss
        }
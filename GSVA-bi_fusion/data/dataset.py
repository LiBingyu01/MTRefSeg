import glob
import os
import random
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything import ResizeLongestSide

from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN,DEFAULT_IMAGE_TOKEN_T1, DEFAULT_IMAGE_TOKEN_T2)


def collate_fn(
    batch, 
    tokenizer=None, 
    conv_type="llava_v1", 
    use_mm_start_end=True, 
    local_rank=-1
):
    image_path_list = []
    # 修改：分别存储 T1 和 T2 的图像
    images_t1_list = []
    images_t2_list = []
    images_clip_t1_list = []
    images_clip_t2_list = []
    
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []

    for (
        image_path,
        images_t1,
        images_t2,
        images_clip_t1,
        images_clip_t2, 
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference
    ) in batch:
        image_path_list.append(image_path)
        
        # 收集双图数据
        images_t1_list.append(images_t1)
        images_t2_list.append(images_t2)
        images_clip_t1_list.append(images_clip_t1)
        images_clip_t2_list.append(images_clip_t2)
        
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    # --- 以下处理 Conversation Token 的逻辑保持不变 ---
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            # T1
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN_T1, replace_token
            )
            # T2
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN_T2, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
    return {
        "image_paths": image_path_list,
        # 修改：返回双图 Batch
        "images_t1": torch.stack(images_t1_list, dim=0),
        "images_t2": torch.stack(images_t2_list, dim=0),
        "images_clip_t1": torch.stack(images_clip_t1_list, dim=0),
        "images_clip_t2": torch.stack(images_clip_t2_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

from .changeref_seg_dataset import ChangeReferDataset 

class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="change_refer", 
        change_refer_seg_data='changeref|train',
        **kwargs,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.change_refer_ds = ChangeReferDataset(
            base_image_dir=base_image_dir,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            samples_per_epoch=samples_per_epoch,
            precision=precision,
            image_size=image_size,
            num_classes_per_sample=num_classes_per_sample,
            exclude_val=exclude_val,
            refer_seg_data=change_refer_seg_data # 你的数据集名称
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.change_refer_ds[idx]


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        # --- 解析数据路径 ---
        dataset_name, split = val_dataset.split("|")
        self.path2split = os.path.join(base_image_dir, dataset_name, split)
        json_pattern = os.path.join(self.path2split, "referring_expression", "*.json")
        self.json_paths = glob.glob(json_pattern)
             
        print(f"Validation set size: {len(self.json_paths)}")

    def __len__(self):
        return len(self.json_paths)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        with open(json_path, "r") as f:
            data = json.load(f)

        # --- 1. A 和 B ---
        img_name_A = os.path.splitext(data["img_A"])[0] + ".jpg"
        img_name_B = os.path.splitext(data["img_B"])[0] + ".jpg"
        path_A = os.path.join(self.path2split, "A", img_name_A)
        path_B = os.path.join(self.path2split, "B", img_name_B)

        image_A = cv2.imread(path_A)
        image_B = cv2.imread(path_B)
        if image_A is None or image_B is None:
            print(f"Error loading images: {path_A} or {path_B}")
            return self.__getitem__(0)
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

        # --- 2. CLIP 预处理 (双图) ---
        image_clip_A = self.clip_image_processor.preprocess(image_A, 
                                    return_tensors="pt")["pixel_values"][0]
        image_clip_B = self.clip_image_processor.preprocess(image_B, 
                                    return_tensors="pt")["pixel_values"][0]

        # SAM transform
        image_A_sam = self.transform.apply_image(image_A)
        image_B_sam = self.transform.apply_image(image_B)
        resize = image_A_sam.shape[:2]
        image_A_tensor = self.preprocess(torch.from_numpy(image_A_sam).permute(2, 0, 1).contiguous())
        image_B_tensor = self.preprocess(torch.from_numpy(image_B_sam).permute(2, 0, 1).contiguous())

        # Construct Conversation
        ref_exps = data["referring_expressions"]
        masks_data = data["mask"]
        questions = []
        answers = [] # 验证时不需要真实答案，或者用于计算指标
        masks = []
        
        ori_h, ori_w = image_A.shape[:2]
        
        for i, text in enumerate(ref_exps):
            questions.append(DEFAULT_IMAGE_TOKEN_T1 + " is the earlier image, and " + 
                             DEFAULT_IMAGE_TOKEN_T2 + " is the later image." + "\n" +
                             f"\n Compare the two images and segment {text}. Please output segmentation mask.")
            answers.append("[SEG].")

            m_data = masks_data[i]
            m = np.zeros((ori_h, ori_w), dtype=np.uint8)
            mask_loaded = False
            if "path" in m_data:
                mask_path = m_data["path"]
                # 这里的 self.base_image_dir 对应参考代码中的 self.path2split
                if not os.path.exists(mask_path):
                    # 尝试拼接路径 1: base_dir/masks/filename
                    temp_path = os.path.join(self.base_image_dir, "masks", mask_path)
                    if os.path.exists(temp_path):
                        mask_path = temp_path
                    else:
                        # 尝试拼接路径 2: base_dir/filename (以防没有 masks 子目录)
                        temp_path = os.path.join(self.base_image_dir, mask_path)
                        if os.path.exists(temp_path):
                            mask_path = temp_path
                # 如果最终路径存在，读取图片
                if os.path.exists(mask_path):
                    m_file = cv2.imread(mask_path, 0) # 读取灰度图
                    if m_file is not None:
                        # 二值化处理：假设 mask 图片是 0 和 255
                        m = m_file // 255 
                        mask_loaded = True
            # -----------------------------------------------------------
            # 2. 如果文件读取失败（或没有path），则回退使用 polygons
            # -----------------------------------------------------------
            if not mask_loaded and "polygons" in m_data:
                # 检查 polygons 是否非空
                polys = m_data["polygons"]
                if len(polys) > 0:
                    for poly in polys:
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(m, [pts], 1)
            # Load GT Mask for metric calculation
            m_data = masks_data[i]
            m = np.zeros((ori_h, ori_w), dtype=np.uint8)
            if "polygons" in m_data:
                for poly in m_data["polygons"]:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(m, [pts], 1)
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())

        inference = True
        # Return format matching collate_fn expectation
        return (
            path_A,
            image_A_tensor,
            image_B_tensor,
            image_clip_A,
            image_clip_B,
            
            conversations,
            masks,
            label,
            resize,
            questions,
            ref_exps,
            inference
        )
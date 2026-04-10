import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, CHANGE_REFER_QUESTIONS


class ChangeReferDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

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
        refer_seg_data="ChangeRef_split|train", # 格式: 数据集名|split
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = CHANGE_REFER_QUESTIONS
        self.answer_list = ANSWER_LIST

        # --- 解析数据路径 ---
        dataset_name, split = refer_seg_data.split("|") # ChangeRef_split train
        self.path2split = os.path.join(base_image_dir, dataset_name, split)
        json_pattern = os.path.join(self.path2split, "referring_expression", "*.json")
        self.json_paths = glob.glob(json_pattern)

        print(f"Dataset {dataset_name} ({split}) has {len(self.json_paths)} samples.")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = idx % len(self.json_paths)
        json_path = self.json_paths[idx]

        with open(json_path, "r") as f:
            data = json.load(f)

        # --- 1. A 和 B ---
        img_name_A = os.path.splitext(data["img_A"])[0] + ".jpg"
        img_name_B = os.path.splitext(data["img_B"])[0] + ".jpg"

        # A/B 文件夹与 referring_expression 同级
        path_A = os.path.join(self.path2split, "A", img_name_A)
        path_B = os.path.join(self.path2split, "B", img_name_B)

        image_A = cv2.imread(path_A)
        image_B = cv2.imread(path_B)
        
        if image_A is None or image_B is None:
            print(f"Error loading images: {path_A} or {path_B}")
            return self.__getitem__(0)
        
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
        
        ori_h, ori_w = image_A.shape[:2]

        # --- 2. CLIP 预处理 (双图) ---
        image_clip_A = self.clip_image_processor.preprocess(image_A, 
                                    return_tensors="pt")["pixel_values"][0]
        image_clip_B = self.clip_image_processor.preprocess(image_B, 
                                    return_tensors="pt")["pixel_values"][0]

        # --- 3. SAM 预处理 (双图) ---
        image_A_sam = self.transform.apply_image(image_A)
        image_B_sam = self.transform.apply_image(image_B)
        resize = image_A_sam.shape[:2]
        image_A_tensor = self.preprocess(torch.from_numpy(image_A_sam).permute(2, 0, 1).contiguous())
        image_B_tensor = self.preprocess(torch.from_numpy(image_B_sam).permute(2, 0, 1).contiguous())
        
        # --- 4. 解析 Referring Expressions 和 Mask ---
        ref_exps = data["referring_expressions"]
        masks_data = data["mask"]
        
        # 从每一个referring_expression.json中随机采样有效个数的json
        valid_indices = list(range(len(ref_exps)))
        if len(valid_indices) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                valid_indices, size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = valid_indices

        sampled_sents = [ref_exps[i] for i in sampled_inds]
        sampled_mask_data = [masks_data[i] for i in sampled_inds]
        
        questions = []
        answers = []
        for text in sampled_sents:
            text = text.strip()
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # --- 6. 生成 Masks ---
        masks = []
        for m_data in sampled_mask_data:
            m = np.zeros((ori_h, ori_w), dtype=np.uint8)
            if "path" in m_data:
                mask_path = m_data["path"]
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(self.path2split, "masks", mask_path)
                mask_loaded = False
                if os.path.exists(mask_path):
                    m_file = cv2.imread(mask_path, 0)
                    if m_file is not None:
                        m = m_file // 255
                        mask_loaded = True 
            if not mask_loaded and "polygons" in m_data and len(m_data["polygons"]) > 0:
                for poly in m_data["polygons"]:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(m, [pts], 1)
            masks.append(m.astype(np.uint8))

        # stack masks
        if len(masks) == 0:
             masks = torch.zeros(0, ori_h, ori_w)
        else:
             masks = np.stack(masks, axis=0)
             masks = torch.from_numpy(masks)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        inference = False
        return (
            path_A,           # 返回路径方便调试
            image_A_tensor,   # SAM Image A
            image_B_tensor,   # SAM Image B
            image_clip_A,     # CLIP Image A
            image_clip_B,     # CLIP Image B
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            inference
        )
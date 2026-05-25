"""
Bi-temporal Change Detection Dataset for SegEarth-R1.

Expected dataset structure (ChangeRef-TV style):
  {base_data_path}/
  ├── train/
  │   ├── A/           # T1 images (before)
  │   │   └── *.jpg / *.png
  │   ├── B/           # T2 images (after)
  │   │   └── *.jpg / *.png
  │   ├── masks/       # ground truth binary masks
  │   │   └── *.png
  │   └── referring_expression/   # JSON with referring expressions
  │       └── *.json
  └── val/  (same structure)

Each JSON in referring_expression/ has the format:
  {
    "img_A": "filename.jpg",
    "img_B": "filename.jpg",
    "referring_expressions": ["expression1", "expression2", ...],
    "mask": [{"path": "relative/mask.png", ...}, ...]
  }
"""

import os
import re
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segearth_r1.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    SEG_TOKEN_INDEX, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX,
)
from segearth_r1 import conversation as conversation_lib


# ─────────────────────── preprocessing helpers ───────────────────────────────

def preprocess_image(image, image_size, pad_value=0):
    """Aspect-ratio-preserving resize + center pad to (image_size, image_size)."""
    h, w = image.shape[:2]
    if w > h:
        new_w = image_size
        new_h = int(h * (image_size / w))
    else:
        new_h = image_size
        new_w = int(w * (image_size / h))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded.transpose(2, 0, 1)   # HWC -> CHW


def preprocess_mask(mask, image_size):
    """Aspect-ratio-preserving resize + center pad for a binary mask."""
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    bs, h, w = mask.shape
    processed = []
    for i in range(bs):
        m = mask[i]
        hh, ww = m.shape
        if ww > hh:
            new_w, new_h = image_size, int(hh * (image_size / ww))
        else:
            new_h, new_w = image_size, int(ww * (image_size / hh))
        m_resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h = image_size - new_h
        pad_w = image_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        m_padded = cv2.copyMakeBorder(m_resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)
        processed.append(m_padded)
    return torch.from_numpy(np.stack(processed, axis=0)).to(torch.uint8)


def tokenizer_special_tokens(prompt, tokenizer,
                              image_token_index=IMAGE_TOKEN_INDEX,
                              seg_token_index=SEG_TOKEN_INDEX,
                              refer_token_index=REFER_TOKEN_INDEX,
                              answer_token_index=ANSWER_TOKEN_INDEX,
                              return_tensors=None):
    special_token_map = {
        '<image>': image_token_index,
        '<seg>': seg_token_index,
        '<refer>': refer_token_index,
        '<answer>': answer_token_index,
    }
    chunks = re.split(r'(<image>|<seg>|<refer>|<answer>)', prompt)
    input_ids = []
    for chunk in chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        else:
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long).squeeze()
    return input_ids


def preprocess_llama2(sources, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack(
        [tokenizer_special_tokens(p, tokenizer, return_tensors='pt')
         for p in conversations],
        dim=0,
    )
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_special_tokens(rou, tokenizer))
            instruction_len = len(tokenizer_special_tokens(parts[0], tokenizer)) - 2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess_referring_instruction(instruction, tokenizer, REFER_token='[SEG]'):
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    tokenized = tokenized + [tokenizer.encode(REFER_token, add_special_tokens=False)[0]]
    return torch.tensor(tokenized)


# ──────────────────────────────────────────────────────────────────────────────


class ChangeDetectionDataset(Dataset):
    """
    Bi-temporal Referring Change Detection Dataset.

    Loads pairs of T1 / T2 images plus a binary change mask and a
    free-text referring expression.  Returns the data in the format
    expected by segearth_r1's training pipeline.

    Args:
        base_data_path: root directory of the change-detection dataset.
        tokenizer: HuggingFace tokenizer.
        split: 'train' or 'val'.
        image_size: spatial size for preprocessing (default 1024).
        random_expression: whether to randomly pick one expression per sample
                           during training (True) or always use the first (False).
    """

    PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD  = torch.Tensor([58.395,  57.12,  57.375]).view(-1, 1, 1)

    def __init__(self, base_data_path, tokenizer, split='train',
                 image_size=1024, random_expression=True, visual_only=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.random_expression = random_expression
        self.visual_only = visual_only

        split_dir = base_data_path # os.path.join(base_data_path, split)
        self.img_A_dir   = os.path.join(split_dir, 'A')
        self.img_B_dir   = os.path.join(split_dir, 'B')
        self.mask_dir    = os.path.join(split_dir, 'masks')
        self.ref_exp_dir = os.path.join(split_dir, 'referring_expression')

        self.samples = self._load_samples()
        print(f'[ChangeDetectionDataset] split={split}, samples={len(self.samples)}')

    # ──────────────────── sample loading ─────────────────────────────────────

    def _load_samples(self):
        samples = []

        if not os.path.isdir(self.ref_exp_dir):
            return self._load_samples_no_json()

        for fname in sorted(os.listdir(self.ref_exp_dir)):
            if not fname.endswith('.json'):
                continue

            json_path = os.path.join(self.ref_exp_dir, fname)
            with open(json_path, 'r') as f:
                ann = json.load(f)

            json_stem = os.path.splitext(fname)[0]
            sample_id = ann.get("sample_id", json_stem)
            img_A_name = ann.get("img_A", sample_id)
            img_B_name = ann.get("img_B", sample_id)

            img_A_path = self._find_image(self.img_A_dir, os.path.splitext(img_A_name)[0])
            img_B_path = self._find_image(self.img_B_dir, os.path.splitext(img_B_name)[0])

            if img_A_path is None or img_B_path is None:
                continue

            expressions = ann.get("referring_expressions", [])
            if not expressions:
                continue

            mask_entries = ann.get("mask", ann.get("masks", []))
            declared_masks = []
            if isinstance(mask_entries, list):
                for entry in mask_entries:
                    mask_rel = entry if isinstance(entry, str) else entry.get("path", "")
                    if not mask_rel:
                        declared_masks.append(None)
                        continue
                    candidate = mask_rel if os.path.isabs(mask_rel) else os.path.join(self.mask_dir, mask_rel)
                    declared_masks.append(candidate if os.path.isfile(candidate) else None)

            folder_masks = self._collect_folder_masks(sample_id, json_stem)
            flat_mask = self._collect_flat_mask(sample_id, json_stem)

            for idx, expression in enumerate(expressions):
                mask_path = None

                if idx < len(declared_masks) and declared_masks[idx] is not None:
                    mask_path = declared_masks[idx]
                elif idx < len(folder_masks):
                    mask_path = folder_masks[idx]
                elif flat_mask is not None:
                    mask_path = flat_mask

                if mask_path is None:
                    continue

                samples.append({
                    'img_name': sample_id,
                    'img_A_path': img_A_path,
                    'img_B_path': img_B_path,
                    'mask_path': mask_path,
                    'expression': expression,
                })

        return samples

    def _load_samples_no_json(self):
        """Pair A and B images by stem name; use the same-name mask."""
        samples = []
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_A_dir)
            if f.lower().endswith(exts)
        ])
        for stem in stems:
            img_A_path = self._find_image(self.img_A_dir, stem)
            img_B_path = self._find_image(self.img_B_dir, stem)
            mask_path  = self._find_image(self.mask_dir,  stem,
                                          exts=('.png', '.tif', '.tiff'))
            if None in (img_A_path, img_B_path, mask_path):
                continue
            samples.append({
                'img_name': stem,
                'img_A_path': img_A_path,
                'img_B_path': img_B_path,
                'mask_path': mask_path,
                'expression': 'the changed area between the two images',
            })
        return samples

    @staticmethod
    def _find_image(directory, stem, exts=('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        for ext in exts:
            p = os.path.join(directory, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def _collect_folder_masks(self, sample_id, json_stem):
        candidates = [
            os.path.join(self.mask_dir, sample_id),
            os.path.join(self.mask_dir, json_stem),
        ]
        mask_files = []
        for folder in candidates:
            if not os.path.isdir(folder):
                continue
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ]
            if not files:
                continue
            mask_files = sorted(files, key=self._mask_sort_key)
            break
        return mask_files

    def _collect_flat_mask(self, sample_id, json_stem):
        for stem in (sample_id, json_stem):
            candidate = self._find_image(
                self.mask_dir,
                stem,
                exts=('.png', '.jpg', '.jpeg', '.tif', '.tiff'),
            )
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _mask_sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (int(stem), '') if stem.isdigit() else (10**9, stem)

    # ──────────────────── dataset protocol ───────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── load & preprocess T1 image ──────────────────────────────────────
        img_A = cv2.imread(sample['img_A_path'])
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        proc_A = preprocess_image(img_A, self.image_size)
        proc_A = (torch.tensor(proc_A, dtype=torch.float32)
                  - self.PIXEL_MEAN) / self.PIXEL_STD   # [3, H, W]

        # ── load & preprocess T2 image ──────────────────────────────────────
        img_B = cv2.imread(sample['img_B_path'])
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        proc_B = preprocess_image(img_B, self.image_size)
        proc_B = (torch.tensor(proc_B, dtype=torch.float32)
                  - self.PIXEL_MEAN) / self.PIXEL_STD   # [3, H, W]

        # ── load & preprocess mask ───────────────────────────────────────────
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)              # binarise
        proc_mask = preprocess_mask(mask, self.image_size) # [1, H, W]

        if self.visual_only:
            return {
                'image':        proc_A,
                'image_A':      proc_A,
                'image_B':      proc_B,
                'mask':         proc_mask,
                'dataset_type': 'change_detection_stage1',
                'image_name':   sample['img_name'],
            }

        # ── build dual-image prompt ──────────────────────────────────────────
        expression = sample['expression']
        prefix_inst = (
            'These are two temporal remote sensing images <image> and <image>. '
            'Please do Referring Change Detection Segmentation according to the '
            'following instruction:'
        )
        sources = [[
            {'from': 'human', 'value': prefix_inst + '\n<refer>'},
            {'from': 'gpt',   'value': '\nSure. It is <seg>. '},
        ]]
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels    = text_dict['labels'][0]

        instruction = ' {}'.format(expression)
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        return {
            'image':                  proc_A,       # kept for DataCollector compat
            'image_A':                proc_A,        # [3, H, W]  T1
            'image_B':                proc_B,        # [3, H, W]  T2
            'mask':                   proc_mask,     # [1, H, W]
            'input_ids':              input_ids,
            'labels':                 labels,
            'dataset_type':           'change_detection',
            'token_refer_id':         token_refer_id,
            'refer_embedding_indices': refer_embedding_indices,
            'image_name':             sample['img_name'],
        }


# ────────────────────────── data collator ────────────────────────────────────

class ChangeDataCollector:
    """Collate bi-temporal change detection samples into a batch."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_dicts):
        if 'input_ids' not in data_dicts[0]:
            batch = {
                'images': torch.stack([d['image_A'] for d in data_dicts]),
                'images_t2': torch.stack([d['image_B'] for d in data_dicts]),
                'dataset_type': [d.get('dataset_type', 'change_detection_stage1') for d in data_dicts],
            }
            if 'mask' in data_dicts[0]:
                batch['masks'] = torch.stack([d['mask'] for d in data_dicts])
            if 'image_name' in data_dicts[0]:
                batch['image_name'] = [d['image_name'] for d in data_dicts]

            for d in data_dicts:
                for key in ['image', 'image_A', 'image_B', 'mask']:
                    d.pop(key, None)

            return batch

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d['input_ids'] for d in data_dicts],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [d['labels'] for d in data_dicts],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels    = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # T1 images → 'images' (used by forward)
        batch['images']    = torch.stack([d['image_A'] for d in data_dicts])
        # T2 images → 'images_t2' (new dual-temporal key)
        batch['images_t2'] = torch.stack([d['image_B'] for d in data_dicts])

        if 'mask' in data_dicts[0]:
            batch['masks'] = torch.stack([d['mask'] for d in data_dicts])

        batch['dataset_type'] = [d.get('dataset_type', 'change_detection') for d in data_dicts]

        if 'token_refer_id' in data_dicts[0]:
            batch['token_refer_id'] = [d['token_refer_id'] for d in data_dicts]

        if 'refer_embedding_indices' in data_dicts[0]:
            ref_indices = torch.nn.utils.rnn.pad_sequence(
                [d['refer_embedding_indices'] for d in data_dicts],
                batch_first=True,
                padding_value=0,
            )
            batch['refer_embedding_indices'] = ref_indices

        if 'image_name' in data_dicts[0]:
            batch['image_name'] = [d['image_name'] for d in data_dicts]

        # Clean up per-sample dicts to avoid memory leaks
        for d in data_dicts:
            for key in ['input_ids', 'labels', 'image', 'image_A', 'image_B', 'mask']:
                d.pop(key, None)

        return batch

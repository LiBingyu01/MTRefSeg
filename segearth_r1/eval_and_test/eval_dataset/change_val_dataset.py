"""
Bi-temporal Change Detection validation dataset for SegEarth-R1.

Directory layout (ChangeRef-TV style):
  {base_data_path}/
  ├── val/           (or any split name)
  │   ├── A/               T1 images
  │   ├── B/               T2 images
  │   ├── masks/           ground truth binary masks
  │   └── referring_expression/   JSON annotations (one per image pair)
  ├── RS/            (optional subset)
  └── ...
"""

import os
import re
import json
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
    h, w = image.shape[:2]
    if w > h:
        new_w, new_h = image_size, int(h * (image_size / w))
    else:
        new_h, new_w = image_size, int(w * (image_size / h))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = image_size - new_h, image_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded.transpose(2, 0, 1)   # CHW


def preprocess_mask(mask, image_size):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 0)
    bs, h, w = mask.shape
    result = []
    for i in range(bs):
        m = mask[i]
        hh, ww = m.shape
        if ww > hh:
            new_w, new_h = image_size, int(hh * (image_size / ww))
        else:
            new_h, new_w = image_size, int(ww * (image_size / hh))
        m_r = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ph, pw = image_size - new_h, image_size - new_w
        t, b = ph // 2, ph - ph // 2
        l, r = pw // 2, pw - pw // 2
        m_p = cv2.copyMakeBorder(m_r, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
        result.append(m_p)
    return torch.from_numpy(np.stack(result, 0)).to(torch.uint8)


def tokenizer_special_tokens(prompt, tokenizer,
                              image_token_index=IMAGE_TOKEN_INDEX,
                              seg_token_index=SEG_TOKEN_INDEX,
                              refer_token_index=REFER_TOKEN_INDEX,
                              answer_token_index=ANSWER_TOKEN_INDEX,
                              return_tensors=None):
    special = {
        '<image>': image_token_index, '<seg>': seg_token_index,
        '<refer>': refer_token_index, '<answer>': answer_token_index,
    }
    chunks = re.split(r'(<image>|<seg>|<refer>|<answer>)', prompt)
    ids = []
    for c in chunks:
        if c in special:
            ids.append(special[c])
        else:
            ids.extend(tokenizer.encode(c, add_special_tokens=False))
    if return_tensors == 'pt':
        return torch.tensor(ids, dtype=torch.long).squeeze()
    return ids


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
         for p in conversations], dim=0)
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


# ─────────────────────────────────────────────────────────────────────────────


class ChangeValDataset(Dataset):
    """
    Validation dataset for bi-temporal referring change detection.

    Args:
        base_data_path: root directory (contains split subdirs like val/, RS/, etc.)
        tokenizer: HF tokenizer.
        split: subdirectory name, e.g. 'val', 'RS', 'NS', 'TT'.
        image_size: preprocessing image size.
    """

    PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD  = torch.Tensor([58.395,  57.12,  57.375]).view(-1, 1, 1)

    def __init__(self, base_data_path, tokenizer, split='val', image_size=1024):
        super().__init__()
        self.tokenizer  = tokenizer
        self.image_size = image_size

        split_dir = os.path.join(base_data_path, split)
        print(f"split_dir_{split_dir}")
        
        # Compatible with two formats:
        # 1) base_data_path = dataset root, split = val
        #    -> base_data_path/val/A
        # 2) base_data_path = full split dir
        #    -> base_data_path/A
        if os.path.isdir(os.path.join(base_data_path, 'A')) and \
        os.path.isdir(os.path.join(base_data_path, 'B')) and \
        os.path.isdir(os.path.join(base_data_path, 'masks')):
            split_dir = base_data_path
        else:
            split_dir = os.path.join(base_data_path, split)

        print(f"split_dir_{split_dir}")

        self.img_A_dir   = os.path.join(split_dir, 'A')
        self.img_B_dir   = os.path.join(split_dir, 'B')
        self.mask_dir    = os.path.join(split_dir, 'masks')
        self.ref_exp_dir = os.path.join(split_dir, 'referring_expression')

        self.samples = self._load_samples()
        print(f'[ChangeValDataset] split={split}, samples={len(self.samples)}')

    # ──────────────────── helpers ─────────────────────────────────────────────

    @staticmethod
    def _find_image(directory, stem, exts=('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        for ext in exts:
            p = os.path.join(directory, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def _load_samples(self):
        samples = []

        if not os.path.isdir(self.ref_exp_dir):
            return self._load_samples_no_json()

        for fname in sorted(os.listdir(self.ref_exp_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(self.ref_exp_dir, fname)) as f:
                ann = json.load(f)
            img_name   = os.path.splitext(fname)[0]
            img_A_name = ann.get('img_A', img_name)
            img_B_name = ann.get('img_B', img_name)
            # Strip any extension from the JSON filename field, then look for
            # the actual image using _find_image (handles .jpg / .png etc.)
            img_A_stem = os.path.splitext(img_A_name)[0]
            img_B_stem = os.path.splitext(img_B_name)[0]
            img_A_path = self._find_image(self.img_A_dir, img_A_stem)
            img_B_path = self._find_image(self.img_B_dir, img_B_stem)
            if img_A_path is None or img_B_path is None:
                continue

            expressions = ann.get('referring_expressions', [])
            mask_entries = ann.get('mask', ann.get('masks', []))
            sample_id = ann.get('sample_id', img_name)

            if not expressions:
                continue

            for expr_idx, expression in enumerate(expressions):
                mask_abs = None

                # 1) 兼容原始格式：JSON 里直接给 mask 路径
                if expr_idx < len(mask_entries):
                    entry = mask_entries[expr_idx]
                    mask_rel = entry if isinstance(entry, str) else entry.get('path', '')
                    candidate = os.path.join(self.mask_dir, mask_rel)
                    if os.path.isfile(candidate):
                        mask_abs = candidate

                # 2) 兼容你的格式：masks/{sample_id}/{expr_idx}.png
                if mask_abs is None:
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        candidate = os.path.join(self.mask_dir, sample_id, f'{expr_idx}{ext}')
                        if os.path.isfile(candidate):
                            mask_abs = candidate
                            break

                # 3) 兼容另一种常见格式：masks/{img_name}/{expr_idx}.png
                if mask_abs is None:
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        candidate = os.path.join(self.mask_dir, img_name, f'{expr_idx}{ext}')
                        if os.path.isfile(candidate):
                            mask_abs = candidate
                            break

                # 4) 兜底：如果只有一个 mask，尝试 masks/{sample_id}.png
                if mask_abs is None:
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        candidate = os.path.join(self.mask_dir, sample_id + ext)
                        if os.path.isfile(candidate):
                            mask_abs = candidate
                            break

                if mask_abs is None:
                    continue

                samples.append({
                    'img_name':   f'{sample_id}_{expr_idx}',
                    'img_A_path': img_A_path,
                    'img_B_path': img_B_path,
                    'mask_path':  mask_abs,
                    'expression': expression,
                })
        return samples

    def _load_samples_no_json(self):
        samples = []
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_A_dir)
            if f.lower().endswith(exts)
        ])
        for stem in stems:
            img_A = self._find_image(self.img_A_dir, stem)
            img_B = self._find_image(self.img_B_dir, stem)
            mask  = self._find_image(self.mask_dir,  stem,
                                     exts=('.png', '.tif', '.tiff'))
            if None in (img_A, img_B, mask):
                continue
            samples.append({
                'img_name':   stem,
                'img_A_path': img_A,
                'img_B_path': img_B,
                'mask_path':  mask,
                'expression': 'the changed area between the two images',
            })
        return samples

    # ──────────────────── dataset ─────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_A = cv2.cvtColor(cv2.imread(s['img_A_path']), cv2.COLOR_BGR2RGB)
        proc_A = preprocess_image(img_A, self.image_size)
        proc_A = (torch.tensor(proc_A, dtype=torch.float32)
                  - self.PIXEL_MEAN) / self.PIXEL_STD

        img_B = cv2.cvtColor(cv2.imread(s['img_B_path']), cv2.COLOR_BGR2RGB)
        proc_B = preprocess_image(img_B, self.image_size)
        proc_B = (torch.tensor(proc_B, dtype=torch.float32)
                  - self.PIXEL_MEAN) / self.PIXEL_STD

        mask = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        proc_mask = preprocess_mask(mask, self.image_size)

        expression = s['expression']
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
            'image_A':                 proc_A,
            'image_B':                 proc_B,
            'mask':                    proc_mask,
            'input_ids':               input_ids,
            'labels':                  labels,
            'token_refer_id':          token_refer_id,
            'refer_embedding_indices': refer_embedding_indices,
            'image_name':              s['img_name'],
        }


# ────────────────────────── collator ─────────────────────────────────────────

class ChangeValDataCollector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_dicts):
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
            images=torch.stack([d['image_A'] for d in data_dicts]),
            images_t2=torch.stack([d['image_B'] for d in data_dicts]),
            masks=torch.stack([d['mask'] for d in data_dicts]),
        )

        if 'token_refer_id' in data_dicts[0]:
            batch['token_refer_id'] = [d['token_refer_id'] for d in data_dicts]

        if 'refer_embedding_indices' in data_dicts[0]:
            ref_idx = torch.nn.utils.rnn.pad_sequence(
                [d['refer_embedding_indices'] for d in data_dicts],
                batch_first=True, padding_value=0)
            batch['refer_embedding_indices'] = ref_idx

        if 'image_name' in data_dicts[0]:
            batch['image_name'] = [d['image_name'] for d in data_dicts]

        return batch

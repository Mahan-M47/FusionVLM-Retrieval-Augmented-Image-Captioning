
import os
import json
import random
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path


# ===============================================================
# Flickr30K Dataset for CLIP
# ===============================================================

class Flickr30KImageDataset(Dataset):
    """
    Dataset notes:
    - Images are named as integers with .jpg extension (e.g. 14121.jpg)
    - captions CSV has columns: image_name, caption_idx, caption
    - Multiple captions per image
    """

    def __init__(self, image_dir: str, captions_dir:str):
        self.image_dir = image_dir
        df = pd.read_csv(captions_dir)
        self.captions_per_image = df.groupby("image_name")["caption"].apply(list).to_dict()
        self.image_names = df['image_name'].unique()

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        captions = self.captions_per_image[image_name]

        return {
            "image": image, 
            "image_name": image_name,
            "captions": captions
        }
        
    def __len__(self):
        return len(self.image_names)
    
    
class CLIPDataCollator:
    def __init__(self, processor, device="cuda"):
        self.processor = processor
        self.device = device

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        image_names = [b["image_name"] for b in batch]
        captions = [b["captions"] for b in batch]
        
        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values

        return {
            "pixel_values": pixel_values.to(self.device),
            "image_names": image_names,
            "captions": captions,
        }

# ===============================================================
# Flickr30K Dataset (Image only)
# ===============================================================
class Flickr30KRawImageDataset(Dataset):
    def __init__(self, image_dir, captions_dir):
        self.image_dir = image_dir
        
        df = pd.read_csv(captions_dir)
        self.image_names = df['image_name'].unique()

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
        
    def __len__(self):
        return len(self.image_names)
    

# ===============================================================
# Flickr30K Dataset for BLIP (Fine-tune with RAG)
# ===============================================================
class BlipRAGDataset(Dataset):
    def __init__(
        self,
        image_dir, 
        metadata_path,
        retriever,
        caption_prompt="Describe the image.",
        num_similar_captions=3,
        transform=None,
    ):
        """
        metadata_path: path to JSON metadata file
        similar_indices: dict[int, list[int]] from FAISS
        image_base_path: base directory for images
        caption_prompt: base captioning instruction
        num_similar_captions: how many retrieved captions to inject
        transform: optional image transform
        """
        self.retriever = retriever
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.image_base_path = Path(image_dir)
        self.caption_prompt = caption_prompt
        self.num_similar_captions = num_similar_captions
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, image_name):
        path = self.image_base_path / image_name
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def _sample_similar_captions(self, neighbors):
        if len(neighbors) == 0:
            return []

        k = min(self.num_similar_captions, len(neighbors))
        sampled = neighbors[:k]
        
        captions = []
        for idx in sampled:
            caps = self.retriever.retrieve_captions(idx)
            captions.append(random.choice(caps))

        return captions

    def _build_prompt(self, similar_captions):
        if len(similar_captions) == 0:
            return self.caption_prompt

        context = "\n".join(f"- {c}" for c in similar_captions)

        return (
            f"Similar images are described as:\n"
            f"{context} \n\n "
            f"{self.caption_prompt}"
        )

    def __getitem__(self, idx):
        img_data = self.metadata[str(idx)]

        image = self._load_image(img_data["image_name"])
        target_caption = max(img_data["captions"], key=len)
        similar_captions = self._sample_similar_captions(img_data['similar_images'])
        prompt = self._build_prompt(similar_captions)

        return {
            "image": image,
            "prompt": prompt,
            "target_caption": target_caption,
        }


class BlipDataCollator:
    def __init__(self, processor, max_seq_len=128, device="cuda"):
        self.processor = processor
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        targets = [b["target_caption"] for b in batch]

        inputs = self.processor(
            images=images,
            text=prompts,
            padding="max_length",
            padding_side='left',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        labels = self.processor.tokenizer(
            targets,
            padding="max_length",
            padding_side='left',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        ).input_ids

        # Important: ignore padding tokens in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        # return {k: v.to(self.device) for k, v in inputs.items()}
        return {k: v for k, v in inputs.items()}


# ===============================================================
# Flickr30K Dataset for FusionVLM
# ===============================================================
class VLMDataset(Dataset):
    def __init__(self, image_dir, ref_image_dir, metadata_path, retriever,
                 num_similar_captions=3, label_prompt="A picture of "):

        self.image_base_dir = Path(image_dir)
        self.ref_image_dir = Path(ref_image_dir)
        self.retriever = retriever
        self.label_prompt = label_prompt
        self.num_similar_captions = num_similar_captions

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)


    def _load_image(self, image_name, base_path):
        path = base_path / image_name
        image = Image.open(path).convert("RGB")
        return image

    def _sample_similar_captions(self, neighbors):
        if len(neighbors) == 0:
            return []

        k = min(self.num_similar_captions, len(neighbors))
        sampled = random.sample(neighbors, k)
        captions = []
        for idx in sampled:
            caps = self.retriever.retrieve_captions(idx)
            captions.append(random.choice(caps))
        return captions

    def _build_prompt(self, similar_captions):
        if len(similar_captions) == 0:
            return ""
        context = " ".join(f"{c}" for c in similar_captions)
        return f"Similar images are described as:\n{context}"


    def __getitem__(self, idx):
        img_data = self.metadata[str(idx)]
        query_image = self._load_image(img_data["image_name"], self.image_base_dir)
        
        retrieved_idx = img_data["similar_images"][0]
        retrieved_image_name = self.retriever.retrieve_image_name(retrieved_idx)
        retrieved_image = self._load_image(retrieved_image_name, self.ref_image_dir)
        
        retrieved_captions = self._sample_similar_captions(img_data["similar_images"])
        prompt = self._build_prompt(retrieved_captions)
        target_caption = max(img_data["captions"], key=len) # Target caption (longest one)
        
        return {
            "query_image": query_image,
            "retrieved_image": retrieved_image,
            "prompt": self.label_prompt + prompt,
            "target_caption": target_caption,
            "all_captions": img_data["captions"]
            # "all_captions": [self.label_prompt + caption for caption in img_data["captions"]]
        }


class VLMDataCollator:
    def __init__(self, processor, tokenizer, label_tokenizer=None, max_seq_len=128, device="cuda"):
        self.processor = processor
        self.tokenizer = tokenizer
        
        self.label_tokenizer = label_tokenizer
        if not self.label_tokenizer:
            self.label_tokenizer = tokenizer
            
        self.max_seq_len = max_seq_len
        self.device = device

    def __call__(self, batch):
        query_images = [b.get("query_image") for b in batch]
        retrieved_images = [b.get("retrieved_image") for b in batch]
        prompts = [b.get("prompt") for b in batch]
        targets = [b.get("target_caption") for b in batch]
        all_captions = [b.get("all_captions") for b in batch]

        # Process Images
        pixel_values = self.processor(
            images=query_images,
            return_tensors="pt"
        ).pixel_values

        retrieved_pixel_values = self.processor(
            images=retrieved_images,
            return_tensors="pt"
        ).pixel_values

        # Process text prompts
        inputs = self.tokenizer(
            prompts,
            padding="longest",
            padding_side='right',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        labels = self.label_tokenizer(
            targets,
            padding="longest",
            padding_side='right',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        ).input_ids
        
        # ignore padding tokens in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "query_pixel_values": pixel_values.to(self.device),
            "retrieved_pixel_values": retrieved_pixel_values.to(self.device),
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device),
            "labels": labels.to(self.device),
            "all_captions": all_captions
        }


from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, BlipProcessor


class Flickr30KImageDataset(Dataset):
    """
    Dataset notes:
    - Images are named as integers with .jpg extension (e.g. 14121.jpg)
    - captions CSV has columns: image_name, caption_idx, caption
    - Multiple captions per image
    """

    def __init__(self, image_dir, captions_dir, processor_name=None, model_name=None, device='cpu'): # Processor: CLIPProcessor, BLIPProcessor
        
        self.image_dir = image_dir
        self.device = device
        
        if processor_name == 'CLIP':
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
        elif processor_name == 'BLIP':
            self.processor = BLIPProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
        else:
            self.processor = None

        # Load captions CSV and group captions by image
        df = pd.read_csv(captions_dir)
        self.captions_per_image = df.groupby("image_name")["caption"].apply(list).to_dict()

        # Load images based on the names in the captions dataset
        self.image_names = [
            img_name for img_name in self.captions_per_image.keys()
            if os.path.exists(os.path.join(self.image_dir, img_name))
        ]
        
        self.to_tensor = transforms.ToTensor() 
        

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # if reprocessing to tensor using CLIPProcessor, returns shape [1, 3, 224, 224], so squeeze to [3, 224, 224]
        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            
            if isinstance(self.processor, CLIPProcessor):
                image_tensor = processed['pixel_values'].squeeze(0)
            else:
                image_tensor = processed['pixel_values']
        else:
            image_tensor = self.to_tensor(image)

        captions = self.captions_per_image[image_name]

        return {
            "image_tensor": image_tensor.to(self.device), 
            "image_name": image_name,
            "captions": captions
        }
        
    def __len__(self):
        return len(self.image_names)


def CLIP_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]

        # If tensor, stack along new batch dimension
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            # leave lists, strings, or other objects as-is
            collated[key] = values
    return collated

    

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
    
    


import json
import random
from pathlib import Path
from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset
# from Retrieval_module import Retriever

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

        # self.metadata_df = pd.DataFrame(metadata)
        
        self.image_base_path = Path(image_dir)
        self.caption_prompt = caption_prompt
        self.num_similar_captions = num_similar_captions
        self.transform = transform

        # assert "image_name" in self.metadata_df.columns
        # assert "captions" in self.metadata_df.columns
        # assert "similiar_images" in self.metadata_df.columns
                    

    def __len__(self):
        # return len(self.metadata_df)
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

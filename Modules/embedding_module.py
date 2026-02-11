
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessorFast, CLIPTokenizer
import torch
import numpy as np

from Modules.config import CLIP_MODEL_NAME

class Embedder:
    def __init__(self, model_name=CLIP_MODEL_NAME, use_local_files=False, device='cuda'):
        self.model_name = model_name
        self.device = device

        self.CLIP_model = CLIPModel.from_pretrained(self.model_name, local_files_only=use_local_files).to(self.device)

        # Freeze CLIP parameters
        self.CLIP_model.eval()
        for param in self.CLIP_model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.CLIP_model.visual_projection.out_features
        
        
    def get_image_embedding(self, pixel_values, return_tensor=False):
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            embeddings = self.CLIP_model.get_image_features(pixel_values)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        if return_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy().astype(np.float32)
        
        
    def get_text_embedding(self, input_ids, attention_mask, return_tensor=False):
        if len(input_ids.shape) == 3:
            input_ids = input_ids.unsqueeze(0)
            
        if len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(0)
        
        input_ids.to(self.device)
        attention_mask.to(self.device)
        with torch.no_grad():
            embeddings = self.CLIP_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        if return_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy().astype(np.float32)
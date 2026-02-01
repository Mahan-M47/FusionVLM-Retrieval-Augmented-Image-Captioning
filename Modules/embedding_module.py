
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessorFast, CLIPTokenizer
import torch
import numpy as np


class Embedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", use_local_files=False, device='cuda'):
        self.model_name = model_name
        self.device = device

        self.CLIP_processor = CLIPImageProcessorFast.from_pretrained(self.model_name, local_files_only=use_local_files)
        self.CLIP_tokenizer = CLIPTokenizer.from_pretrained(self.model_name, local_files_only=use_local_files)
        self.CLIP_model = CLIPModel.from_pretrained(self.model_name, local_files_only=use_local_files).to(self.device)

        # Freeze CLIP parameters
        self.CLIP_model.eval()
        for param in self.CLIP_model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.CLIP_model.visual_projection.out_features
        
        
    def get_image_embedding(self, query, return_tensor=False):
        if len(query.shape) == 3:
            query = query.unsqueeze(0)
        
        query.to(self.device)
        with torch.no_grad():
            embeddings = self.CLIP_model.get_image_features(query)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        if return_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy().astype(np.float32)
        
        
    def get_text_embedding(self, query, return_tensor=False):
        
        query = self.CLIP_tokenizer(query,
                                    padding="longest",
                                    padding_side='right',
                                    truncation=True,
                                    max_length=77,
                                    return_tensors="pt")
        
        query.to(self.device)
        with torch.no_grad():
            embeddings = self.CLIP_model.get_text_features(**query)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        if return_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy().astype(np.float32)
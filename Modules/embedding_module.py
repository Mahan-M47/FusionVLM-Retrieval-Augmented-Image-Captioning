
from transformers import CLIPProcessor, CLIPModel
import torch


class Embedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cpu'):
        self.model_name = model_name
        self.device = device

        self.CLIP_processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True, use_fast=True)
        self.CLIP_model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)

        # Freeze CLIP parameters
        self.CLIP_model.eval()
        for param in self.CLIP_model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.CLIP_model.visual_projection.out_features
        
        
    def get_image_embedding(self, query):
        if len(query.shape) == 3:
            query = query.unsqueeze(0)
        
        query.to(self.device)
        with torch.no_grad():
            embedding = self.CLIP_model.get_image_features(query)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        return embedding
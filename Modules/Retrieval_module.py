
import os
import json
import numpy as np
import faiss
from PIL import Image


class Retriever():
    def __init__(self, metadata_path=None, faiss_path=None, base_image_path=None,
                 faiss_index=None, metadata=None):
        
        self.metadata = None
        self.faiss_index = None
        self.base_image_path = base_image_path
        
        if metadata:
            self.metadata = metadata
        elif metadata_path:
            self.set_metadata(metadata_path)
        else:
            print('Retriever created without metadata file.')
        
        if faiss_index:
            self.faiss_index = faiss_index
        elif faiss_path:
            self.set_faiss_index(faiss_path)
        else:
            print('Retriever created without FAISS index.')
        
        
    def set_metadata(self, metadata_path):
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    def set_faiss_index(self, faiss_path):
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        else:
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")
        
        
    def retrieve_similar_from_metadata(self, query_embeddings, k=1):
        if self.metadata and self.faiss_index:
            distance, idx = self.faiss_index.search(query_embeddings, k=k)
            
            retrieval_results = []
            for i in idx[0]:
                retrieval_results.append(self.metadata[str(i)])
                
            return retrieval_results
        else:
            raise Exception("FAISS index or metadata not initialized. Use set_metadata() and set_faiss_index().")
    
    def retrieve_similar_indices(self, query_embeddings, k=1):
        if self.faiss_index:
            distance, idx = self.faiss_index.search(query_embeddings, k=k)
            return idx.tolist()
        else:
            raise Exception("FAISS index not initialized. Use set_faiss_index().")

    
    def retrieve_image_name(self, idx):
        if self.metadata:
            return self.metadata[str(idx)]['image_name']
        else:
            raise Exception("Metadata index not initialized. Use set_metadata().")
    
    def retrieve_captions(self, idx):
        if self.metadata:
            return self.metadata[str(idx)]['captions']
        else:
            raise Exception("Metadata index not initialized. Use set_metadata().")
            
    def retrieve_image(self, idx):
        if self.base_image_path:
            path = self.base_image_path / self.retrieve_image_name(idx)
            image = Image.open(path).convert("RGB")
            return image
        else:
            raise Exception("Base image path not initialized. Set value for base_image_path.")
    
    
# correct metadata sample
#     "17415": {
#     "image_name": "4003129619.jpg",
#     "captions": [
#       "Two older people in marching uniforms both playing saxophones .",
#       "A man and woman in costume are playing saxophones outside .",
#       "A man and a woman are playing saxophone in the band .",
#       "Three men are playing saxophone in uniforms .",
#       "The musicians are playing for the crowd ."
#     ]
#     }
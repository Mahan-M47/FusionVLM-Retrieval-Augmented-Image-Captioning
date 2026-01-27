
import os
import json
import numpy as np
import faiss


class Retriever():
    def __init__(self, metadata_path, faiss_path, base_image_path=None):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")
        
        # if base_image_path is not None and not os.path.exists(base_image_path):
        #     raise FileNotFoundError(f"Image base path not found: {base_image_path}")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_path)

        # Save image base path
        self.base_image_path = base_image_path
        
        
    def retrieve_similar(self, query_embedding, k=1):
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        distance, idx = self.faiss_index.search(query_embedding, k=k)
        
        retrieval_results = []
        for i in idx[0]:
            retrieval_results.append(self.metadata[str(i)])
            
        return retrieval_results
    
    def retrieve_similar_indices(self, query_embedding, k=1):
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        distance, idx = self.faiss_index.search(query_embedding, k=k)
        
        # return [str(i) for i in idx[0]]
        return idx[0]
    
    def retrieve_image_name(self, idx):
        return self.metadata[str(idx)]['image_name']
    
    def retrieve_captions(self, idx):
        return self.metadata[str(idx)]['captions']
    
    
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

import os
import json
import numpy as np
import faiss
from PIL import Image
import torch


class Retriever():
    def __init__(self, metadata_path=None, faiss_path=None, faiss_index=None, metadata=None):
        
        self.metadata = None
        self.faiss_index = None
        
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
            
            
    def retrieve_from_metadata(self, idx, key=None):
        if self.metadata:
            def _get_item(i):
                item = self.metadata[str(i)]
                return item.get(key) if key else item   
            
            if torch.is_tensor(idx):
                idx = idx.cpu().tolist()

            if isinstance(idx, (list, tuple, np.ndarray)):
                return [_get_item(i) for i in idx]
            else:
                return _get_item(idx)
        else:
            raise Exception("Metadata index not initialized. Use set_metadata().")
        
        
    def retrieve_similar_from_metadata(self, query_embeddings, key=None, k=1): # query embeddings shape: [B, hidden_dim]
        if self.metadata and self.faiss_index:
            
            if torch.is_tensor(query_embeddings):
               query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)
            
            distance, indices = self.faiss_index.search(query_embeddings, k=k)
            retrieval_results = []
            
            for similar_index_list in indices:  # indices shape: [B, k]
                retrieval_results.append(self.retrieve_from_metadata(similar_index_list, key))
                
            return retrieval_results
        else:
            raise Exception("FAISS index or metadata not initialized. Use set_metadata() and set_faiss_index().")
    
    
    def retrieve_similar_indices(self, query_embeddings, k=1):
        if self.faiss_index:
            
            if torch.is_tensor(query_embeddings):
               query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)
               
            distance, indices = self.faiss_index.search(query_embeddings, k=k)
            return indices.tolist() # shape [B, k]
        else:
            raise Exception("FAISS index not initialized. Use set_faiss_index().")
        
    def metadata_keys(self):
        if self.metadata:
            return self.metadata.keys()
        else:
            raise Exception("Metadata index not initialized. Use set_metadata().")
    

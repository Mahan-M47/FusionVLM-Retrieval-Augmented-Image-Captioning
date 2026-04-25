# use streamlit to create browser app for captioning. create noRAG and RAG captions, display ground truth captions, compare metrics for both.

import torch
import matplotlib.pyplot as plt
from transformers import T5TokenizerFast, CLIPImageProcessorFast
# from transformers import BlipProcessor, BlipForConditionalGeneration

from Modules.inference import caption_image_FusionVLM, caption_image_BLIP, load_image
from Modules.FusionVLM import load_default_FusionVLM
from Modules.retrieval_module import Retriever
from Modules.embedding_module import Embedder
from Modules.config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# place the model weights (fusion.pt + lora_text_dcoder folder) in FusionVLM/MODEL_NAME
MODEL_NAME = 'epoch20'

def main():
    # Initialize FusionVLM model and components
    fusionVLM_model = load_default_FusionVLM(MODEL_NAME, use_local_files=True).to(DEVICE)
    embedder = Embedder(CLIP_MODEL_NAME, use_local_files=True, device=DEVICE)

    CLIP_processor = CLIPImageProcessorFast.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    T5_tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_NAME, local_files_only=True)

    image_retriever = Retriever(faiss_path=FAISS_IMAGE_PATH, metadata_path=TRAIN_METADATA_PATH)
    caption_retriever = Retriever(faiss_path=FAISS_CAPTION_PATH, metadata_path=CAPTION_METADATA_PATH)

    print("FusionVLM Loaded")
    img_path = input("Enter image path: ")
    query = load_image(img_path, '')
    
    captions = caption_image_FusionVLM(query=[query], model=fusionVLM_model, embedder=embedder, processor=CLIP_processor, tokenizer=T5_tokenizer, 
                        image_retriever=image_retriever, caption_retriever=caption_retriever, ref_image_dir=TRAIN_IMAGE_DIR, device=DEVICE)
    
    print("Caption: " + captions)
    

if __name__ == "__main__":
    main()

import torch
from Modules.embedding_module import Embedder
from Modules.retrieval_module import Retriever
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5TokenizerFast, CLIPImageProcessorFast


def load_image(image_name, base_path):
    path = Path(base_path) / image_name
    image = Image.open(path).convert("RGB")
    return image


def build_context(similar_captions, num_similar_captions=3):
    k = min(len(similar_captions), num_similar_captions)
    context = ' '.join(similar_captions[:k])
    return f"Similar images are described as: \n {context}"


def caption_image_FusionVLM(query, model, embedder, processor, tokenizer, image_retriever: Retriever, caption_retriever: Retriever, ref_image_dir, device='cuda'):
    model.eval()
    query_pixel_values = processor(
            images=query,
            return_tensors="pt"
        ).pixel_values
    
    embedding = embedder.get_image_embedding(query_pixel_values)
    
    retrieved_image_names = image_retriever.retrieve_similar_from_metadata(embedding, key='image_name', k=1)
    
    retrieved_images = []
    for retrieved_image_name in retrieved_image_names:
        retrieved_images.append(load_image(retrieved_image_name[0], ref_image_dir))
        
    retrieved_pixel_values = processor(
            images=[retrieved_images],
            return_tensors="pt"
    ).pixel_values

    
    similar_captions = caption_retriever.retrieve_similar_from_metadata(embedding, key='caption', k=3)
    context = []
    
    for captions in similar_captions:
        context.append(build_context(captions))
        
    inputs = tokenizer(
            context,
            padding="longest",
            padding_side='right',
            truncation=True,
            max_length=128,
            return_tensors="pt"
    )
    
    query_pixel_values = query_pixel_values.to(device)
    retrieved_pixel_values = retrieved_pixel_values.to(device)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
            generated_ids = model.generate(
                query_pixel_values=query_pixel_values,
                retrieved_pixel_values=retrieved_pixel_values,
                # retrieved_pixel_values=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.1, # longer outputs 
                repetition_penalty=1.2 # penalize repeats
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded          
            


def caption_image_BLIP(query, model, processor, device):
    model.eval()

    pixel_values = processor(
                images=query,
                return_tensors="pt"
            ).pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=64,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.1, # longer outputs 
            repetition_penalty=1.2 # penalize repeats
        )

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded      
    
    


# if __name__ == '__main__':
#     from Modules.config import *
#     from Modules.utils import display_image
#     from Modules.FusionVLM import load_default_FusionVLM
#     from transformers import T5TokenizerFast, CLIPImageProcessorFast
#     from Modules.retrieval_module import Retriever
    
#     query1 = load_image('211295363.jpg', TEST_IMAGE_DIR)
#     query2 = load_image('311406998.jpg', TEST_IMAGE_DIR)
#     display_image(query1)
#     display_image(query2)
    
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

#     model = load_default_FusionVLM('epoch10', use_local_files=True).to(DEVICE)
#     embedder = Embedder(CLIP_MODEL_NAME, use_local_files=True, device=DEVICE)
#     CLIP_processor = CLIPImageProcessorFast.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
#     T5_tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_NAME, local_files_only=True)

#     image_retriever = Retriever(faiss_path=FAISS_IMAGE_PATH, metadata_path=TRAIN_METADATA_PATH)
#     caption_retriever = Retriever(faiss_path=FAISS_CAPTION_PATH, metadata_path=CAPTION_METADATA_PATH)

#     captions = caption_image_FusionVLM(query=[query1, query2], model=model, embedder=embedder, processor=CLIP_processor, tokenizer=T5_tokenizer, 
#                             image_retriever=image_retriever, caption_retriever=caption_retriever, ref_image_dir=TRAIN_IMAGE_DIR, device=DEVICE)

#     print(captions)
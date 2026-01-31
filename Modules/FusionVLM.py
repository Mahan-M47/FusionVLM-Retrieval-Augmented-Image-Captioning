
import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    ViTModel,
    T5EncoderModel,
    T5ForConditionalGeneration,
    CLIPProcessor,
    T5TokenizerFast
)

from transformers.modeling_outputs import BaseModelOutput

from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from pathlib import Path
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os 

from Modules.config import (CLIP_MODEL_NAME, T5_MODEL_NAME, FUSION_BLOCKS,
                            T5_DECODER_LORA_CONFIG, T5_ENCODER_LORA_CONFIG,
                            CLIP_LORA_CONFIG, VLM_CHECKPOINT_DIR)
# from torch.utils.data import DataLoader
# from Modules.retrieval_module import Retriever
# from Modules.FusionVLM import FusionVLM
# from Modules.datasets import VLMDataset, VLMDataCollator


class FusionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(self, text_hidden, vision_hidden, attention_mask):
        key_padding_mask = ~attention_mask.bool()

        # cross-attention
        x, _ = self.cross_attn(
            query=text_hidden, key=vision_hidden, value=vision_hidden
        )
        x = self.ln1(text_hidden + x)

        # self-attention
        sa, _ = self.self_attn(query=x, key=x, value=x, 
                               key_padding_mask=key_padding_mask)
        x = self.ln2(x + sa)

        # feed-forward
        x = self.ln3(x + self.ff(x))
        return x

        
class EnhancedFusionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        # Cross-modal attention: image attending to text and vice versa
        self.image_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.text_to_image = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch, 2, hidden_dim]
        # Assume first token is image, second is text
        image_features = x[:, 0:1, :]  # [B, 1, D]
        text_features = x[:, 1:2, :]   # [B, 1, D]
        
        # Cross-attention
        image_attended, _ = self.image_to_text(
            image_features, text_features, text_features
        )
        text_attended, _ = self.text_to_image(
            text_features, image_features, image_features
        )
        
        # Residual connections
        image_features = self.ln1(image_features + self.dropout(image_attended))
        text_features = self.ln2(text_features + self.dropout(text_attended))
        
        # Combine back
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Self attention
        attended, _ = self.self_attn(combined, combined, combined)
        combined = self.ln3(combined + self.dropout(attended))
        
        # Feed forward
        ff_out = self.ff(combined)
        combined = combined + self.dropout(ff_out)
        
        return combined

    


class FusionVLM(nn.Module):
    def __init__(self, vision_encoder_name: str, text_encoder_name: str, text_decoder_name: str,
                 num_fusion_blocks: int, num_heads: int = 8, prompt='A picture of ', use_local_files=True):
        super().__init__()

        # Vision encoder
        vision_config = CLIPVisionConfig()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name, config=vision_config, local_files_only=use_local_files)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, local_files_only=use_local_files)
        self.text_dim = self.text_encoder.config.d_model

        # Project vision → text space
        self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)

        # Fusion blocks
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(self.text_dim, num_heads) for _ in range(num_fusion_blocks)]
        )
        self.post_fusion_ln = nn.LayerNorm(self.text_dim)
        
        # Text decoder (LM head)
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(text_decoder_name, local_files_only=use_local_files)
        self.tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_NAME)
        self.text_decoder.config.decoder_start_token_id = self.tokenizer.pad_token_id
        # self.decoder_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids


    def forward(self, query_pixel_values, retrieved_pixel_values, input_ids, attention_mask, labels=None):   
             
        q_vis = self.vision_encoder(query_pixel_values).last_hidden_state

        if retrieved_pixel_values is not None:
            r_vis = self.vision_encoder(retrieved_pixel_values).last_hidden_state
            vision_hidden = torch.cat([q_vis, r_vis], dim=1)
        else:
            vision_hidden = q_vis

        vision_hidden = self.vision_proj(vision_hidden)

        # Encode text
        text_hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Fusion
        for block in self.fusion_blocks:
            text_hidden = block(text_hidden, vision_hidden, attention_mask)
            
        text_hidden = self.post_fusion_ln(text_hidden)

        # Decode
        # batched_decoder_prompt = self.decoder_prompt.repeat(len(input_ids), 1).to(input_ids.device)

        outputs = self.text_decoder(
            inputs_embeds=text_hidden,
            attention_mask=attention_mask,
            # decoder_input_ids=batched_decoder_prompt,
            labels=labels,
            return_dict=True,
        )

        return outputs
    
    
    @torch.no_grad()
    def generate(self, query_pixel_values, retrieved_pixel_values, input_ids, attention_mask, 
        max_length=128, num_beams=3, do_sample=False, temperature=1.0, top_p=1.0, **generate_kwargs):

        # ---- Vision ----
        q_vis = self.vision_encoder(query_pixel_values).last_hidden_state

        if retrieved_pixel_values is not None:
            r_vis = self.vision_encoder(retrieved_pixel_values).last_hidden_state
            vision_hidden = torch.cat([q_vis, r_vis], dim=1)
        else:
            vision_hidden = q_vis

        vision_hidden = self.vision_proj(vision_hidden)

        # ---- Text encoder ----
        text_hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # ---- Fusion ----
        for block in self.fusion_blocks:
            text_hidden = block(text_hidden, vision_hidden, attention_mask)
            
        text_hidden = self.post_fusion_ln(text_hidden)

        # ---- Pretend fused embeddings are encoder outputs ----
        encoder_outputs = BaseModelOutput(
            last_hidden_state=text_hidden
        )
        
        # ---- Generate ----
        # batched_decoder_prompt = self.decoder_prompt.repeat(len(input_ids), 1).to(input_ids.device)
        
        generated_ids = self.text_decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            # decoder_input_ids=batched_decoder_prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs,
        )

        return generated_ids
    
    
    @torch.no_grad()
    def T5_generate(self, input_ids, attention_mask, 
        max_length=128, num_beams=3, do_sample=False, temperature=1.0, top_p=1.0, **generate_kwargs):

        # ---- Text encoder ----
        text_hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # ---- Pretend fused embeddings are encoder outputs ----
        encoder_outputs = BaseModelOutput(
            last_hidden_state=text_hidden
        )
        
        # batched_decoder_prompt = self.decoder_prompt.repeat(len(input_ids), 1).to(input_ids.device)
        

        # ---- Generate ----
        generated_ids = self.text_decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            # decoder_input_ids=batched_decoder_prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs,
        )

        return generated_ids
    

def apply_lora_config(fusionVLM: FusionVLM, T5_decoder_config=T5_DECODER_LORA_CONFIG,
                      T5_encoder_config=T5_ENCODER_LORA_CONFIG, vision_encoder_config=CLIP_LORA_CONFIG):
    # Freeze encoders
    # def freeze_module(module: torch.nn.Module):
    #     for p in module.parameters():
    #         p.requires_grad = False
            
    # freeze_module(fusionVLM.vision_encoder)
    # freeze_module(fusionVLM.text_encoder)

    # Apply LoRA 
    fusionVLM.vision_encoder = get_peft_model(fusionVLM.vision_encoder, vision_encoder_config)
    fusionVLM.text_encoder = get_peft_model(fusionVLM.text_encoder, T5_encoder_config)
    fusionVLM.text_decoder = get_peft_model(fusionVLM.text_decoder, T5_decoder_config)

    # Unfreeze lm_head layer
    # for param in fusionVLM.text_decoder.lm_head.parameters():
    #     param.requires_grad = True
    
    return fusionVLM


def create_default_FusionVLM():
    model = FusionVLM(vision_encoder_name=CLIP_MODEL_NAME,
                    text_encoder_name=T5_MODEL_NAME,
                    text_decoder_name=T5_MODEL_NAME,
                    num_fusion_blocks=FUSION_BLOCKS,
                    use_local_files=True
                    )
    
    return apply_lora_config(model)
 

def save_FusionVLM(model: FusionVLM, checkpoint_name: str, save_dir=VLM_CHECKPOINT_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save LoRA adapters ONLY (decoder)
    lora_state = get_peft_model_state_dict(model.text_decoder)
    torch.save(lora_state, save_dir / f"DecoderLoRA_{checkpoint_name}.pt")

    # 2) Save all other trainable (non-frozen, non-LoRA) weights
    fusion_state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_decoder"):
            continue  # LoRA already handled
        fusion_state[name] = param.detach().cpu()

    torch.save(fusion_state, save_dir / f"FusionVLM_{checkpoint_name}.pt")



def load_FusionVLM(checkpoint_name: str, save_dir: str, vision_encoder_name: str,
                   text_encoder_name: str, text_decoder_name: str, lora_config: LoraConfig,
                   num_fusion_blocks=FUSION_BLOCKS, use_local_files=True):
    
    save_dir = Path(save_dir)

    # Recreate base model
    model = FusionVLM(
    vision_encoder_name=vision_encoder_name,
    text_encoder_name=text_encoder_name,
    text_decoder_name=text_decoder_name,
    num_fusion_blocks=num_fusion_blocks,
    use_local_files=use_local_files
    )
    
    # apply LoRA
    model = apply_lora_config(model, lora_config)
    
    # Load LoRA adapter weights
    lora_state = torch.load(save_dir / f"DecoderLoRA_{checkpoint_name}.pt", map_location="cpu")
    model.text_decoder.load_state_dict(lora_state, strict=False)

    # Load fusion block weights
    fusion_state = torch.load(save_dir / f"FusionVLM_{checkpoint_name}.pt", map_location="cpu")
    model.load_state_dict(fusion_state, strict=False)

    return model



def load_default_FusionVLM(checkpoint_name: str):
    return load_FusionVLM(checkpoint_name=checkpoint_name,
                          save_dir=VLM_CHECKPOINT_DIR,
                          vision_encoder_name=CLIP_MODEL_NAME,
                          text_encoder_name=T5_MODEL_NAME,
                          text_decoder_name=T5_MODEL_NAME,
                          lora_config=T5_DECODER_LORA_CONFIG
                          )


import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    CLIPTextModel,
    CLIPTextConfig,
    CLIPProcessor,
    CLIPTokenizerFast,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    T5Config
)

from transformers.modeling_outputs import BaseModelOutput

from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict 
from pathlib import Path
import warnings
import os 

from Modules.config import (CLIP_MODEL_NAME, T5_MODEL_NAME, FUSION_BLOCKS, FUSION_DIM,
                            FUSION_HEADS, T5_DECODER_LORA_CONFIG, T5_ENCODER_LORA_CONFIG,
                            CLIP_LORA_CONFIG, VLM_CHECKPOINT_DIR)


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
        
        return x, vision_hidden

        
class BidirectionalFusionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        # Cross-modal attention: image attending to text and vice versa
        self.text_to_vision_CA = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_to_text_CA = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Self attention
        self.self_attn_text = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_attn_vision = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Layer norms
        self.ln1_text = nn.LayerNorm(hidden_dim)
        self.ln2_text = nn.LayerNorm(hidden_dim)
        self.ln3_text = nn.LayerNorm(hidden_dim)
        self.ln1_vision = nn.LayerNorm(hidden_dim)
        self.ln2_vision = nn.LayerNorm(hidden_dim)
        self.ln3_vision = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_hidden, vision_hidden, attention_mask):
        key_padding_mask = ~attention_mask.bool()  # in pytorch, 1 is pad
             
        # Cross-attention
        text_q_vision_kv, _ = self.text_to_vision_CA(
            query=text_hidden, key=vision_hidden, value=vision_hidden
        )
        vision_q_text_kv, _ = self.vision_to_text_CA(
            query=vision_hidden, key=text_hidden, value=text_hidden, key_padding_mask= key_padding_mask
        )
        
        # Residual + Layer Norm 1
        text_hidden = self.ln1_text(text_hidden + self.dropout(text_q_vision_kv))
        vision_hidden = self.ln1_vision(vision_hidden + self.dropout(vision_q_text_kv))
        
        # Self attention
        text_attended, _ = self.self_attn_text(
            text_hidden, text_hidden, text_hidden, key_padding_mask=key_padding_mask
        )
        
        vision_attended, _ = self.self_attn_vision(
            vision_hidden, vision_hidden, vision_hidden
        )
        
        # Residual + Layer Norm 2
        text_hidden = self.ln2_text(text_hidden + self.dropout(text_attended))
        vision_hidden = self.ln2_vision(vision_hidden + self.dropout(vision_attended))
        
        # Feed forward
        ff_out_text = self.ff(text_hidden)
        ff_out_vision = self.ff(vision_hidden)
        
        # Residual + Layer Norm 3
        text_hidden = self.ln3_text(text_hidden + self.dropout(ff_out_text))
        vision_hidden = self.ln3_vision(vision_hidden + self.dropout(ff_out_vision))
        
        return text_hidden, vision_hidden


class FusionVLM(nn.Module):
    def __init__(self, vision_encoder_name: str, text_encoder_name: str, T5_text_decoder_name: str,
                 bidirectional_fusion=True, fusion_dim=FUSION_DIM, num_fusion_blocks=FUSION_BLOCKS,
                 num_fusion_heads=FUSION_HEADS, prompt='A picture of ', use_local_files=False):
        
        super().__init__()
        
        config = T5Config.from_pretrained(T5_text_decoder_name, local_files_only=use_local_files)
        self.text_decoder_dim = config.d_model

        self.fusion_dim = fusion_dim
        self.num_fusion_blocks = num_fusion_blocks
        self.num_fusion_heads = num_fusion_heads
        
        # Vision encoder
        vision_config = CLIPVisionConfig()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name, config=vision_config, local_files_only=use_local_files)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Text encoder
        if text_encoder_name[:2] == 't5':
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, local_files_only=use_local_files)
            self.text_dim = self.text_encoder.config.d_model
        else:
            text_config = CLIPTextConfig()
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name, config=text_config, local_files_only=use_local_files)
            self.text_dim = self.text_encoder.config.hidden_size
            
        
        # Project vision,text → Fusion space
        self.vision_proj = nn.Linear(self.vision_dim, self.fusion_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fusion_dim)

        # Fusion blocks
        if bidirectional_fusion:
            self.fusion_blocks = nn.ModuleList(
                [BidirectionalFusionBlock(self.fusion_dim, self.num_fusion_heads) for _ in range(self.num_fusion_blocks)]
            )
        else:
            self.fusion_blocks = nn.ModuleList(
                [FusionBlock(self.fusion_dim, self.num_fusion_heads) for _ in range(self.num_fusion_blocks)]
            )
        
        # Post fusion projection
        self.post_fusion_ln = nn.LayerNorm(self.fusion_dim)
        self.fusion_proj = nn.Linear(self.fusion_dim, self.text_decoder_dim)

        # Text Decoder
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(T5_text_decoder_name, local_files_only=use_local_files)
        self.tokenizer = T5TokenizerFast.from_pretrained(T5_text_decoder_name, local_files_only=use_local_files)
        self.text_decoder.config.decoder_start_token_id = self.tokenizer.pad_token_id

        # Tokenize generate prompt
        self.generate_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids
        

    def _pipeline_forward(self, query_pixel_values, retrieved_pixel_values, input_ids, attention_mask):
        # Encode image
        q_vis = self.vision_encoder(query_pixel_values).last_hidden_state

        if retrieved_pixel_values is not None:
            r_vis = self.vision_encoder(retrieved_pixel_values).last_hidden_state
            vision_hidden = torch.cat([q_vis, r_vis], dim=1)
        else:
            vision_hidden = q_vis

        # Encode text
        text_hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        
        # Projection
        vision_hidden = self.vision_proj(vision_hidden)
        text_hidden = self.text_proj(text_hidden)

        # Fusion
        for block in self.fusion_blocks:
            text_hidden, vision_hidden = block(text_hidden, vision_hidden, attention_mask)
            
        text_hidden = self.post_fusion_ln(text_hidden)
        text_hidden = self.fusion_proj(text_hidden)

        # ---- Pretend fused embeddings are encoder outputs ----
        encoder_outputs = BaseModelOutput(
            last_hidden_state=text_hidden
        )
        
        return encoder_outputs
        
        
    def forward(self, query_pixel_values, retrieved_pixel_values, input_ids, attention_mask, labels):   
        
        encoder_outputs = self._pipeline_forward(query_pixel_values, retrieved_pixel_values, input_ids, attention_mask,)
        
        outputs = self.text_decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs
    
    
    @torch.no_grad()
    def generate(self, query_pixel_values, retrieved_pixel_values, input_ids, attention_mask, 
        max_length=128, num_beams=3, do_sample=False, temperature=1.0, top_p=1.0, **generate_kwargs):

        encoder_outputs = self._pipeline_forward(query_pixel_values, retrieved_pixel_values, input_ids, attention_mask,)
        batched_decoder_prompt = self.generate_prompt.repeat(len(input_ids), 1).to(input_ids.device)
        
        # Generate
        generated_ids = self.text_decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=batched_decoder_prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs,
        )

        return generated_ids
    

def freeze_encoders(model: FusionVLM):
    # Freeze encoders
    def freeze_module(module: torch.nn.Module):
        for p in module.parameters():
            p.requires_grad = False
            
    freeze_module(model.vision_encoder)
    freeze_module(model.text_encoder)
    return model


def apply_lora_config(fusionVLM: FusionVLM, T5_decoder_config=T5_DECODER_LORA_CONFIG,
                      T5_encoder_config=T5_ENCODER_LORA_CONFIG, vision_encoder_config=CLIP_LORA_CONFIG):
    
    # fusionVLM.vision_encoder = get_peft_model(fusionVLM.vision_encoder, vision_encoder_config)
    # fusionVLM.text_encoder = get_peft_model(fusionVLM.text_encoder, T5_encoder_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fusionVLM.text_decoder = get_peft_model(fusionVLM.text_decoder, T5_decoder_config)
    
    return fusionVLM


def create_default_FusionVLM(use_local_files=False):
    model = FusionVLM(vision_encoder_name=CLIP_MODEL_NAME,
                    text_encoder_name=T5_MODEL_NAME,
                    T5_text_decoder_name=T5_MODEL_NAME,
                    num_fusion_blocks=FUSION_BLOCKS,
                    use_local_files=use_local_files
                    )
    
    model = freeze_encoders(model)
    return apply_lora_config(model)
 

def save_FusionVLM(model: FusionVLM, checkpoint_name: str, save_dir=VLM_CHECKPOINT_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) save LoRA adapters
    # model.vision_encoder.save_pretrained(save_dir / checkpoint_name / "lora_vision_encoder")
    # model.text_encoder.save_pretrained(save_dir / checkpoint_name / "lora_text_encoder")
    model.text_decoder.save_pretrained(save_dir / checkpoint_name / "lora_text_decoder")
    
    # 2) save all other trainable (non-LoRA) params
    fusion_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith(("text_decoder", "text_encoder", "vision_encoder"))
    }
    torch.save(fusion_state, save_dir / checkpoint_name / "fusion.pt")
    

def load_FusionVLM(checkpoint_name: str, save_dir: str, vision_encoder_name: str,
                   text_encoder_name: str, text_decoder_name: str,
                   num_fusion_blocks=FUSION_BLOCKS, use_local_files=False):
    
    save_dir = Path(save_dir)

    # Recreate base model
    model = FusionVLM(
    vision_encoder_name=vision_encoder_name,
    text_encoder_name=text_encoder_name,
    T5_text_decoder_name=text_decoder_name,
    num_fusion_blocks=num_fusion_blocks,
    use_local_files=use_local_files
    )
    
    # Load LoRA weights
    # model.vision_encoder = PeftModel.from_pretrained(
    #     model.vision_encoder, 
    #     save_dir / checkpoint_name / "lora_vision_encoder",
    #     is_trainable=True
    # )
    
    # model.text_encoder = PeftModel.from_pretrained(
    #     model.text_encoder, 
    #     save_dir / checkpoint_name / "lora_text_encoder",
    #     is_trainable=True
    # )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.text_decoder = PeftModel.from_pretrained(
            model.text_decoder, 
            save_dir / checkpoint_name / "lora_text_decoder",
            is_trainable=True
        )

    # Load fusion block weights
    fusion_state = torch.load(save_dir / checkpoint_name / 'fusion.pt', map_location="cpu")
    model.load_state_dict(fusion_state, strict=False)
    
    model = freeze_encoders(model)
    return model



def load_default_FusionVLM(checkpoint_name: str, use_local_files=False):
    model = load_FusionVLM(checkpoint_name=checkpoint_name,
                          save_dir=VLM_CHECKPOINT_DIR,
                          vision_encoder_name=CLIP_MODEL_NAME,
                          text_encoder_name=T5_MODEL_NAME,
                          text_decoder_name=T5_MODEL_NAME,
                          use_local_files=use_local_files
                          )
    return model

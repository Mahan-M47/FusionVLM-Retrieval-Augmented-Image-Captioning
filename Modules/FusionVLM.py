
import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    ViTModel,
    T5EncoderModel,
    T5ForConditionalGeneration,
    CLIPProcessor
)

from transformers.modeling_outputs import BaseModelOutput

from peft import LoraConfig, get_peft_model, TaskType
import os 
from torch.amp import autocast, GradScaler
from tqdm import tqdm

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

    def forward(self, text_hidden, vision_hidden):
        # cross-attention
        x, _ = self.cross_attn(
            query=text_hidden, key=vision_hidden, value=vision_hidden
        )
        x = self.ln1(text_hidden + x)

        # self-attention
        sa, _ = self.self_attn(x, x, x)
        x = self.ln2(x + sa)

        # feed-forward
        x = self.ln3(x + self.ff(x))
        return x



class FusionVLM(nn.Module):
    def __init__(self, vision_encoder_name: str, text_encoder_name: str, text_decoder_name: str,
                 num_fusion_blocks: int, num_heads: int = 8, use_local_files=True):
        super().__init__()

        # Vision encoder
        vision_config = CLIPVisionConfig()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name, config=vision_config, local_files_only=use_local_files)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, local_files_only=use_local_files)
        self.text_dim = self.text_encoder.config.d_model

        # Text decoder (LM head)
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(text_decoder_name, local_files_only=use_local_files)

        # Project vision → text space
        self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)

        # Fusion blocks
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(self.text_dim, num_heads) for _ in range(num_fusion_blocks)]
        )

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
            text_hidden = block(text_hidden, vision_hidden)

        # Decode
        outputs = self.text_decoder(
            inputs_embeds=text_hidden,
            attention_mask=attention_mask,
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
            text_hidden = block(text_hidden, vision_hidden)

        # ---- Pretend fused embeddings are encoder outputs ----
        encoder_outputs = BaseModelOutput(
            last_hidden_state=text_hidden
        )

        # ---- Generate ----
        generated_ids = self.text_decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs,
        )

        return generated_ids
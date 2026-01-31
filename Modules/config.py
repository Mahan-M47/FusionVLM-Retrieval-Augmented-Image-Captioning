
import torch
from peft import LoraConfig, get_peft_model, TaskType

CAPTIONS_PATH = "dataset/captions.csv"
IMAGE_ROOT = "dataset/flickr30k_images"

TRAIN_IMAGE_DIR = "dataset/flickr30k_images/train"
TRAIN_CAPTIONS_PATH = "dataset/captions-train.csv"

TEST_IMAGE_DIR = "dataset/flickr30k_images/test"
TEST_CAPTIONS_PATH = "dataset/captions-test.csv"

FAISS_PATH = "flickr30k_clip_images.faiss"
TRAIN_METADATA_PATH = "train_metadata.json"
TEST_METADATA_PATH = "test_metadata.json"


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
T5_MODEL_NAME = "t5-base"
ViT_MODEL_NAME = "google/vit-base-patch16-224"

VLM_CHECKPOINT_DIR = "FusionVLM"
FUSION_BLOCKS = 2

T5_DECODER_LORA_CONFIG = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules = [
        # decoder self-attention
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",

        # decoder cross-attention
        "EncDecAttention.q",
        "EncDecAttention.k",
        "EncDecAttention.v",
        "EncDecAttention.o",
    ],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["lm_head"],
    # task_type=TaskType.SEQ_2_SEQ_LM,
    task_type=TaskType.CAUSAL_LM
)

T5_ENCODER_LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

# CLIP_LORA_CONFIG = LoraConfig(
#     r=16,                      
#     lora_alpha=32,
#     target_modules=[
#         "self_attn.q_proj",
#         "self_attn.v_proj",
#         "self_attn.v_proj",
#         "self_attn.out_proj"
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.FEATURE_EXTRACTION
# )
CLIP_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["visual_projection"],
    # task_type=TaskType.FEATURE_EXTRACTION
)
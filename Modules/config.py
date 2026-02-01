
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

SEED = 13
N_TEST_IMAGES = 1000

CAPTIONS_PATH = "dataset/captions.csv"
IMAGE_ROOT = "dataset/flickr30k_images"

TRAIN_IMAGE_DIR = "dataset/flickr30k_images/train"
TRAIN_CAPTIONS_PATH = "dataset/captions-train.csv"

TEST_IMAGE_DIR = "dataset/flickr30k_images/test"
TEST_CAPTIONS_PATH = "dataset/captions-test.csv"

FAISS_IMAGE_PATH = "flickr30k_clip_images.faiss"
FAISS_CAPTION_PATH = "flickr30k_clip_captions.faiss"

CAPTION_METADATA_PATH = "caption_metadata.json"
TRAIN_METADATA_PATH = "train_metadata.json"
TEST_METADATA_PATH = "test_metadata.json"

VLM_CHECKPOINT_DIR = "FusionVLM"

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"  # or "Salesforce/blip-image-captioning-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # or "openai/clip-vit-base-patch16"
T5_MODEL_NAME = "t5-base"  # or "t5-large"

FUSION_BLOCKS = 4
FUSION_HEADS = 8
FUSION_DIM = 768

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
    # ensure_weight_tying=True,
    task_type=TaskType.SEQ_2_SEQ_LM,
    # task_type=TaskType.CAUSAL_LM  # this one worked 
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

CLIP_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["visual_projection"],
    # task_type=TaskType.FEATURE_EXTRACTION
)
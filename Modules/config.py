
import torch

CAPTIONS_PATH = "dataset/captions.csv"
IMAGE_ROOT = "dataset/flickr30k_images"

TRAIN_IMAGE_DIR = "dataset/flickr30k_images/train"
TRAIN_CAPTIONS_PATH = "dataset/captions-train.csv"

TEST_IMAGE_DIR = "dataset/flickr30k_images/test"
TEST_CAPTIONS_PATH = "dataset/captions-test.csv"

FAISS_PATH = "flickr30k_clip_images.faiss"
TRAIN_METADATA_PATH = "train_metadata.json"
TEST_METADATA_PATH = "test_metadata.json"

CHECKPOINT_DIR = "FusionVLM"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
T5_MODEL_NAME = "t5-base"
ViT_MODEL_NAME = "google/vit-base-patch16-224"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FusionVLM: Retrieval-Augmented Image Captioning

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)


**FusionVLM** is a vision-language model for image captioning that integrates **Multimodal Retrieval** with a custom **Bidirectional Fusion Block** architecture. Instead of relying solely on learned parameters, it retrieves visually or semantically similar images and captions from a dataset to improve caption quality, reduce hallucinations, and enhance generalization.


## Table of Contents

- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Results & Samples](#️-results--samples)
- [References](#-references)
- [License](#-license)



## 📌 Key Features

- **Retrieval-Augmented Generation**: Retrieves similar images and captions from a FAISS vector database to ground the captioning process.
- **Bidirectional Cross-Attention**: Custom Fusion Blocks enable symmetric information flow between visual and textual modalities.
- **Parameter-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) on the T5 decoder while keeping encoders frozen.
- **Multi-Modal Retrieval**: Leverages CLIP embeddings to build indexes for both images and captions. **FAISS** is used to create separate vector databases for the image and caption embedding.
- **Comprehensive Evaluation**: Supports BLEU-1/2/3/4, METEOR, ROUGE-L, and CIDEr metrics.


## 🏗️ Project Structure

```
├── Modules/
│   ├── config.py                      # Paths, constants, and configurations
│   ├── datasets.py                    # Dataset classes for embeddings & training
│   ├── embedding_module.py            # CLIP embedding wrapper
│   ├── FusionVLM.py                   # Custom vision-language model implementation
│   ├── retrieval_module.py            # FAISS-based multi-modal retriever
│   ├── train_VLM.py                   # Training and evaluation functions
│   ├── metrics.py                     # Evaluation metrics (BLEU, METEOR, CIDEr, etc.)
│   ├── inference.py                   # Caption generation utilities
│   └── utils.py                       # Shared helper functions
├── dataset/     
│   ├── flickr30kImages                # Manually download and place images
│   └── captions.csv                   # Manually download and place captions
├── FusionVLM/     
│   └── MODEL_NAME/                    # Model weights (optional download)
├── 01-data_split.ipynb                # Split dataset with fixed seed
├── 02-FAISS_image_index.ipynb         # Build FAISS index for images
├── 03-FAISS_caption_index.ipynb       # Build FAISS index for captions
├── 04-FusionVLM.ipynb                 # Train FusionVLM
├── 05-Inference_and_evaluation.ipynb  # Inference & comparison with BLIP
├── app.py                             # Script to run inference via CLI
├── faiss_indexes.faiss                # Stored FAISS vector databases (created after running the notebooks)
├── metadata.json                      # Stored JSON metadata files (created after running the notebooks)
└── requirements.txt                   # A list of required libraries for running the project
```

## 🧠 Model Architecture


The Architecture of the Custom Vision-Language Model:
(Diagram created using the free online tool [app.diagrams.net](https://app.diagrams.net))

<p align="center">
  <img height=250 src="media/FusionVLM.jpg" >
</p>

FusionVLM consists of seven main components:

| Component | Description |
|-----------|-------------|
| **Retriever** | Retrieves the top similar images and captions using cosine similarity on CLIP embeddings |
| **CLIP Image Encoder** | `openai/clip-vit-base-patch32` – Encodes query image + n retrieved images into visual embeddings |
| **T5 Text Encoder** | `t5-base` – Encodes top-k retrieved captions as a context string |
| **Input Projection Layers** | Projects vision/text embeddings to fusion block dimension (768) |
| **Fusion Blocks** | 4 bidirectional cross-attention blocks (8 heads each) that fuse modalities iteratively |
| **Text Encoder Projection** | Projects fused text embeddings to decoder input space |
| **T5 Text Decoder** | `t5-base` – Autoregressively generates the final caption (fine-tined with LoRA) |


### Fusion Block Details

Fustion Blocks are custom transformer-based multimodal modules designed to integrate (fuse) textual and visual representations before decoding.

<p align="center">
  <img height=120 src="media/Fusion_Block.png" >
</p>

Each fusion block performs:
1. **Cross-attention**: Text → Vision and Vision → Text (bidirectional)
2. **Self-attention**: Modality-specific refinement after cross-modal interaction
3. **Feed-forward network**: Position-wise transformation with residual connections

This design allows each modality to continuously adapt to the other across multiple layers, producing vision-aware textual embeddings for the decoder. After being processed by the fusion blocks, the vision hidden representation is **discarded** and the text hidden representation is **projected** to the same space as the text decoder to prepare for caption generation



## 📊 Dataset

The dataset used for this project is [**Flickr30k**](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), a widely used benchmark in vision-and-language research that contains 31,783 images, each paired with 5 human-written
captions. The dataset is split such that 1,000 images are reserved for validation and testing, while the remaining images (~31,000) are used for training the
model

⚠️ Note that the dataset must be **downloaded manually** and placed in the path specified in `config.py`.

### 🔍 Multimodal Retrieval

The retrieval module encodes all images and captions into a shared embedding space using **CLIP**, allowing **direct comparison between visual and textual representations**. Given an input image embedding, it computes similarity scores
(using cosine similarity) against the precomputed embeddings and returns the most similar images and captions.

Image and caption embeddings are stored in **FAISS vector databases**. Each vector database is accompanied by a metadata JSON file, which includes details such as the name of the image and its corresponding captions for the image embedding database.
- **Image embedding DB**: stores CLIP embeddings of all images in the training dataset
- **Caption embedding DB**: stores CLIP embeddings of all captions in the training dataset


During **Training**, the retrieved results are cached for efficiency, while during **Inference**, they are retrieved dynamically. For each sample, the input given to the model is:

- Query image  
- 1 retrieved image (most similar from the Image FAISS index)
- Top 3 retrieved captions (most similar from the Caption FAISS index)  

### 🔍 Similar ImageRetrieval Examples


<p align="center">
  <img height=220 src="media/faiss_retrieved_images1.jpg" >
</p>

<p align="center">
  <img height=225 src="media/faiss_retrieved_images2.jpg" >
</p>



## 📦 Requirements

- Python 3.14 (recommended)  
- CUDA-capable GPU (≥ 8GB VRAM recommended)  

Install dependencies:

```bash
pip install -r requirements.txt
```

### Pretrained Models

The following models are required:
- `openai/clip-vit-base-patch32  `
- `t5-base ` 
- `Salesforce/blip-image-captioning-base` (optional, for comparison)

These models will be automatically downloaded when running the code.


## 🚀 Usage

### Step 1: Prepare Dataset
Download [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and place the images and captions in the `dataset/` directory. You can update the dataset path in `config.py`.

### Step 2: Run Notebooks to create FAISS indexes

```bash
01-data_split.ipynb  
02-FAISS_image_index.ipynb  
03-FAISS_caption_index.ipynb  
```

⚠️ Order matters — each step depends on previous outputs. After running the first 3 notebooks, you should see the following files in the project's root directory:

```bash
caption_metadata.json
flickr30k_clip_captions.faiss
flickr30k_clip_images.faiss
test_metadata.json
train_metadata.json
```

### Step 3: Train model or use Pretrained weights

you can either run the training notebook:
```bash
04-FusionVLM.ipynb  
```
with the default training configuration (Can be changed in `config.py` or `FusionVLM.py`):
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-2)
- **Epochs**: 20
- **Fusion Blocks**: 4 layers, 8 attention heads, hidden size 768
- **LoRA**: Applied to T5 decoder's self-attention and cross-attention layers
- **Frozen components**: CLIP image encoder, T5 text encoder
- **Loss**: Cross-entropy (next-token prediction) with the Target caption (randomly sampled) 

Or you can skip training by downloading the FusionVLM model weights after 20 epochs of training. The checkpoint includes the trained fusion blocks, LoRA-adapted T5 decoder layers, and all projection layers. The CLIP vision encoder and T5 text encoder weights are frozen and not included in the checkpoint (they are loaded from their respective pretrained models)

Download the model checkpoint from Google Drive:
**[FusionVLM - Trained Weights (20 epochs)](https://drive.google.com/your-link-here)**

Place the model weights in the `FusionVLM/` directory before running inference.


### Step 4: Inference and Evaluation

Run the inference notebook (requires BLIP download for comparison):
```bash
05-Inference_and_evaluation.ipynb
```

Or run `app.py` to manually test the model on any image.



## 🖼️ Results & Samples


In order to evaluate our model more accurately, we compare the captioning results of our model
with those of BLIP (`Salesforce/blip-image-captioning-base`):

### Sample Captions (Train Dataset)

<p align="center">
  <img height=220 src="media/inference_train_images.jpg" >
</p>


| Image | FusionVLM | BLIP |
|------|----------|------|
| Image 1 | A picture of a little boy on a sunny day | a small child standing in a field of grass |
| Image 2 | A picture of a man in a white shirt is walking in front of a building. | a crowd of people walking on a busy street at night |
| Image 3 | A picture of hikers in the mountains | a man with a backpack |

---

### Sample Captions (Test Dataset)


<p align="center">
  <img height=220 src="media/inference_test_images.jpg" >
</p>

| Image | FusionVLM | BLIP |
|------|----------|------|
| Image 1 | A picture of on a sunny day with a boat in the background | a large body of water |
| Image 2 | A picture of a man and a woman are walking down the street | a group of people sitting at a table eating |
| Image 3 | A picture of soccer players in a stadium | a group of people on a soccer field |



### 📊 Quantitative Results


Comparison between FusionVLM and BLIP on the Flickr30k validation set. Higher values indicate better caption quality:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|-------|--------|--------|--------|--------|--------|---------|-------|
| **FusionVLM** | **0.6614** | **0.4960** | **0.3823** | **0.2899** | **0.4842** | **0.5138** | **0.3072** |
| BLIP | 0.3253 | 0.2462 | 0.1759 | 0.1183 | 0.2672 | 0.4204 | 0.2539 |


### 📉 Training Curves

<p align="center">
  <img height=300 src="media/training_history_loss.png" >
</p>

---

## 📚 References

- [Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Captioning](https://aclanthology.org/2023.findings-emnlp.791/) (Yang et al., EMNLP 2023)
- [Flickr30k Dataset](https://aclanthology.org/Q14-1006/) (Young et al., TACL 2014)
- [BLIP: Bootstrapping Language-Image Pre-training](https://proceedings.mlr.press/v162/li22n.html) (Li et al., ICML 2022)
- [FAISS: Billion-scale similarity search](https://arxiv.org/abs/2001.08910) (Johnson et al.)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) (Radford et al., ICML 2021)
- [T5: Exploring the Limits of Transfer Learning](https://jmlr.org/papers/v21/20-074.html) (Raffel et al., JMLR 2020)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al.)


## 📄 License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
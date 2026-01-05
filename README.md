# Vision Transformer for Image Captioning
A custom implementation of Vision Transformer (ViT) for image captioning, built from scratch using PyTorch. This project implements the encoder-decoder Transformer architecture with a Vision Transformer encoder and a text decoder for generating natural language descriptions of images.

## 🎯 Overview

This project implements an image captioning model that:
- Uses a **Vision Transformer (ViT)** encoder to process images as sequences of patches
- Employs a **Transformer decoder** with causal masking for autoregressive caption generation
- Trains on the **Flickr30k** dataset
- Supports greedy decoding for inference

## 🏗️ Architecture

### Model Components

| Component | Description |
|-----------|-------------|
| **Image Embedding** | Splits images into 16×16 patches and projects them to d_model dimensions using Conv2D |
| **Positional Encoding** | Sinusoidal positional encodings for both image patches and text tokens |
| **Encoder** | Stack of N encoder layers with multi-head self-attention and feed-forward networks |
| **Decoder** | Stack of N decoder layers with masked self-attention, cross-attention, and FFN |
| **Linear Projection** | Projects decoder output to vocabulary size for token prediction |

### Default Hyperparameters

```python
IMG_SIZE = (224, 224)      # Input image size
PATCH_SIZE = (16, 16)      # Patch size (results in 196 patches)
D_MODEL = 768              # Model dimension
NUM_HEADS = 12             # Number of attention heads
D_FF = 1024                # Feed-forward dimension
NUM_ENC = 4                # Number of encoder layers
NUM_DEC = 4                # Number of decoder layers
MAX_SEQ_LEN = 100          # Maximum caption length
```

## 📁 Project Structure

```
my_llm/
├── train.py                 # Training script with validation
├── test.py                  # Testing and evaluation (BLEU score)
├── dataset.py               # Dataset and vocabulary classes
├── count_param.py           # Parameter counting utility
├── my_transformer/
│   ├── __init__.py
│   ├── layers.py            # Core Transformer layers (attention, FFN, etc.)
│   └── models.py            # BaseTransformer and VisionTransformer models
├── dataset/
│   ├── flickr_annotations_30k.csv
│   ├── flickr30k.py
│   └── flickr30k-images/    # Image directory
└── checkpoints/             # Saved model checkpoints
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision pandas pillow tqdm nltk matplotlib
```

### Dataset Setup

1. Download the Flickr30k dataset images and place them in `dataset/flickr30k-images/`
2. Ensure `dataset/flickr_annotations_30k.csv` contains the annotations with columns:
   - `filename`: Image filename
   - `raw`: List of 5 captions per image
   - `split`: 'train', 'val', or 'test'

### Training

```bash
python train.py
```

**Training Features:**
- Automatic vocabulary building from training data
- Teacher forcing during training
- Gradient clipping (max_norm=1.0)
- Validation at configurable intervals
- Early stopping with patience
- Best model checkpoint saving
- AdamW optimizer with configurable learning rate

### Testing

```bash
python test.py
```

**Evaluation Features:**
- BLEU-4 score calculation on test set
- Visual captioning samples with matplotlib
- Support for loading best or specific epoch checkpoints

## 📊 Model Statistics

To count model parameters:

```bash
python count_param.py
```

## ⚙️ Configuration

Both `train.py` and `test.py` use a `Config` class for hyperparameters:

```python
class Config:
    # Data paths
    CSV_FILE = 'dataset/flickr_annotations_30k.csv'
    IMG_ROOT = 'dataset/flickr30k-images/'
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 60
    VAL_FREQUENCY = 1
    EARLY_STOP_PATIENCE = 10
```

## 🔧 Custom Transformer Implementation

The `my_transformer/` module contains a complete Transformer implementation:

### Key Classes in `layers.py`
- `Embedding` - Token embedding with scaling
- `ImageEmbedding` - Patch-based image embedding using Conv2D
- `PositionalEncoding` - Sinusoidal positional encodings
- `MultiHeadAttention` - Scaled dot-product multi-head attention
- `FeedForwardNetwork` - Position-wise FFN
- `LayerNormalization` - Layer normalization
- `Encoder` / `Decoder` - Full encoder/decoder stacks

### Key Classes in `models.py`
- `BaseTransformer` - Generic encoder-decoder Transformer
- `VisionTransformer` - ViT variant with image embedding encoder
- `InitializeTransformer` - Xavier uniform initialization

## 📈 Training Progress

Checkpoints are saved to `./checkpoints/`:
- `vit_caption_epoch_N.pth` - Checkpoint at epoch N
- `vit_caption_best.pth` - Best model based on validation loss

Each checkpoint contains:
- `model_state_dict`
- `optimizer_state_dict`
- `vocab` (vocabulary object)
- `epoch`
- `train_loss`
- `val_loss`

## 📝 Usage Example

```python
from my_transformer.models import VisionTransformer, InitializeTransformer
from dataset import Vocabulary

# Initialize model
model = VisionTransformer(
    num_enc=4,
    num_dec=4,
    d_model=768,
    num_heads=12,
    d_ff=1024,
    img_size=(224, 224),
    patch_size=(16, 16),
    target_vocab_size=vocab_size,
    max_seq_len=100
)
model = InitializeTransformer(model)

# Load checkpoint
checkpoint = torch.load('checkpoints/vit_caption_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
vocab = checkpoint['vocab']

# Generate caption (see test.py for full implementation)
caption = generate_caption(model, image, vocab, max_len=20)
```

## ⚠️ Limitations & Why This Is a Learning Project

**This model is designed for educational purposes and will exhibit poor performance on real image captioning tasks.** Here's why:

### 1. No Pre-trained Vision Encoder
- **The Problem:** The ViT encoder is trained **from scratch** on only ~30k images
- **Real-world models:** Use encoders pre-trained on ImageNet (1.2M images) or larger datasets (e.g., CLIP trained on 400M image-text pairs)
- **Impact:** The model cannot learn robust visual features from such limited data

### 2. Small Dataset
- **Flickr30k** contains only ~31,000 images with 5 captions each
- **State-of-the-art models** train on:
  - COCO Captions: 330k images
  - Conceptual Captions: 3.3M images
  - LAION-5B: 5 billion image-text pairs
- **Impact:** Severe overfitting and poor generalization to unseen images

### 3. Model Scale
| Aspect | This Model | Production Models |
|--------|------------|-------------------|
| Encoder Layers | 4 | 12-24+ |
| Decoder Layers | 4 | 12-24+ |
| d_model | 768 | 768-1024+ |
| d_ff | 1024 | 3072-4096 |
| Total Parameters | ~50M | 100M-13B+ |

### 4. Training Limitations
- **No learning rate scheduling** (warmup, cosine decay)
- **No label smoothing** for better generalization
- **Simple cross-entropy loss** instead of CIDEr optimization or SCST (Self-Critical Sequence Training)
- **No image augmentation** during training

### 5. Inference Limitations
- **Greedy decoding only** - picks the most probable token at each step
- **No beam search** - would explore multiple hypotheses for better captions
- **No length normalization** or repetition penalties

### 6. Missing Modern Techniques
- No **attention pooling** or **CLS token** for global image features
- No **cross-modal pre-training** (like BLIP, Flamingo)
- Simple **whitespace tokenization** instead of BPE/WordPiece
- No **frozen backbone** fine-tuning strategy

### Expected Performance
| Metric    | This Model (Expected) | State-of-the-Art |
|-----------|-----------------------|------------------|
| BLEU-4    | less than 4  | 40+    |
| CIDEr     | less than 20  | 140+   |
| METEOR | less than 10 | 30+   |

### What Would Improve It?
1. Use pre-trained ViT (e.g., `timm` library) or CLIP encoder
2. Use pre-trained GPT-2 or similar as decoder
3. Train on larger datasets (COCO, Conceptual Captions)
4. Implement beam search decoding
5. Add learning rate scheduling and label smoothing
6. Use BPE tokenization (e.g., from `transformers` library)

> **Bottom Line:** This project demonstrates the *architecture* of vision-language models, not production-ready performance. It's a learning exercise in building Transformers from scratch.

## 🙏 Acknowledgments

- Original Transformer: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- Vision Transformer: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)
- Dataset: [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)

## 📄 License

This project is for educational purposes.
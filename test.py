import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu

# Import your modules
from my_transformer.models import VisionTransformer, InitializeTransformer
from dataset import Flickr30kDataset, MyCollate, Vocabulary

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================================
# Configuration (Aligned with train.py)
# ==========================================
class Config:
    # Data Paths
    CSV_FILE = 'dataset/flickr_annotations_30k.csv'
    IMG_ROOT = 'dataset/flickr30k-images/'
    CHECKPOINT_DIR = "./checkpoints"
    
    # Checkpoint options
    CHECKPOINT_PATH = "./checkpoints/vit_caption_best.pth"  # Best model from validation
    # Alternative: use specific epoch
    # CHECKPOINT_PATH = "./checkpoints/vit_caption_epoch_35.pth"
    
    # Model Params (Must match training config)
    IMG_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 1024
    NUM_ENC = 4
    NUM_DEC = 4
    MAX_SEQ_LEN = 100
    
    # Test settings
    BATCH_SIZE = 32
    NUM_VISUAL_SAMPLES = 5  # Number of samples for visual check
    MAX_EVAL_SAMPLES = 1000  # Max samples for BLEU evaluation (None for all)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint and return model with vocab."""
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'N/A')
    train_loss = checkpoint.get('train_loss', checkpoint.get('loss', 'N/A'))
    val_loss = checkpoint.get('val_loss', 'N/A')
    print(f"  Epoch: {epoch}")
    print(f"  Train Loss: {train_loss}")
    if val_loss != 'N/A':
        print(f"  Val Loss: {val_loss}")
    
    return model, checkpoint.get('vocab', None)

def create_causal_mask(seq_len):
    """Creates a mask to prevent the decoder from looking at future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(Config.DEVICE)

def generate_caption(model, image, vocab, max_len=20):
    """
    Generates a caption for a single image using Greedy Search.
    """
    model.eval()
    
    # Preprocess image
    if isinstance(image, Image.Image):
        # Apply transforms if it's a raw PIL image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = transform(image).unsqueeze(0).to(Config.DEVICE)
    else:
        # Assume it's already a tensor (1, 3, 224, 224)
        image = image.to(Config.DEVICE)
        if image.dim() == 3: image = image.unsqueeze(0)

    # Prepare indices
    sos_idx = vocab.stoi["<SOS>"]
    eos_idx = vocab.stoi["<EOS>"]
    
    with torch.no_grad():
        # 1. Encode Image
        enc_output, _ = model.encode(image, src_mask=None)
        
        # 2. Initialize Decoder Input with <SOS>
        decoder_input = torch.tensor([[sos_idx]]).to(Config.DEVICE)
        
        # 3. Autoregressive Generation
        generated_tokens = []
        for _ in range(max_len):
            # Create mask
            tgt_mask = create_causal_mask(decoder_input.size(1))

            # Decode
            dec_output, _, _ = model.decode(decoder_input, enc_output, src_mask=None, tgt_mask=tgt_mask)
            
            # Project to Vocab
            # We only care about the last token prediction
            logits = model.project(dec_output[:, -1, :]) 
            
            # Greedy selection (argmax)
            next_token_id = logits.argmax(1).item()

            # Stop if EOS
            if next_token_id == eos_idx:
                break
                
            generated_tokens.append(next_token_id)
            
            # Append to input for next step
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]]).to(Config.DEVICE)], dim=1)

    # Convert indices to words
    caption = [vocab.itos[idx] for idx in generated_tokens]
    return " ".join(caption)

def evaluate_model(model, dataset, vocab):
    """
    Calculates BLEU-4 score on the test set.
    """
    print("\n" + "="*50)
    print("Starting Quantitative Evaluation (BLEU Score)")
    print("="*50)
    
    model.eval()
    references = []
    hypotheses = []
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=MyCollate(pad_idx=vocab.stoi["<PAD>"])
    )
    
    # Determine number of samples to evaluate
    max_samples = Config.MAX_EVAL_SAMPLES if Config.MAX_EVAL_SAMPLES else len(dataset)
    
    for idx, (image, caption_tensor) in enumerate(tqdm(loader, desc="Generating Captions", total=min(max_samples, len(loader)))):
        if idx >= max_samples:
            break
        
        # Generate Hypothesis
        generated_text = generate_caption(model, image, vocab, max_len=Config.MAX_SEQ_LEN)
        hypotheses.append(generated_text.split())
        
        # Get Reference (Ground Truth)
        # Note: In a real test, we should compare against ALL 5 captions for an image.
        # dataset[i] returns one random caption. For strict BLEU, we'd need to modify 
        # the dataset to return all 5. Here we compare against the single loaded caption.
        real_caption_indices = [idx.item() for idx in caption_tensor[0] if idx.item() not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]]
        real_text = [vocab.itos[idx] for idx in real_caption_indices]
        references.append([real_text]) # standard BLEU expects list of lists of references

    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"\nResults on {len(hypotheses)} samples:")
    print(f"  BLEU-1: {bleu1 * 100:.2f}")
    print(f"  BLEU-2: {bleu2 * 100:.2f}")
    print(f"  BLEU-3: {bleu3 * 100:.2f}")
    print(f"  BLEU-4: {bleu4 * 100:.2f}")
    
    if bleu4 * 100 < 5:
        print("\nNote: Score is low. Check if model is trained or if vocab matches checkpoint.")
    elif bleu4 * 100 > 20:
        print("\nNote: Good score for a custom implementation!")
    
    return {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4}

def visualize_prediction(model, dataset, vocab):
    """
    Picks a random image, displays it, and prints Real vs Generated caption.
    """
    idx = random.randint(0, len(dataset)-1)
    image, caption_tensor = dataset[idx]
    
    # Convert image tensor back to PIL for display
    # Undo normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_disp = inv_normalize(image)
    img_disp = transforms.ToPILImage()(img_disp)
    
    # Generate
    pred_caption = generate_caption(model, image, vocab, max_len=Config.MAX_SEQ_LEN)
    
    # Decode Real
    real_indices = [i.item() for i in caption_tensor if i.item() not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]]
    real_caption = " ".join([vocab.itos[i] for i in real_indices])
    
    print("\n" + "-"*50)
    print(f"Sample #{idx}")
    print(f"  Real Caption:      {real_caption}")
    print(f"  Generated Caption: {pred_caption}")
    print("-"*50)
    
    return img_disp, real_caption, pred_caption

def main():
    print("="*50)
    print("Vision Transformer Image Captioning - Test Script")
    print("="*50)
    print(f"Device: {Config.DEVICE}")
    
    # 1. Prepare Data (Use same transforms as validation in train.py)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    print("\nLoading Test Dataset...")
    # Load train dataset first to get vocabulary
    train_dataset = Flickr30kDataset(
        csv_file=Config.CSV_FILE,
        root_dir=Config.IMG_ROOT,
        transform=transform,
        split='train'
    )
    vocab = train_dataset.vocab
    
    # Load test dataset with same vocabulary
    test_dataset = Flickr30kDataset(
        csv_file=Config.CSV_FILE,
        root_dir=Config.IMG_ROOT,
        transform=transform,
        vocab=vocab,  # Use training vocabulary
        split='test'
    )
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 2. Initialize Model
    print("\nInitializing Model...")
    model = VisionTransformer(
        num_enc=Config.NUM_ENC,
        num_dec=Config.NUM_DEC,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        d_ff=Config.D_FF,
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        target_vocab_size=len(vocab),
        max_seq_len=Config.MAX_SEQ_LEN
    ).to(Config.DEVICE)
    
    # 3. Load Checkpoint
    try:
        model, saved_vocab = load_checkpoint(Config.CHECKPOINT_PATH, model)
        # If the checkpoint saved the vocab, use that one for consistency
        if saved_vocab is not None:
            vocab = saved_vocab
            print("Using vocabulary from checkpoint.")
    except FileNotFoundError:
        print(f"\nERROR: Checkpoint file not found at {Config.CHECKPOINT_PATH}")
        print("Please train the model first or check the path.")
        print("Available checkpoints in directory:")
        import os
        if os.path.exists(Config.CHECKPOINT_DIR):
            for f in sorted(os.listdir(Config.CHECKPOINT_DIR)):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return

    # 4. Qualitative Test (Visual Check)
    print(f"\n{'='*50}")
    print(f"Running Visual Checks ({Config.NUM_VISUAL_SAMPLES} samples)")
    print("="*50)
    for _ in range(Config.NUM_VISUAL_SAMPLES):
        visualize_prediction(model, train_dataset, vocab)

    # 5. Quantitative Test (BLEU)
    evaluate_model(model, test_dataset, vocab)

if __name__ == "__main__":
    main()
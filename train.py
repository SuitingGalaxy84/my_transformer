import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm # Progress bar
import os

# Import your custom modules
from my_transformer.models import VisionTransformer, InitializeTransformer
from dataset import Flickr30kDataset, MyCollate

# ==========================================
# Configuration & Hyperparameters
# ==========================================
class Config:
    # Data paths
    CSV_FILE = 'dataset/flickr_annotations_30k.csv' # Path to your CSV
    IMG_ROOT = 'dataset/flickr30k-images/'          # Path to your image folder
    
    # Model Hyperparameters (Must match what you want to train)
    IMG_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 2048
    NUM_ENC = 4
    NUM_DEC = 4
    MAX_SEQ_LEN = 100 # Matches models.py default
    DROPOUT = 0.1
    
    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 60
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "./checkpoints"

def create_causal_mask(seq_len):
    """
    Creates a mask to prevent the decoder from looking at future tokens.
    True = mask (ignore), False = attend
    """
    # Create upper triangular matrix: True for future positions (to be masked)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(Config.DEVICE) # (1, 1, seq_len, seq_len)

def train_step(model, loader, optimizer, criterion, epoch_index):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_index+1}/{Config.NUM_EPOCHS}")

    for batch_idx, (images, captions) in enumerate(progress_bar):
        # 1. Move data to device
        images = images.to(Config.DEVICE)
        captions = captions.to(Config.DEVICE)
        # 2. Prepare Decoder Inputs and Targets for Teacher Forcing
        # Input: <SOS> ... token_n
        # Target: token_1 ... <EOS>
        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]
        

        # 3. Create Masks
        # Encoder mask is None for Vision Transformer (all patches are valid)
        src_mask = None 
        # Decoder mask (Causal)
        tgt_mask = create_causal_mask(decoder_input.size(1))
        # Padding mask for decoder (optional but recommended if batches vary in length)
        # Assuming 0 is PAD index (based on dataset.py)
        pad_mask = (decoder_input == 0).unsqueeze(1).unsqueeze(2) # (Batch, 1, 1, Seq_Len)
        # Combine causal and padding mask
        tgt_mask = tgt_mask | pad_mask.to(Config.DEVICE)

        # 4. Forward Pass
        optimizer.zero_grad()
        
        # Encode Image
        enc_output, _ = model.encode(images, src_mask)
        
        # Decode Caption
        dec_output, _, _ = model.decode(decoder_input, enc_output, src_mask, tgt_mask)
        # Project to Vocabulary
        logits = model.project(dec_output) # (Batch, Seq_Len, Vocab_Size)
        # 5. Calculate Loss
        # Flatten outputs and targets for CrossEntropyLoss
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]), 
            targets.reshape(-1)
        )

        # 6. Backward Pass
        loss.backward()
        
        # Optional: Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Update progress bar
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / len(loader)

def main():
    # 0. Setup
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    print(f"Training on device: {Config.DEVICE}")

    # 1. Prepare Data
    # Transforms must match ImageEmbedding expectations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
        transforms.ToTensor(),
        # Standard ImageNet normalization
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Loading Dataset...")
    dataset = Flickr30kDataset(
        csv_file=Config.CSV_FILE,
        root_dir=Config.IMG_ROOT,
        transform=transform,
        split='train' # Ensure your CSV has a 'split' column or remove this arg
    )
    
    # Get vocab size from the built vocabulary
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    print(f"Vocabulary Size: {vocab_size}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize Model
    print("Initializing Model...")
    model = VisionTransformer(
        num_enc=Config.NUM_ENC,
        num_dec=Config.NUM_DEC,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        d_ff=Config.D_FF,
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        target_vocab_size=vocab_size, # Dynamic based on dataset
        max_seq_len=Config.MAX_SEQ_LEN
    ).to(Config.DEVICE)

    # Apply Xavier initialization
    model = InitializeTransformer(model)

    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    
    # Ignore the <PAD> token when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 4. Training Loop
    print("Starting Training...")
    for epoch in range(Config.NUM_EPOCHS):
        avg_loss = train_step(model, loader, optimizer, criterion, epoch)
        
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        checkpoint_path = os.path.join(Config.SAVE_DIR, f"vit_caption_epoch_{epoch+1}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab': dataset.vocab # Optional: Save vocab to ensure consistency during inference
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
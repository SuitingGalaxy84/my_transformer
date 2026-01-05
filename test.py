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
# Configuration
# ==========================================
class Config:
    # Paths
    CHECKPOINT_PATH = "./checkpoints/vit_caption_epoch_20.pth" # CHANGE THIS to your best epoch
    CSV_FILE = 'dataset/flickr_annotations_30k.csv'
    IMG_ROOT = 'dataset/flickr30k-images/'
    
    # Model Params (Must match training config)
    IMG_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 2048
    NUM_ENC = 4
    NUM_DEC = 4
    MAX_SEQ_LEN = 100
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path, model, optimizer=None):
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('vocab', None)

def generate_caption(model, image, vocab, max_len=20):
    """
    Generates a caption for a single image using Greedy Search.
    """
    model.eval()
    
    # Preprocess image
    if isinstance(image, Image.Image):
        # Apply transforms if it's a raw PIL image
        transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
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
            tgt_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1)), diagonal=1).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(Config.DEVICE)

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
    print("\n" + "="*40)
    print("Starting Quantitative Evaluation (BLEU Score)")
    print("="*40)
    
    model.eval()
    references = []
    hypotheses = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # We only need a subset to get a quick estimate, or full set for accuracy
    # Using tqdm for progress
    for idx, (image, caption_tensor) in enumerate(tqdm(loader, desc="Generating Captions")):
        if idx > 1000: break # Optional: limit to 1000 samples for speed
        
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

    # Calculate BLEU
    # weights=(0.25, 0.25, 0.25, 0.25) is standard BLEU-4
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"\nBLEU-4 Score: {bleu4 * 100:.2f}")
    if bleu4 * 100 < 5:
        print("Note: Score is low. Check if model is trained or if vocab matches checkpoint.")
    elif bleu4 * 100 > 20:
        print("Note: Good score for a custom implementation!")

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
    pred_caption = generate_caption(model, image, vocab)
    
    # Decode Real
    real_indices = [i.item() for i in caption_tensor if i.item() not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]]
    real_caption = " ".join([vocab.itos[i] for i in real_indices])
    
    print("\n" + "-"*50)
    print(f"Sample #{idx}")
    print(f"Real Caption:      {real_caption}")
    print(f"Real Token IDs:   {real_indices}")
    print(f"Generated Caption: {pred_caption}")
    print(f"Generated Token IDs: {[vocab.stoi[word] for word in pred_caption.split()]}")
    print("-"*50)
    
    # Optional: Show image
    # plt.imshow(img_disp)
    # plt.title(f"Gen: {pred_caption}")
    # plt.show()

def main():
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    print("Loading Dataset & Vocab...")
    # NOTE: We use 'train' split just to ensure vocab matches if you didn't save it.
    # Ideally, you use 'test' split here.
    dataset = Flickr30kDataset(
        csv_file=Config.CSV_FILE,
        root_dir=Config.IMG_ROOT,
        transform=transform,
        split='train' 
    )
    vocab = dataset.vocab
    
    # 2. Load Model
    print("Initializing Model...")
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
    
    try:
        model, saved_vocab = load_checkpoint(Config.CHECKPOINT_PATH, model)
        # If the checkpoint saved the vocab, it's safer to use that one
        if saved_vocab is not None:
            vocab = saved_vocab
            print("Loaded vocabulary from checkpoint.")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {Config.CHECKPOINT_PATH}")
        print("Please train the model first or check the path.")
        return

    # 3. Qualitative Test (Visual Check)
    print("\nRunning Visual Checks...")
    for _ in range(3):
        visualize_prediction(model, dataset, vocab)

    # 4. Quantitative Test (BLEU)
    # evaluate_model(model, dataset, vocab) # Uncomment to run full evaluation

if __name__ == "__main__":
    main()
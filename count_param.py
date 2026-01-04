import torch
from my_transformer.models import VisionTransformer

def count_parameters(model):
    """
    Counts total and trainable parameters of a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_breakdown(model: VisionTransformer):
    """
    Prints the parameter count for specific sub-modules (Encoder vs Decoder).
    """
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    
    # Embeddings / Projections (often shared or separate depending on implementation)
    # In your models.py, LinearProjectionLayer is separate.
    linear_params = sum(p.numel() for p in model.linear.parameters())
    
    print("-" * 40)
    print(f"Encoder (Vision):   {enc_params:,}")
    print(f"Decoder (Text):     {dec_params:,}")
    print(f"Linear Projection:  {linear_params:,}")
    print("-" * 40)

if __name__ == "__main__":
    # Configuration (Matches your Demo + Safety Limit)
    CONFIG = {
        'num_enc': 4,
        'num_dec': 4,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'img_size': (224, 224),
        'patch_size': (16, 16),
        'target_vocab_size': 5000, # Example vocab size
        'max_seq_len': 100         # The safe limit we set
    }

    print(f"Initializing Model with d_model={CONFIG['d_model']}...")
    
    model = VisionTransformer(
        num_enc=CONFIG['num_enc'],
        num_dec=CONFIG['num_dec'],
        d_model=CONFIG['d_model'],
        num_heads=CONFIG['num_heads'],
        d_ff=CONFIG['d_ff'],
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        target_vocab_size=CONFIG['target_vocab_size'],
        max_seq_len=CONFIG['max_seq_len']
    )

    total, trainable = count_parameters(model)

    print("\n" + "=" * 40)
    print("MODEL PARAMETER STATISTICS")
    print("=" * 40)
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    # Show breakdown
    print_breakdown(model)
    
    # Size in MB (assuming float32 = 4 bytes)
    size_mb = total * 4 / (1024 ** 2)
    print(f"Approximate Model Size: {size_mb:.2f} MB")
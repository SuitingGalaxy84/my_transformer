"""
Simple VisionTransformer Demo Application
Demonstrates image captioning setup with the VisionTransformer model
"""

import torch
import torch.nn as nn
from my_transformer.models import VisionTransformer, InitializeTransformer


def create_causal_mask(seq_len):
    """
    Create a causal mask for decoder self-attention.
    Prevents attending to future positions.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask == 0  # Convert to boolean: True = attend, False = mask
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def demo_vision_transformer():
    """
    Demonstrate the VisionTransformer for image captioning.
    """
    # Configuration
    IMG_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    INPUT_CHANNELS = 3
    D_MODEL = 768
    NUM_HEADS = 8
    D_FF = 1024
    NUM_ENC = 4
    NUM_DEC = 4
    TARGET_VOCAB_SIZE = 5000
    MAX_SEQ_LEN = 100
    BATCH_SIZE = 2

    print("=" * 60)
    print("VisionTransformer Demo - Image Captioning Setup")
    print("=" * 60)

    # Create model
    model = VisionTransformer(
        num_enc=NUM_ENC,
        num_dec=NUM_DEC,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        input_channel=INPUT_CHANNELS,
        target_vocab_size=TARGET_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    # Initialize weights
    model = InitializeTransformer(model)

    # Print model info
    num_patches = (IMG_SIZE[0] // PATCH_SIZE[0]) * (IMG_SIZE[1] // PATCH_SIZE[1])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    print(f"  - Image Size: {IMG_SIZE}")
    print(f"  - Patch Size: {PATCH_SIZE}")
    print(f"  - Number of Patches: {num_patches}")
    print(f"  - Model Dimension (d_model): {D_MODEL}")
    print(f"  - Number of Heads: {NUM_HEADS}")
    print(f"  - Feed-Forward Dimension: {D_FF}")
    print(f"  - Encoder Layers: {NUM_ENC}")
    print(f"  - Decoder Layers: {NUM_DEC}")
    print(f"  - Target Vocab Size: {TARGET_VOCAB_SIZE}")
    print(f"  - Max Sequence Length: {MAX_SEQ_LEN}")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Create dummy input data
    print("\n" + "-" * 60)
    print("Running Forward Pass...")
    print("-" * 60)

    # Dummy image batch: (batch_size, channels, height, width)
    dummy_images = torch.randn(BATCH_SIZE, INPUT_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])
    print(f"\nInput Image Shape: {dummy_images.shape}")

    # Dummy target caption tokens: (batch_size, seq_len)
    target_seq_len = 20
    dummy_captions = torch.randint(0, TARGET_VOCAB_SIZE, (BATCH_SIZE, target_seq_len))
    print(f"Target Caption Shape: {dummy_captions.shape}")

    # Create masks
    # For image encoder, we typically don't need a mask (all patches are valid)
    src_mask = None
    
    # Causal mask for decoder (autoregressive generation)
    tgt_mask = create_causal_mask(target_seq_len)
    print(f"Target Mask Shape: {tgt_mask.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Encode the image
        enc_output, enc_attention = model.encode(dummy_images, src_mask)
        print(f"\nEncoder Output Shape: {enc_output.shape}")
        print(f"  -> (batch_size={BATCH_SIZE}, num_patches={num_patches}, d_model={D_MODEL})")

        # Decode with target caption
        dec_output, dec_self_attn, dec_cross_attn = model.decode(
            dummy_captions, enc_output, src_mask, tgt_mask
        )
        print(f"\nDecoder Output Shape: {dec_output.shape}")
        print(f"  -> (batch_size={BATCH_SIZE}, seq_len={target_seq_len}, d_model={D_MODEL})")

        # Project to vocabulary
        logits = model.project(dec_output)
        print(f"\nLogits Shape: {logits.shape}")
        print(f"  -> (batch_size={BATCH_SIZE}, seq_len={target_seq_len}, vocab_size={TARGET_VOCAB_SIZE})")

        # Get probabilities
        probs = model.project(dec_output, softmax=True)
        print(f"\nProbabilities Shape: {probs.shape}")
        print(f"  -> Sum of probs for first token: {probs[0, 0, :].sum().item():.4f} (should be ~1.0)")

        # Get predicted tokens
        predicted_tokens = torch.argmax(logits, dim=-1)
        print(f"\nPredicted Tokens Shape: {predicted_tokens.shape}")
        print(f"Sample Predicted Tokens (first batch): {predicted_tokens[0].tolist()}")

    print("\n" + "=" * 60)
    print("Demo Completed Successfully!")
    print("=" * 60)

    return model


def demo_greedy_generation():
    """
    Demonstrate greedy decoding for caption generation.
    """
    print("\n" + "=" * 60)
    print("Greedy Generation Demo")
    print("=" * 60)

    # Smaller model for demo
    model = VisionTransformer(
        num_enc=2,
        num_dec=2,
        num_heads=4,
        d_ff=512,
        img_size=(224, 224),
        patch_size=(16, 16),
        input_channel=3,
        d_model=768, # pH * pW * input_channel = 16 * 16 * 3
        target_vocab_size=1000,
        max_seq_len=30
    )
    model = InitializeTransformer(model)
    model.eval()

    # Special tokens
    BOS_TOKEN = 1  # Beginning of sentence
    EOS_TOKEN = 2  # End of sentence
    MAX_GEN_LEN = 20

    # Single image input
    image = torch.randn(1, 3, 224, 224)

    print(f"\nGenerating caption with greedy decoding...")
    print(f"  BOS Token: {BOS_TOKEN}")
    print(f"  EOS Token: {EOS_TOKEN}")
    print(f"  Max Generation Length: {MAX_GEN_LEN}")

    with torch.no_grad():
        # Encode image once
        enc_output, _ = model.encode(image, None)

        # Start with BOS token
        generated = torch.tensor([[BOS_TOKEN]])

        for step in range(MAX_GEN_LEN):
            # Create causal mask
            tgt_mask = create_causal_mask(generated.size(1))

            # Decode
            dec_output, _, _ = model.decode(generated, enc_output, None, tgt_mask)

            # Get logits for last position
            logits = model.project(dec_output[:, -1, :])

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if next_token.item() == EOS_TOKEN:
                print(f"  -> EOS reached at step {step + 1}")
                break

        print(f"\nGenerated Token Sequence: {generated[0].tolist()}")
        print(f"Sequence Length: {generated.size(1)}")

    print("\n" + "=" * 60)
    print("Greedy Generation Demo Completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demos
    demo_vision_transformer()
    demo_greedy_generation()

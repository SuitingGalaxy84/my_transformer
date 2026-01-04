from .layers import *

# Default Transformer Model
class BaseTransformer(nn.Module):

    '''
    Base Transformer Model: Father class for different Transformer variants
    
    Args:
        num_enc (int): Number of encoder layers
        num_dec (int): Number of decoder layers
        d_model (int): Dimension of model
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feedforward network
        input_vocab_size (int): Size of input vocabulary
        target_vocab_size (int): Size of target vocabulary
        max_seq_len (int): Maximum sequence length
    '''
    def __init__(
            self,
            num_enc = 4,
            num_dec = 4, 
            d_model = 512, 
            num_heads = 8, 
            d_ff = 2048, 
            input_vocab_size = 10000,
            target_vocab_size = 10000,
            max_seq_len = 100
            ):
        
        super(BaseTransformer, self).__init__()
        self.num_enc = num_enc
        self.num_dec = num_dec
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_seq_len = max_seq_len

        self.encoder = Encoder(num_enc, d_model, num_heads, d_ff, input_vocab_size, max_seq_len)
        self.decoder = Decoder(num_dec, d_model, num_heads, d_ff, target_vocab_size, max_seq_len)
        self.linear = LinearProjectionLayer(d_model, target_vocab_size)
        # self.softmax = nn.Softmax(dim=-1)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
    
    def project(self, dec_output):
        # if softmax:
        #     return self.softmax(self.linear(dec_output))
        
        return self.linear(dec_output)
    

class VisionTransformer(BaseTransformer):
    '''
    Vision Transformer Model: Child class of BaseTransformer for vision tasks e.g. image Captioning

    Args:
        img_size (tuple): Size of input images (H, W)
        patch_size (tuple): Size of image patches (h, w)
        input_channel (int): Number of input channels (e.g., 3 for RGB)
        target_vocab_size (int): Size of target vocabulary for captions
        max_seq_len (int): Maximum sequence length for captions

        Other args are inherited from BaseTransformer

    Replaced Input Embedding with Image Embedding
    '''
    def __init__(
            self,
            num_enc = 4,
            num_dec = 4,
            d_model = 512,
            num_heads = 8,
            d_ff = 2048,
            img_size = (224, 224),
            patch_size = (16, 16),
            input_channel = 3,
            target_vocab_size = 10000,
            max_seq_len = 100
            ):
        
        super(VisionTransformer, self).__init__(num_enc, num_dec, d_model, num_heads, d_ff, 
                                                 input_vocab_size=1, # Not used, replaced by ImageEmbedding
                                                 target_vocab_size=target_vocab_size, 
                                                 max_seq_len=max_seq_len)
        
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculate number of patches
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        # Replace encoder embedding with ImageEmbedding
        self.encoder.embedding = ImageEmbedding(input_channel, d_model, img_size, patch_size)
        
        # Replace encoder positional encoding to match number of patches (no CLS token)
        self.encoder.pos_encoding = PositionalEncoding(d_model, max_len=num_patches)


def InitializeTransformer(transformer: nn.Module):
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

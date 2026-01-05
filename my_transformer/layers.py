import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import quantized



class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model) # pytorch embedding layer
        # Pre-compute scaling factor
        self.scale = d_model ** 0.5
    
    def forward(self, x):
        '''
            multiply embedding by sqrt(d_model) to scale, ref. "Attention is All You Need"
        '''
        return self.embedding(x) * self.scale 
    
class ImageEmbedding(nn.Module):
    def __init__(self, input_channel: int, d_model: int, img_size: tuple, patch_size: tuple):
        super(ImageEmbedding, self).__init__()
        '''
            Args:
                image_size: tuple (H, W)
                patch_size: tuple (pH, pW)
        '''
        self.input_channel = input_channel
        self.d_model = d_model
        self.img_size = img_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size."
        
        kh = patch_size[0]
        kw = patch_size[1]

        self.conv2d = nn.Conv2d(in_channels=input_channel, out_channels=d_model, kernel_size=(kh, kw), stride=(kh, kw))
        self.scale = d_model ** 0.5

    def forward(self, img):
        x = self.conv2d(img) # Shape: (batch_size, d_model, pH, pW)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, d_model)
        return x * self.scale
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Initialize the Positional Encoding layer.
        This layer adds positional information to input embeddings using sine and cosine functions
        of different frequencies, allowing the model to understand sequence order.
        Args:
            d_model (int): The dimension of the model's embeddings.
            max_len (int, optional): Maximum sequence length to pre-compute positional encodings for.
                Defaults to 5000.
            dropout (float, optional): Dropout probability applied after adding positional encoding.
                Defaults to 0.1.
        Notes:
            The positional encoding is computed using:
                - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model)) for even dimensions
                - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) for odd dimensions
            Where:
                - pos: position in the sequence
                - i: dimension index

        """

        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # [X::Y] expression: start at X, go to the end, step by Y
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)

        # Register as buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            Add positional encoding to input embeddings. 
            Args:
                x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
            Returns:
                Tensor: Embeddings with added positional encoding, same shape as input.
        """
        try:
            x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        except Exception as e:
            print(f"Error in PositionalEncoding forward: {e}")
            print(f"x.size(): {x.size()}, pe.size(): {self.pe.size()}")
            raise e
        return self.dropout(x)
       
class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(d_model))  # Shift parameter

    def forward(self, x):
        """
        Apply layer normalization to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return self.alpha * normalized_x + self.bias
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Apply position-wise feed-forward network to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)  # Dropout after final projection
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
    
    @staticmethod
    def dot_product_attention(query, key, value, mask=None):
        d_k = query.size(-1)
        atten_score = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            # mask == True means "mask this position" (don't attend)
            atten_score = atten_score.masked_fill(mask, -1e9)
        
        atten_prob = F.softmax(atten_score, dim=-1)
        output = torch.matmul(atten_prob, value)

        atten_dict = {
            'raw': atten_score,
            'prob': atten_prob
        }

        return output, atten_dict
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.w_q(query)  # (batch_size, seq_len, d_model)
        key = self.w_k(key)      # (batch_size, seq_len, d_model)
        value = self.w_v(value)  # (batch_size, seq_len, d_model)

        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, atten_dict = MultiHeadAttention.dot_product_attention(query, key, value, mask)
        
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)
        
        output = self.dropout(output)
        
        return output, atten_dict
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        super(ResidualConnection, self).__init__()
        # Note: Dropout is applied in sublayer (MHA/FFN), not here
    
    def forward(self, x, sublayer_output):
        return x + sublayer_output
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln_1 = LayerNormalization(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.ln_2 = LayerNormalization(d_model)
        self.res_add_1 = ResidualConnection(dropout)
        self.res_add_2 = ResidualConnection(dropout)


        
    def forward(self, x, src_mask=None):
        # POST-LN Architecture
        # atten_x, atten = self.mha(x, x, x, src_mask)
        # x = self.ln_1(self.res_add_1(x, atten_x))
        
        # ffn_x = self.ffn(x)
        # x = self.ln_2(self.res_add_2(x, ffn_x))

        # PRE-LN Architecture
        x_ = self.ln_1(x)
        atten_x, atten = self.mha(x_, x_, x_, src_mask)
        x = self.res_add_1(x, atten_x)
        
        x_ = self.ln_2(x)
        ffn_x = self.ffn(x_)
        x = self.res_add_2(x, ffn_x)

        return x, atten
        
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask):

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = self.pos_encoding(x)  # (batch_size, input_seq_len, d_model)

        atten = []
        for i in range(self.num_layers):
            x, _atten = self.enc_layers[i](x, mask)
            atten.append(_atten)

        return x, atten
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mha_1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln_1 = LayerNormalization(d_model)

        self.mha_2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln_2 = LayerNormalization(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.ln_3 = LayerNormalization(d_model)

        self.res_add_1 = ResidualConnection(dropout)
        self.res_add_2 = ResidualConnection(dropout)
        self.res_add_3 = ResidualConnection(dropout)


    def forward(self, x, enc_output, src_mask, tgt_mask):

        # POST-LN Architecture
        # atten_x, atten_1 = self.mha_1(x, x, x, tgt_mask)
        # x = self.ln_1(self.res_add_1(x, atten_x))

        # atten_x, atten_2 = self.mha_2(x, enc_output, enc_output, src_mask)
        # x = self.ln_2(self.res_add_2(x, atten_x))
        
        # ffn_x = self.ffn(x)
        # x = self.ln_3(self.res_add_3(x, ffn_x))

        # PRE-LN Architecture
        x_ = self.ln_1(x)
        atten_x, atten_1 = self.mha_1(x_, x_, x_, tgt_mask)
        x = self.res_add_1(x, atten_x)
        
        x_ = self.ln_2(x)
        atten_x, atten_2 = self.mha_2(x_, enc_output, enc_output, src_mask)
        x = self.res_add_2(x, atten_x)
        
        x_ = self.ln_3(x)
        ffn_x = self.ffn(x_)
        x = self.res_add_3(x, ffn_x)

        return x, atten_1, atten_2
    
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.pos_encoding(x)  # (batch_size, target_seq_len, d_model)

        atten_1 = []
        atten_2 = []
        for i in range(self.num_layers):
            x, attn1, attn2 = self.dec_layers[i](x, enc_output, src_mask, tgt_mask)
            atten_1.append(attn1)
            atten_2.append(attn2)

        return x, atten_1, atten_2
    
class LinearProjectionLayer(nn.Module):
    def __init__(self, d_model, target_vocab_size):
        super(LinearProjectionLayer, self).__init__()
        self.d_model = d_model
        self.target_vocab_size = target_vocab_size
        self.linear = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, x):
        return self.linear(x)




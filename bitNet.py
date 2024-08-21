import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import math
import copy
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder

def absmean_quantize_weight(weight):
    scale = torch.mean(torch.abs(weight))
    weight_scaled = weight / scale
    weight_quantized = torch.round(weight_scaled).clamp(-1, 1)
    return weight_quantized * scale

def quantize_activation(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = torch.max(torch.abs(x)) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        quantized_weight = absmean_quantize_weight(self.weight)
        quantized_input = quantize_activation(input)
        return F.linear(quantized_input, quantized_weight)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class BitNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = config.relative_attention_num_buckets > 0

        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = BitLinear(self.d_model, self.inner_dim)
        self.k = BitLinear(self.d_model, self.inner_dim)
        self.v = BitLinear(self.d_model, self.inner_dim)
        self.o = BitLinear(self.inner_dim, self.d_model)

        self.rotary_emb = RotaryEmbedding(self.key_value_proj_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = q.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.key_value_proj_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        return attn_output, attn_weights

class BitNetT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = BitNetT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = BitNetT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, 
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, 
                past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, 
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        
        # Implementation remains the same as the original T5ForConditionalGeneration
        # but uses the custom encoder and decoder stacks
        pass

class BitNetT5Stack(nn.Module):
    def __init__(self, config, embed_tokens=None):
        super().__init__()
        self.config = config

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [BitNetT5Block(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, inputs_embeds=None, head_mask=None, 
                cross_attn_head_mask=None, past_key_values=None, use_cache=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None):
        
        # Implementation remains similar to the original T5Stack
        # but uses the custom BitNetT5Block
        pass

class BitNetT5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(BitNetT5LayerSelfAttention(config, has_relative_attention_bias=True))
        if self.is_decoder:
            self.layer.append(BitNetT5LayerCrossAttention(config))
        self.layer.append(BitNetT5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                encoder_decoder_position_bias=None, layer_head_mask=None, 
                cross_attn_layer_head_mask=None, past_key_value=None, 
                use_cache=False, output_attentions=False, return_dict=False):
        
        # Implementation remains similar to the original T5Block
        # but uses the custom BitNetT5LayerSelfAttention, BitNetT5LayerCrossAttention, and BitNetT5LayerFF
        pass

class BitNetT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = BitNetAttention(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, 
                layer_head_mask=None, past_key_value=None, use_cache=False, 
                output_attentions=False):
        
        # Implementation remains similar to the original T5LayerSelfAttention
        # but uses the custom BitNetAttention and RMSNorm
        pass

class BitNetT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = BitNetAttention(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, 
                position_bias=None, layer_head_mask=None, past_key_value=None, 
                use_cache=False, query_length=None, output_attentions=False):
        
        # Implementation remains similar to the original T5LayerCrossAttention
        # but uses the custom BitNetAttention and RMSNorm
        pass

class BitNetT5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = BitNetDenseReluDense(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class BitNetDenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = BitLinear(config.d_model, config.d_ff * 2)
        self.wo = BitLinear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = SwiGLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

def preprocess_function(examples):
    inputs = examples["article"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, return_tensors='pt')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

config = T5Config.from_pretrained("t5-small")
model = BitNetT5(config)

api = HfApi()
# Create the repository
api.create_repo(repo_id="Rajpreet2206/bitnet-t5", private=False) 
model_name="bitnet-t5"
model.save_pretrained(model_name)
#tokenizer.save_pretrained(model_name)
token = HfFolder.get_token()

# Upload the model to your repository
api.upload_folder(
    folder_path=model_name,   # path to the folder containing the model and tokenizer
    path_in_repo="",          # leave empty to push directly to the model repo
    repo_id="Rajpreet2206/bitnet-t5",  # replace with your Hugging Face username and desired repo name
    repo_type="model",
    token=token,
)
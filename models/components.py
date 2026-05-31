import torch
import torchvision
import torch.nn as nn
import numpy as np

from utils.hrnet import hrnet_w32

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True):
        super(Encoder, self).__init__()

        if encoder == 'swin':
            '''Swin Transformer encoder'''
            self.encoder = torchvision.models.swin_b(weights='DEFAULT')
            self.encoder.head = nn.GELU()
        elif encoder == 'hrnet':
            '''HRNet encoder'''
            self.encoder = hrnet_w32(pretrained=pretrained)
        else:
            raise NotImplementedError('Encoder not implemented')

    def forward(self, x):
        out = self.encoder(x)
        return out  

class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, out_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, q, k, v):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height = q.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(q.permute(0, 2, 1))
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(k.permute(0, 2, 1))
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(v.permute(0, 2, 1))
        # out: B x C x N
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, C, height)
        out = out/np.sqrt(self.channel_in)
        
        return out

class Cross_Att(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cross_Att, self).__init__()

        self.cross_attn_1 = Self_Attn(in_dim, out_dim)
        self.cross_attn_2 = Self_Attn(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm([1, in_dim])

    def forward(self, sem_seg, part_seg):
        cross1 = self.cross_attn_1(sem_seg, part_seg, part_seg)
        cross2 = self.cross_attn_1(part_seg, sem_seg, sem_seg)

        out = cross1 * cross2
        out = self.layer_norm(out)

        return out

class Spatial_Cross_Att(nn.Module):
    """Bidirectional spatial cross-attention between two token sequences.

    Unlike Cross_Att (which collapses each branch to a single token and then mixes the
    1280 channels), this attends across the H*W spatial locations, so the two feature
    maps exchange information per-location before aggregation. This keeps the spatial
    detail that global pooling discards. Inputs / output are (B, N, dim) token tensors.
    """
    def __init__(self, in_dim, num_heads=8):
        super(Spatial_Cross_Att, self).__init__()
        self.attn_sem = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.attn_part = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.norm_sem = nn.LayerNorm(in_dim)
        self.norm_part = nn.LayerNorm(in_dim)
        self.norm_out = nn.LayerNorm(in_dim)

    def forward(self, sem_seg, part_seg):
        # sem tokens attend to part tokens (and vice-versa), each with a residual + norm
        sem_x, _ = self.attn_sem(sem_seg, part_seg, part_seg)       # , need_weights=False
        part_x, _ = self.attn_part(part_seg, sem_seg, sem_seg)      # , need_weights=False
        sem_out = self.norm_sem(sem_seg + sem_x)
        part_out = self.norm_part(part_seg + part_x)
        # DECO-style multiplicative fusion of the two streams
        out = self.norm_out(sem_out * part_out)
        return out

class _VertexDecoderLayer(nn.Module):
    """One cross-attention + FFN block (no query self-attention)."""
    def __init__(self, dim, num_heads, ffn_dim):
        super(_VertexDecoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, mem):
        attn_out, _ = self.cross_attn(q, mem, mem)   # vertex queries attend to image tokens    # , need_weights=False
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q

class VertexContactDecoder(nn.Module):
    """Per-vertex contact head: learnable vertex queries cross-attend to image tokens.

    Replaces the global-vector MLP `Classifier`. Each of `num_vertices` SMPL vertices is
    a learnable query that attends to the fused image tokens (memory), so contact is
    decoded per-vertex with spatial grounding instead of from one pooled descriptor.

    Cross-attention only (no query-query self-attention): a 6890x6890 self-attention map
    would be ~prohibitive in memory, whereas 6890 queries x N image keys is cheap.
    Output is (B, num_vertices) contact probabilities in [0, 1] (sigmoid), matching the
    BCELoss the trainer uses.
    """
    def __init__(self, context_dim, num_vertices=6890, dim=256, num_heads=8,
                 num_layers=3, ffn_dim=1024):
        super(VertexContactDecoder, self).__init__()
        self.num_vertices = num_vertices
        self.query = nn.Embedding(num_vertices, dim)
        self.mem_proj = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()
        self.layers = nn.ModuleList(
            [_VertexDecoderLayer(dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, memory):
        # memory: (B, N, context_dim) fused image tokens
        B = memory.shape[0]
        mem = self.mem_proj(memory)                              # (B, N, dim)
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)     # (B, V, dim)
        for layer in self.layers:
            q = layer(q, mem)
        logits = self.head(q).squeeze(-1)                        # (B, V)
        return torch.sigmoid(logits)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder='hrnet'):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        if encoder == 'swin':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        elif encoder == 'hrnet':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                # nn.ReLU(),
                # nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        elif encoder == 'sam_vit':
            # SAM-3D-Body (DINOv3/ViT) features are at 1/16 resolution, e.g. 16x16 for a
            # 256x256 input. Upsample x16 with 4 transposed convs to reach 256x256.
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        else:
            raise NotImplementedError('Decoder not implemented')

    def forward(self, x):
        out = self.upsample(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim=6890):
        super(Classifier, self).__init__()

        self.out_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4096, True), 
            nn.ReLU(),
            nn.Linear(4096, out_dim, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out.reshape(-1, self.out_dim)
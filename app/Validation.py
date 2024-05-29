import os
import warnings
from typing import List
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from MedViT.MedViT import MedViT, MedViT_large
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchdistill.core.forward_hook import ForwardHookManager
from io import BytesIO
import base64

import sys
sys.path.append('..')

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

class MedViTConfig(PretrainedConfig):
    model_type = "medvit"

    def __init__(
        self,
        stem_chs: List[int] = [64, 32, 64],
        depths: List[int] = [3, 4, 30, 3],
        path_dropout: float = 0.2,
        attn_drop: int = 0,
        drop: int = 0,
        num_classes: int = 5,
        strides: List[int] = [1, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        head_dim: int = 32,
        mix_block_ratio: float = 0.75,
        use_checkpoint: bool = False,
        pretrained: bool = False,
        pretrained_cfg: str = None,
        **kwargs
    ):
        self.stem_chs = stem_chs
        self.depths = depths
        self.path_dropout = path_dropout
        self.attn_drop = attn_drop
        self.drop = drop
        self.num_classes = num_classes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        self.pretrained_cfg = pretrained_cfg
        super().__init__(**kwargs)

class MedViTClassification(PreTrainedModel):
    config_class = MedViTConfig

    def __init__(self, config, pretrained=False):
        super().__init__(config)

        if pretrained is False:
            print('Initialized with random weights:')
            self.model = MedViT(
                stem_chs=config.stem_chs,
                depths=config.depths,
                path_dropout=config.path_dropout,
                attn_drop=config.attn_drop,
                drop=config.drop,
                num_classes=config.num_classes,
                strides=config.strides,
                sr_ratios=config.sr_ratios,
                head_dim=config.head_dim,
                mix_block_ratio=config.mix_block_ratio,
                use_checkpoint=config.use_checkpoint)
        else:
            print('Initialized with pretrained weights:')
            self.model = MedViT_large(use_checkpoint=config.use_checkpoint)
            self.model.proj_head = nn.Linear(1024, 5)

    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
def extract_attention_map(model, image_tensor, head_dim=32, device='cpu'):
    """
    image_tensore with shape batch_size x 3 x width x height    
    """
    forward_hook_manager = ForwardHookManager(model)
    
    forward_hook_manager.add_hook(model, 'model.features.-1.e_mhsa.q', requires_input=False, requires_output=True)
    forward_hook_manager.add_hook(model, 'model.features.-1.e_mhsa.k', requires_input=False, requires_output=True)

    with torch.no_grad():
        _ = model(image_tensor)

    io_dict = forward_hook_manager.pop_io_dict()
    # print(io_dict['model.features.-1.e_mhsa.q']['output'].shape)

    q = io_dict['model.features.-1.e_mhsa.q']['output'].squeeze()
    k = io_dict['model.features.-1.e_mhsa.k']['output'].squeeze()

    # print(q.shape, k.shape)
    
    num_heads = int(768 / head_dim)
    q = q.reshape(q.shape[0], num_heads, q.shape[1] // num_heads).permute(1, 0, 2)  # 256, 32, 24 -> 32, 256, 24
    k = k.reshape(k.shape[0], num_heads, k.shape[1] // num_heads).permute(1, 0, 2)  # 32, 256, 24
    kT = k.permute(0, 2, 1)  # 32, 24, 256
    attention_matrix = q @ kT  # 32, 256, 24 x 32, 24, 256   -> 32, 256x256

    # Average the attention weights across all heads.
    attention_matrix_mean = torch.mean(attention_matrix, dim=0)  # 256x256
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(attention_matrix_mean.size(1)).to(device)
    aug_att_mat = attention_matrix_mean + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
    cmaps = {}

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))


    def plot_color_gradients(category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
        axs[0].set_title(f'{category}', fontsize=14)

        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=matplotlib.colormaps[name])
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
         ax.set_axis_off()

        # Save colormap list for later.
        cmaps[category] = cmap_list

    attn_heatmap = joint_attentions[0, :].reshape((16,16))
    attn_heatmap_resized = torch.nn.functional.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0), [512, 512], mode='bilinear').view(512, 512, 1)

    return attn_heatmap_resized

def plot_attention_map(image, attention_map, cmap='jet'):
    fig, ax = plt.subplots(figsize=(6, 6))
    size = attention_map.squeeze().detach().cpu().numpy().shape
    image_ = image.resize(size)
    ax.imshow(image_)
    ax.imshow(attention_map.squeeze().detach().cpu().numpy(), alpha=0.5, cmap=cmap)
    ax.axis('off')

    # Convert attention map to base64
    attention_buffer = BytesIO()
    plt.savefig(attention_buffer, format='png', bbox_inches='tight')
    plt.close()
    attention_data = base64.b64encode(attention_buffer.getvalue()).decode()

    return attention_data

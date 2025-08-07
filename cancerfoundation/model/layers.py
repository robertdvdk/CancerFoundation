from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones


class MHA(nn.Module):
    """A custom Multi-Head Attention layer for a perceptual-generative model.

    This layer wraps PyTorch's MultiheadAttention but is designed to handle two separate input sequences: a perceptual (context) sequence and a generative (target) sequence.
    It combines them, and applies an attention mask where the generative sequence attends to the perceptual sequence and not to other tokens in the generative sequence.
    Finally, it splits the output back into perceptual and generative parts.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes the custom MHA layer.

        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            bias (bool, optional): If `True`, add a learnable bias to the input and output projections. Defaults to True.
            batch_first (bool, optional): If `True`, then the input and output tensors are provided as (batch, seq, feature). Defaults to True.
            attention_dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
            causal (bool, optional): If `True`, apply a causal mask to the attention scores. Defaults to False.
            device: The desired device of the parameters and buffers in this module.
            dtype: The desired floating point type of the parameters and buffers in this module.
        """
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
    ):
        """Performs the forward pass with perceptual and generative sequences.

        Args:
            pcpt_total_embs (Tensor): The perceptual (context) sequence embeddings of shape (batch, pcpt_len, embed_dim).
            gen_total_embs (Optional[Tensor]): The generative (target) sequence embeddings of shape (batch, gen_len, embed_dim). Can be None.
            pcpt_key_padding_mask (Optional[Tensor], optional): Mask for the perceptual sequence where `True` indicates a padded element. Defaults to None.
            gen_key_padding_mask (Optional[Tensor], optional): Mask for the generative sequence where `True` indicates a padded element. Defaults to None.
            need_weights (bool, optional): If `True`, returns attention weights. Currently not supported. Defaults to False.

        Returns:
            A tuple containing two elements:
            1. A tuple of (perceptual_output, generative_output) tensors.
            2. A tuple of (None, None) as a placeholder for attention weights.
        """
        assert not need_weights

        if gen_total_embs is None:
            return self._forward_perceptual(pcpt_total_embs, pcpt_key_padding_mask)
        pcpt_seq_len, gen_seq_len = (
            pcpt_total_embs.shape[1],
            0 if gen_total_embs is None else gen_total_embs.shape[1],
        )
        total_seq_len = pcpt_seq_len + gen_seq_len

        if pcpt_key_padding_mask is None:
            pcpt_key_padding_mask = (
                torch.zeros(pcpt_total_embs.shape[0], pcpt_seq_len)
                .bool()
                .to(pcpt_total_embs.device)
            )
        if gen_key_padding_mask is None:
            gen_key_padding_mask = (
                torch.zeros(gen_total_embs.shape[0], gen_seq_len)
                .bool()
                .to(pcpt_total_embs.device)
            )

        key_padding_mask = torch.cat(
            [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
        ).to(pcpt_total_embs.device)

        @lru_cache(maxsize=1)
        def make_mask(len, gen_len, device):
            """Creates a custom attention mask for the combined sequence.

            The mask prevents all tokens from attending to the generative tokens,
            effectively treating the perceptual sequence as context for generating the entire generative sequence at once.
            """
            attn_mask = torch.zeros(len, len).bool()
            attn_mask[:, -gen_len:] = True
            attn_mask.diagonal().fill_(False)
            return attn_mask.to(device)

        attn_mask = make_mask(total_seq_len, gen_seq_len, pcpt_total_embs.device)

        total_embs = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)

        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(pcpt_total_embs.device),
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        return (out[:, :pcpt_seq_len], out[:, pcpt_seq_len:]), (None, None)

    def _forward_perceptual(self, total_embs, key_padding_mask):
        """A simplified forward pass for only the perceptual sequence."""
        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(key_padding_mask.device),
        )
        return (out, None), (None, None)


class CFLayer(nn.Module):
    """A custom Transformer Encoder Layer for the perceptual-generative model.

    This layer is composed of a custom Multi-Head Attention module (`MHA`) and a standard feed-forward network.
    It supports both "pre-norm" and "post-norm" layer normalization schemes.
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        """Initializes the custom Transformer Encoder Layer.

        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multi-head attention models.
            dim_feedforward (int, optional): The dimension of the feed-forward network. Defaults to 2048.
            dropout (float, optional): The dropout value. Defaults to 0.1.
            activation (str, optional): The activation function ('relu' or 'gelu'). Defaults to "relu".
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-5.
            batch_first (bool, optional): If `True`, inputs are (batch, seq, feature). Defaults to True.
            device: The desired device of the parameters and buffers.
            dtype: The desired floating point type of the parameters and buffers.
            norm_scheme (str, optional): The normalization scheme, either "pre" or "post". Defaults to "post".
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        """Returns the specified activation function."""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Passes the perceptual and generative sequences through the encoder layer.

        Args:
            pcpt_total_embs (Tensor): The perceptual sequence embeddings.
            gen_total_embs (Optional[Tensor]): The generative sequence embeddings.
            pcpt_key_padding_mask (Optional[Tensor], optional): Mask for the perceptual sequence.
            gen_key_padding_mask (Optional[Tensor], optional): Mask for the generative sequence.

        Returns:
            A tuple of (perceptual_output, generative_output) tensors after processing.
        """
        pcpt_key_padding_mask_ = pcpt_key_padding_mask
        gen_key_padding_mask_ = gen_key_padding_mask

        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
        else:
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm1(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)

        return pcpt_total_embs, gen_total_embs


class CFGenerator(nn.Module):
    """A Transformer Encoder composed of a stack of custom CFLayer layers.

    This module sequentially processes input sequences through multiple `CFLayer` instances to produce final contextualized embeddings.
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        """Initializes the custom Transformer Encoder.

        Args:
            encoder_layer (CFLayer): An instance of the CFLayer class.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            norm (Optional[nn.Module], optional): An optional final layer normalization.
            mask_check (bool, optional): If `True`, performs checks on mask data types. Defaults to True.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Passes the input through the stack of encoder layers.

        Args:
            pcpt_total_embs (Tensor): The perceptual sequence embeddings.
            gen_total_embs (Optional[Tensor]): The generative sequence embeddings.
            pcpt_key_padding_mask (Optional[Tensor], optional): Mask for the perceptual sequence.
            gen_key_padding_mask (Optional[Tensor], optional): Mask for the generative sequence.

        Returns:
            A tuple of the final (perceptual_output, generative_output) tensors.
        """
        if pcpt_key_padding_mask is not None:
            _skpm_dtype = pcpt_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                pcpt_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        for mod in self.layers:
            pcpt_total_embs, gen_total_embs = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
            )

        if self.norm is not None:
            pcpt_total_embs = self.norm(pcpt_total_embs)
            gen_total_embs = self.norm(gen_total_embs)

        return pcpt_total_embs, gen_total_embs

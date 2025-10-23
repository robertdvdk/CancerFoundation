from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones


class RefactoredCFGenerator(nn.Module):
    """
    A refactored Transformer Encoder that uses standard PyTorch components.

    This module replicates the behavior of the custom CFGenerator by handling
    the concatenation of perceptual/generative sequences, creating the specific
    attention mask, and splitting the outputs. This logic is wrapped around
    a standard `torch.nn.TransformerEncoder`.

    Performance optimizations:
    - Uses standard PyTorch TransformerEncoder for optimized kernels
    - Caches attention masks to avoid recreation on every forward pass
    - Single concatenation/split operation vs per-layer operations
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_scheme: str = "post",
        norm: Optional[nn.Module] = None,
    ):
        """
        Initializes the refactored Transformer Encoder.

        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multi-head attention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feed-forward network.
            dropout (float): The dropout value.
            activation (str): The activation function ('relu' or 'gelu').
            layer_norm_eps (float): The epsilon value for layer normalization.
            batch_first (bool): If True, inputs are (batch, seq, feature).
            norm_scheme (str): The normalization scheme, "pre" or "post".
            norm (Optional[nn.Module]): An optional final layer normalization.
        """
        super().__init__()
        assert batch_first, "This implementation requires batch_first=True"
        assert norm_scheme in [
            "pre",
            "post",
        ], "norm_scheme must be either 'pre' or 'post'"

        # Map norm_scheme to the 'norm_first' parameter in the standard layer
        norm_first = True if norm_scheme == "pre" else False

        # Create a standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        # Create the standard Transformer Encoder by stacking the layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

        # Cache for attention masks to avoid recreation on every forward pass
        self._mask_cache = {}

    def _create_attention_mask(
        self, pcpt_seq_len: int, gen_seq_len: int, device: torch.device
    ) -> Tensor:
        """
        Create and cache the attention mask for the perceptual-generative architecture.

        The mask prevents all tokens from attending to generative tokens (except themselves).
        This enables parallel generation of all generative tokens conditioned on perceptual tokens.

        Args:
            pcpt_seq_len (int): Length of the perceptual sequence.
            gen_seq_len (int): Length of the generative sequence.
            device (torch.device): Device to place the mask on.

        Returns:
            Tensor: Boolean attention mask of shape (total_seq_len, total_seq_len).
        """
        total_seq_len = pcpt_seq_len + gen_seq_len
        cache_key = (pcpt_seq_len, gen_seq_len)

        # Check if mask is already cached
        if cache_key not in self._mask_cache:
            # Create the custom attention mask
            # Shape: (total_seq_len, total_seq_len)
            src_mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool)
            src_mask[:, pcpt_seq_len:] = True  # Block attention to generative tokens
            src_mask.diagonal().fill_(False)  # Allow tokens to attend to themselves
            self._mask_cache[cache_key] = src_mask

        # Return cached mask on the correct device
        return self._mask_cache[cache_key].to(device)

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Optional[Tensor],
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Passes the perceptual and generative sequences through the encoder.

        Args:
            pcpt_total_embs (Tensor): The perceptual sequence embeddings.
            gen_total_embs (Optional[Tensor]): The generative sequence embeddings.
            pcpt_key_padding_mask (Optional[Tensor]): Mask for the perceptual sequence.
            gen_key_padding_mask (Optional[Tensor]): Mask for the generative sequence.

        Returns:
            A tuple of the final (perceptual_output, generative_output) tensors.
        """
        # If there's no generative sequence, process only the perceptual one
        if gen_total_embs is None:
            output = self.transformer_encoder(
                src=pcpt_total_embs, src_key_padding_mask=pcpt_key_padding_mask
            )
            return output, None

        # --- Pre-processing Step ---
        pcpt_seq_len = pcpt_total_embs.shape[1]
        gen_seq_len = gen_total_embs.shape[1]

        # 1. Concatenate inputs
        src = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)

        # 2. Create the combined key padding mask
        src_key_padding_mask = None
        if pcpt_key_padding_mask is not None or gen_key_padding_mask is not None:
            # Both masks should be provided together for consistency
            assert (
                pcpt_key_padding_mask is not None
            ), "If any mask is provided, both must be provided"
            assert (
                gen_key_padding_mask is not None
            ), "If any mask is provided, both must be provided"

            src_key_padding_mask = torch.cat(
                (pcpt_key_padding_mask, gen_key_padding_mask), dim=1
            )

        # 3. Get the cached attention mask
        src_mask = self._create_attention_mask(pcpt_seq_len, gen_seq_len, src.device)

        # --- Call the standard Transformer Encoder ---
        output = self.transformer_encoder(
            src=src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        # --- Post-processing Step ---
        # Split the output back into perceptual and generative parts
        pcpt_output = output[:, :pcpt_seq_len]
        gen_output = output[:, pcpt_seq_len:]

        return pcpt_output, gen_output


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

        # Padding masks should be provided by the caller to avoid dynamic tensor creation
        # This is important for torch.compile compatibility
        assert (
            pcpt_key_padding_mask is not None
        ), "pcpt_key_padding_mask must be provided"
        assert gen_key_padding_mask is not None, "gen_key_padding_mask must be provided"

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
        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask,
                gen_key_padding_mask=gen_key_padding_mask,
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
                pcpt_key_padding_mask=pcpt_key_padding_mask,
                gen_key_padding_mask=gen_key_padding_mask,
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

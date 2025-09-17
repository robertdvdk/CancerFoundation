from typing import Dict, Mapping, Optional, Union, Type, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .grad_reverse import grad_reverse
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)  # forbid the big MATH fallback


def with_sdp_kernel(func):
    """Decorator to run a function within the Scaled Dot-Product Attention kernel context."""

    def wrapped_func(*args, **kwargs):
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return func(*args, **kwargs)

    return wrapped_func


class TransformerModule(nn.Module):
    """The main Transformer model for gene expression modeling.

    This model can be configured for both perceptual (masked language model-style) and generative tasks.
    It handles gene and expression value encoding, optional conditional information,
    and can be extended with modules for Masked Value Prediction for Cell-embeddings (MVC) and Domain Adversarial Training (DAT).
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        out_dim: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        pad_value: int,
        pad_token_id: int,
        criterion,
        activation: Callable[[Tensor], Tensor],
        do_mvc: bool,
        dropout: float,
        conditions: Dict,
        input_emb_style: str,
        n_input_bins: Optional[int],
        cell_emb_style: str,
        mvc_decoder_style: str,
        explicit_zero_prob: bool,
        use_generative_training: bool,
        norm_first: bool,
        do_dat: bool,
        batchnorm: bool,
        weight_conditionloss: float,
    ):
        """Initializes the TransformerModule.

        Args:
            ntoken (int): The number of unique gene tokens.
            d_model (int): The dimensionality of the model embeddings.
            out_dim (int): The output dimension of the decoders.
            nhead (int): The number of attention heads in the transformer.
            d_hid (int): The dimension of the feedforward network model in the transformer.
            nlayers (int): The number of transformer encoder layers.
            pad_value (int): The value used for padding in the input expression values.
            pad_token_id (int): The token ID used for padding.
            criterion: The loss function for expression prediction.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            do_mvc (bool, optional): Whether to include the MVC decoder. Defaults to False.
            conditions (Dict, optional): A dictionary defining conditional variables, mapping condition names to the number of categories. Defaults to None.
            input_emb_style (str, optional): The style of input value embedding ("continuous", "category", "scaling"). Defaults to "continuous".
            n_input_bins (Optional[int], optional): The number of bins for categorical value embedding. Required if `input_emb_style` is "category". Defaults to None.
            cell_emb_style (str, optional): The method to obtain cell embeddings ("cls", "avg-pool", "w-pool"). Defaults to "cls".
            mvc_decoder_style (str, optional): The architecture for the MVC decoder. Defaults to "inner product".
            explicit_zero_prob (bool, optional): Whether to explicitly predict zero-expression probability. Defaults to False.
            use_generative_training (bool, optional): Whether to use the generative training setup. Defaults to False.
            pre_norm (bool, optional): Whether to use pre-layer normalization in the transformer. Defaults to False.
            do_dat (bool, optional): Whether to include Domain Adversarial Training. Defaults to False.
            batchnorm (bool, optional): Whether to use batch normalization on the input embeddings. Defaults to False.
        """
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.conditions = conditions
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.pad_token_id = pad_token_id
        self.norm_scheme = "pre" if norm_first else "post"
        self.weight_conditionloss = weight_conditionloss

        self.n_input_bins = n_input_bins
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoricalValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        self.do_dat = do_dat
        self.do_mvc = do_mvc
        self.criterion_conditions = nn.CrossEntropyLoss()
        self.criterion = criterion
        if conditions:
            self.condition_encoders = nn.ModuleDict({})
            for cond_name, cond_num in self.conditions.items():
                self.condition_encoders[cond_name] = ConditionEncoder(cond_num, d_model)

            if do_dat:
                self.grad_reverse_discriminators = nn.ModuleDict({})
                for cond_name, cond_num in self.conditions.items():
                    self.grad_reverse_discriminators[cond_name] = (
                        AdversarialDiscriminator(
                            d_model,
                            n_cls=cond_num,
                        )
                    )

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            conditions=self.conditions,
            out_dim=out_dim,
        )

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                conditions=self.conditions,
                out_dim=out_dim,
            )

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes the weights of the gene embedding layer."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
    ) -> Tensor:
        """Encodes gene IDs and expression values into contextual embeddings. This method is used during perceptual (non-generative) training.

        Args:
            src (Tensor): Input gene token IDs of shape (batch, seq_len).
            values (Tensor): Input expression values of shape (batch, seq_len).
            src_key_padding_mask (Tensor): Padding mask of shape (batch, seq_len).
            conditions (Optional[Dict], optional): Dictionary of condition tensors. Defaults to None.
            domain_labels (Optional[Tensor], optional): Domain labels for domain-specific batch normalization. Defaults to None.

        Returns:
            Tensor: The output of the Transformer encoder, of shape (batch, seq_len, embsize).
        """
        self._check_condition_labels(conditions)

        self.cur_gene_token_embs = src  # exclude the condition token

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        values = torch.cat(
            [
                values,
                torch.zeros(values.shape[0], 1, values.shape[2]).to(values.device),
            ],
            dim=1,
        )

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Optional[Tensor] = None
    ) -> Tensor:
        """Extracts cell embeddings from the transformer's output layer.

        Args:
            layer_output (Tensor): The transformer output tensor of shape (batch, seq_len, embsize).
            weights (Optional[Tensor], optional): A tensor of weights of shape (batch, seq_len) used only when `self.cell_emb_style` is "w-pool".

        Returns:
            Tensor: The extracted cell embeddings of shape (batch, embsize).
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:  # self.cell_emb_style == "w-pool"
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_condition_labels(
        self, condition_labels: Optional[Tensor] = None
    ) -> None:
        """Validates that condition labels are provided if and only if conditions are defined for the model."""
        assert bool(self.conditions) == bool(condition_labels)

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        condition_emb: Optional[Tensor] = None,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """Extends the output dictionary with cell embeddings and optional predictions.

        Args:
            output (Mapping[str, Tensor]): The dictionary of current outputs.
            transformer_output (Tensor): The raw output from the transformer encoder.
            condition_emb (Optional[Tensor], optional): The embedding for conditional variables. Defaults to None.
            do_sample (bool, optional): If True, samples from the Bernoulli distribution for zero-inflation. Defaults to False.

        Returns:
            Mapping[str, Tensor]: The extended output dictionary.
        """
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.conditions
                else torch.cat(
                    [cell_emb, condition_emb.view(condition_emb.shape[0], -1)], dim=1
                ),
                self.cur_gene_token_embs,
            )
            output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)

        if self.do_dat:
            if self.conditions:
                output["condition_output"] = {}
                for (
                    cond_name,
                    discriminator,
                ) in self.grad_reverse_discriminators.items():
                    output["condition_output"][cond_name] = discriminator(cell_emb)

        return output

    def _prepare_perceptual_input(self, tensors: dict[str, torch.Tensor]):
        """Prepares tensors for the perceptual forward pass."""
        input_gene_ids = tensors["gene"]
        input_values = tensors["masked_expr"]

        col = torch.zeros(tensors["expr"].shape[0], 1).bool().to(tensors["expr"].device)
        src_key_padding_mask = torch.cat(
            [input_gene_ids.eq(self.pad_token_id), col], dim=1
        )

        col = torch.fill(torch.zeros(tensors["expr"].shape[0], 1), -2).to(
            tensors["expr"].device
        )
        target_values = torch.cat([tensors["expr"], col], dim=1)

        return input_gene_ids, input_values, src_key_padding_mask, target_values

    def forward(
        self, tensors: dict[str, torch.Tensor], use_cell_embedding: bool = False
    ) -> Mapping[str, Tensor]:
        """Main forward pass that dispatches to generative or perceptual mode.

        This wrapper determines the training mode based on the `use_generative_training` attribute,
        computes the primary predictions and losses, and adds auxiliary losses from MVC, DAT, and a generative consistency loss.

        Args:
            tensors (dict[str, torch.Tensor]): A dictionary of input tensors from the dataloader.
            use_cell_embedding (bool, optional): If True, a consistency loss is added by feeding the cell embedding back into the generative forward pass. Defaults to False.

        Returns:
            Mapping[str, Tensor]: A dictionary of losses for training.
        """
        loss_dict = {}
        conditions_batch = tensors["conditions"] if self.conditions else None
        input_gene_ids, input_values, src_key_padding_mask, target_values = (
            self._prepare_perceptual_input(tensors)
        )
        output_dict = self.perceptual_forward(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            conditions=conditions_batch,
        )

        output_values = output_dict["mlm_output"]
        positions_to_match = ~src_key_padding_mask & (target_values != -2)
        loss = loss_expr = self.criterion(
            output_values, target_values, positions_to_match
        )
        loss_dict["loss_expr"] = loss_expr

        if self.do_mvc:
            loss_mvc = self.criterion(
                output_dict["mvc_output"], target_values, positions_to_match
            )
            loss = loss + loss_mvc
            loss_dict["loss_mvc"] = loss_mvc

        if self.do_dat:
            if self.conditions:
                for condition in self.conditions:
                    condition_loss = (
                        self.criterion_conditions(
                            output_dict["condition_output"][condition],
                            conditions_batch[condition].squeeze(),
                        )
                        * self.weight_conditionloss
                    )
                    loss_dict["condition_acc_" + condition] = (
                        (
                            conditions_batch[condition]
                            == output_dict["condition_output"][condition]
                            .argmax(dim=-1)
                            .unsqueeze(1)
                        )
                        .float()
                        .mean()
                    )

                    loss += condition_loss / len(self.conditions)

                    loss_dict["condition_" + condition] = condition_loss.detach() / len(
                        self.conditions
                    )

        loss_dict["total_loss"] = loss

        return loss_dict

    def training_step(self, batch, batch_idx):
        """Performs a single training step (for PyTorch Lightning).

        Args:
            batch: The batch of data from the DataLoader.
            batch_idx: The index of the batch.

        Returns:
            The total loss for the batch.
        """
        loss_dict = self(batch, use_cell_embedding=False)
        return loss_dict["total_loss"]

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """Forward pass for the perceptual (MLM-style) training mode.

        Args:
            src (Tensor): Input token IDs, shape [batch_size, seq_len].
            values (Tensor): Input expression values (with masking), shape [batch_size, seq_len].
            src_key_padding_mask (Tensor): Mask for src, shape [batch_size, seq_len].
            conditions (Optional[Dict], optional): Dictionary of condition tensors. Defaults to None.
            MVC (bool, optional): If True, computes MVC output. Defaults to False.
            do_sample (bool, optional): If True, samples from Bernoulli for zero predictions. Defaults to False.

        Returns:
            Mapping[str, Tensor]: A dictionary containing MLM predictions ('mlm_output'), cell embeddings ('cell_emb'), and other optional outputs.
        """
        if self.conditions:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](cond_values).unsqueeze(1)
                    for cond_name, cond_values in conditions.items()
                ],
                dim=1,
            )
            src = self.encoder(src)
            src = torch.cat([condition_emb, src], dim=1)
        transformer_output = self.encode(src, values, src_key_padding_mask, conditions)
        output = {}
        mlm_output = self.decoder(transformer_output)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)

        output = self._extend_output(
            output,
            transformer_output,
            condition_emb=condition_emb if self.conditions else None,
            do_sample=do_sample,
        )

        return output


class GeneEncoder(nn.Module):
    """Embeds integer gene IDs. Allows the model to learn a distinct representation for each gene."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """Initializes the gene encoder.

        Args:
            num_embeddings (int): The total number of unique genes.
            embedding_dim (int): The dimensionality of the gene embeddings.
            padding_idx (Optional[int], optional): The index of the padding token. Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes a batch of gene IDs.

        Args:
            x (Tensor): A tensor of gene IDs of shape (batch, seq_len).

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, embsize).
        """
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """Embeds continuous gene expression values using a small feed-forward network. Used when the input values aren't binned."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """Embeds a batch of continuous expression values.

        Args:
            x (Tensor): A tensor of expression values of shape (batch, seq_len).

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, d_model).
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoricalValueEncoder(nn.Module):
    """Embeds discretized (binned) gene expression values using an embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """Initializes the categorical value encoder.

        Args:
            num_embeddings (int): The number of discrete bins for expression values.
            embedding_dim (int): The dimensionality of the value embeddings.
            padding_idx (Optional[int], optional): The index of the padding value. Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Embeds a batch of binned expression values.

        Args:
            x (Tensor): A tensor of binned values of shape (batch, seq_len).

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, embsize).
        """
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ConditionEncoder(nn.Module):
    """Embeds integer condition IDs, allowing the model to be conditioned on categorical metadata, such as cell type or sequencing technology."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """Initializes the condition encoder.

        Args:
            num_embeddings (int): Number of unique conditions.
            embedding_dim (int): Dimension of the condition embeddings.
            padding_idx (Optional[int], optional): Padding index for the embeddings. Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the input conditions.

        Args:
            x (Tensor): Input tensor of shape (batch).

        Returns:
            Tensor: Encoded tensor of shape (batch, embedding_dim).
        """
        x = self.embedding(x)  # (batch, embedding_dim)
        x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    """Decodes contextual gene embeddings to predict gene expression values.

    Takes the output of the transformer encoder for a gene as input and passes it through a feed-forward network to predict its expression value.
    If configured, it can also predict the probability of the gene's expression being zero.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        explicit_zero_prob: bool = False,
        conditions: Optional[Dict] = None,
    ):
        """Initialises the gene expression value decoder.

        Args:
            d_model (int): Dimension of the input embeddings.
            out_dim (int): Dimension of the output gene expression values.
            explicit_zero_prob (bool, optional): Whether to predict the probability of zero expression. Defaults to False.
            conditions (Optional[Dict], optional): Configuration for additional conditions, used to adjust the input dimension. Defaults to None.
        """
        super().__init__()
        # d_in = d_model * (len(conditions) + 1) if conditions else d_model
        d_in = d_model  # If we feed condition token to transformer, rather than at the end.
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, out_dim),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass for the gene expression value decoder.

        Args:
            x (Tensor): Input tensor from the transformer encoder of shape (batch, seq_len, d_in).

        Returns:
            Dict[str, Tensor]: A dictionary containing the predicted expression values ('pred') and,
            if applicable, the zero expression probabilities ('zero_probs').
        """
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class MVCDecoder(nn.Module):
    """Decoder for the Masked Value Prediction for Cell embeddings (MVC) task."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        arch_style: str = "inner product",
        query_activation: Type[nn.Module] = nn.Sigmoid,
        hidden_activation: Type[nn.Module] = nn.PReLU,
        explicit_zero_prob: bool = False,
        conditions: Optional[Dict] = None,
    ) -> None:
        """Initialises the MVC decoder.

        Args:
            d_model (int): Dimension of the model embeddings.
            out_dim (int): Dimension of the output gene expression values.
            arch_style (str, optional): Architecture style of the decoder ('inner product', 'concat query', 'sum query'). Defaults to "inner product".
            query_activation (Type[nn.Module], optional): Activation function for the query vectors. Defaults to nn.Sigmoid.
            hidden_activation (Type[nn.Module], optional): Activation function for hidden layers. Defaults to nn.PReLU.
            explicit_zero_prob (bool, optional): Whether to predict the probability of zero expression. Defaults to False.
            conditions (Optional[Dict], optional): Configuration for additional conditions. Defaults to None.

        Raises:
            ValueError: If an unknown architecture style is provided.
        """
        super().__init__()
        # Inner products don't work with output dimension > 1

        d_in = d_model * (len(conditions) + 1) if conditions else d_model
        self.out_dim = out_dim
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
            if self.out_dim > 1:
                self.fc1 = nn.Linear(1, out_dim)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_in + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, out_dim)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, out_dim)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass for the MVC decoder.

        Args:
            cell_emb (Tensor): Cell embedding tensor of shape (batch, d_in).
            gene_embs (Tensor): Gene embedding tensor of shape (batch, seq_len, d_model).

        Raises:
            NotImplementedError: If explicit zero probability is enabled with 'concat query' or 'sum query' architecture.

        Returns:
            Dict[str, Tensor]: A dictionary of predicted gene expression values ('pred') and optional zero probabilities ('zero_probs').
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if self.out_dim > 1:
                pred_value = pred_value.unsqueeze(2)
                pred_value = self.fc1(pred_value)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return dict(pred=self.fc2(h).squeeze(2))  # (batch, seq_len)
        else:  # self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """A discriminator for Domain Adversarial Training (DAT).

    This network takes cell embeddings as input and tries to predict their domain (e.g., batch of origin). It is used with a gradient reversal layer to encourage the main model to learn domain-invariant representations.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """Initializes the AdversarialDiscriminator.

        Args:
            d_model (int): Dimension of the input embeddings (cell embeddings).
            n_cls (int): Number of domain classes to predict.
            nlayers (int, optional): Number of layers in the discriminator network. Defaults to 3.
            activation (Type[nn.Module], optional): Activation function for hidden layers. Defaults to nn.GELU.
        """
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the discriminator.

        Args:
            x (Tensor): Input tensor (cell embeddings) of shape [batch_size, d_model].

        Returns:
            Tensor: Output logits of shape [batch_size, n_cls].
        """
        x = grad_reverse(x, scale=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

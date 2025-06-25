from typing import Dict, Mapping, Optional, Tuple, Any, Union
import warnings

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange

from cancerfoundation.loss import LossType, criterion_neg_log_bernoulli, get_loss, masked_relative_error
from .grad_reverse import grad_reverse
import lightning as pl
from .layers import CFLayer, CFGenerator
from torch.nn.attention import SDPBackend, sdpa_kernel

def with_sdp_kernel(func):
    def wrapped_func(*args, **kwargs):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return func(*args, **kwargs)
    return wrapped_func


class TransformerModule(pl.LightningModule):
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
        dropout: float = 0.0,
        do_mvc: bool = False,
        conditions: Dict = None,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        explicit_zero_prob: bool = False,
        use_generative_training=False,
        pre_norm: bool = False,
        do_dat: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.conditions = conditions
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.pad_token_id = pad_token_id
        self.norm_scheme = "pre" if pre_norm else "post"

        self.n_input_bins = n_input_bins
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.encoder = GeneEncoder(
                ntoken, d_model, padding_idx=pad_token_id)
        
        self.flag_encoder = nn.Embedding(2, d_model)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        self.do_dat = do_dat
        self.criterion_conditions = nn.CrossEntropyLoss()
        self.criterion = criterion
        
        if conditions:
            self.condition_encoders = nn.ModuleDict({})
            for cond_name, cond_num in self.conditions.items():
                self.condition_encoders[cond_name] = ConditionEncoder(
                    cond_num, d_model)
            
            if do_dat:
                self.grad_reverse_discriminators = nn.ModuleDict({})
                for cond_name, cond_num in self.conditions.items():
                    self.grad_reverse_discriminators[cond_name] = AdversarialDiscriminator(
                        d_model,
                        n_cls=cond_num,
                    )

        if batchnorm:
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        # else:
        #     print("Using simple batchnorm instead of domain specific batchnorm")
        #     self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        # bug
        self.use_generative_training = use_generative_training

        if use_generative_training:
            encoder_layers = CFLayer(
                d_model,
                nhead,
                d_hid,
                dropout,
                batch_first=True,
                norm_scheme=self.norm_scheme,
            )
            self.transformer_encoder = CFGenerator(
                encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layers, nlayers)

        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            conditions=self.conditions,
            out_dim=out_dim
        )

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                conditions=self.conditions,
                out_dim=out_dim
            )
        self.MVC = do_mvc
        

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        domain_labels: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.use_generative_training:
            self._check_condition_labels(conditions)

            src = self.encoder(src)  # (batch, seq_len, embsize)

            self.cur_gene_token_embs = src

            values = self.value_encoder(values)  # (batch, seq_len, embsize)

            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
            
            if getattr(self, "dsbn", None) is not None:
                batch_label = int(domain_labels[0].item())
                total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                    0, 2, 1
                )  # the batch norm always works on dim 1
            elif getattr(self, "bn", None) is not None:
                total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
            

            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask
            )

        else:
            output_pcpt, _ = self.transformer_generate(
                pcpt_genes=src, pcpt_values=values,
                pcpt_key_padding_mask=src_key_padding_mask,
                gen_genes=None,
                gen_key_padding_mask=None,
                conditions=conditions,
            )
            output = output_pcpt
        return output  # (batch, seq_len, embsize)

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        conditions: Optional[Tensor] = None,  # (batch,)
        input_cell_emb: Optional[Tensor] = None,  # (batch, seq_len, embsize)
        domain_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        self._check_condition_labels(conditions)

        # (batch, pcpt_len, embsize)
        pcpt_token_embs = self.encoder(pcpt_genes)
        pcpt_values = self.value_encoder(
            pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values

        assert self.input_emb_style != "scaling"
        if gen_genes is not None:
            # (batch, gen_len, embsize)
            gen_token_embs = self.encoder(gen_genes)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs], dim=1
            )

            """print(gen_genes.shape)
            print(pcpt_genes.shape)
            print(pcpt_values.shape)"""

            gen_flags = self.flag_encoder(
                torch.tensor(1).to(pcpt_values.device)
            ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs,
            gen_total_embs,
            pcpt_key_padding_mask=pcpt_key_padding_mask,
            gen_key_padding_mask=gen_key_padding_mask,
        )

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError(
                    "weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_condition_labels(self, condition_labels: Optional[Tensor] = None) -> None:
        """if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )"""
        assert bool(self.conditions) == bool(condition_labels)

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)  # (batch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        if self.domain_spec_batchnorm:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        else:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(
                        1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        condition_emb: Optional[Tensor] = None,
        MVC: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.conditions
                else torch.cat([cell_emb, condition_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]

        if self.do_dat:
            if self.conditions:
                output["condition_output"] = {}
                for cond_name, discriminator in self.grad_reverse_discriminators.items():
                    output["condition_output"][cond_name] = discriminator(cell_emb)

        return output
    
    def _prepare_generative_input(self, tensors: dict[str, torch.Tensor]):
        pcpt_gene = tensors["pcpt_gene"]
        pcpt_expr = tensors["pcpt_expr"]
        pcpt_key_padding_mask = pcpt_gene.eq(
            self.pad_token_id)
        gen_gene = tensors["gen_gene"]
        gen_expr_target = target_values = tensors["gen_expr_target"]
        gen_key_padding_mask = gen_gene.eq(
            self.pad_token_id)
        
        return pcpt_gene, pcpt_expr, pcpt_key_padding_mask, gen_gene, gen_expr_target, gen_key_padding_mask
        
    def _prepare_perceptual_input(self, tensors: dict[str, torch.Tensor]):
        input_gene_ids = tensors["gene"]
        input_values = tensors["masked_expr"]
        target_values = tensors["expr"]
        src_key_padding_mask = input_gene_ids.eq(
            self.pad_token_id)
         
        return input_gene_ids, input_values, src_key_padding_mask, target_values
        
        

    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        use_cell_embedding: bool = False
    ) -> Mapping[str, Tensor]:
        """
        Wrapper to call either generative_forward or perceptual_forward, depending
        on the value of the "generative_training" kwarg.
        """
        
        loss_dict = {}
        conditions_batch = tensors["conditions"] if self.conditions else None
        if self.use_generative_training:
            
            pcpt_gene, pcpt_expr, pcpt_key_padding_mask, gen_gene, gen_expr_target, gen_key_padding_mask = self._prepare_generative_input(tensors)
            output_dict = self.generative_forward(pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    MVC=self.MVC,
                    conditions=conditions_batch
                    )
            
            gen_expr_preds = output_values = output_dict["gen_preds"]

            positions_to_match = ~gen_key_padding_mask

            loss = loss_expr = self.criterion(
                gen_expr_preds, gen_expr_target, positions_to_match
            )
            loss_dict["loss_expr"] = loss_expr       
            
            if self.MVC:
                loss_mvc = self.criterion(
                    output_dict["mvc_output"][:, pcpt_gene.shape[1]:], gen_expr_target, positions_to_match)
                loss = loss + loss_mvc
                loss_dict["loss_mvc"] = loss_mvc       

            if self.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], gen_expr_target, positions_to_match
                )
                loss = loss + loss_zero_log_prob
                loss_dict["loss_zero_log_prob"] = loss_zero_log_prob
                if self.MVC:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], gen_expr_target, positions_to_match
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    loss_dict["loss_gepc_zero_log_prob"] = loss_gepc_zero_log_prob

        else:
            input_gene_ids, input_values, src_key_padding_mask, target_values = self._prepare_perceptual_input(tensors)
            output_dict = self.perceptual_forward(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask, conditions=conditions_batch, MVC=MVC)
            
            output_values = output_dict["mlm_output"]

            positions_to_match = input_values.eq(
                self.mask_value
            )  # the postions to predict
            loss = loss_expr = self.criterion(
                output_values, target_values, positions_to_match
            )

            if self.MVC:
                loss_mvc = self.criterion(
                    output_dict["mvc_output"], target_values, positions_to_match
                )
                loss = loss + loss_mvc
            
            if self.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, positions_to_match
                )
                loss = loss + loss_zero_log_prob

                if self.MVC:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, positions_to_match
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    
        
        if self.do_dat:
            if self.conditions:
                for condition in self.conditions:
                    
                    
                    condition_loss = self.criterion_conditions(
                        output_dict["condition_output"][condition], conditions_batch[condition].squeeze())
                    
                    loss += condition_loss / len(self.conditions)
                    
                    loss_dict["condition_" + condition] = condition_loss.detach() / len(self.conditions)
                    
                
            
        if use_cell_embedding:
            previous_cell_embs = output_dict["cell_emb"].detach().clone()
            preds = self.generative_forward(
                pcpt_gene.clone(),
                pcpt_expr.clone(),
                pcpt_key_padding_mask.clone(),
                gen_gene.clone(),
                gen_key_padding_mask.clone(),
                MVC=False,
                input_cell_emb=previous_cell_embs,
                conditions=conditions_batch
            )["gen_preds"]

            loss_gen = self.criterion(
                preds, gen_expr_target, positions_to_match)
            loss = loss + loss_gen
            loss_dict["loss_gen"] = loss_gen
          
        loss_dict["total_loss"] = loss       
        return loss_dict
        
    @with_sdp_kernel
    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch, use_cell_embedding=False)
        return loss_dict["total_loss"]

    def generative_forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        MVC: bool = False,
        do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            pcpt_genes (:obj:`Tensor`): token ids of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_values (:obj:`Tensor`): token values of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_key_padding_mask (:obj:`Tensor`): mask for pcpt_genes, shape
                [batch_size, seq_len]
            gen_genes (:obj:`Tensor`): token ids of the generative part, shape
                [batch_size, seq_len]
            gen_key_padding_mask (:obj:`Tensor`): mask for gen_genes, shape
                [batch_size, seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            do_sample (:obj:`bool`): whether to do sampling from bernoulli for
                generated zero predictions.
            input_cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                embsize]

        Returns:
            :obj:`Mapping[str, Tensor]`:
                - pred (:obj:`Tensor`): prediction, shape [batch_size, seq_len]
                - cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                    embsize]
        """

        pcpt_output, gen_output = self.transformer_generate(
            pcpt_genes,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_genes,
            gen_key_padding_mask,
            conditions,
            input_cell_emb=input_cell_emb,
        )
        if gen_output is None:
            transformer_output = pcpt_output
        else:
            transformer_output = torch.cat([pcpt_output, gen_output], dim=1)

        if self.conditions:
            """print("condition encoders: ", self.condition_encoders)
            for cond_name, cond_values in conditions.items():
                print(cond_name, cond_values)"""
            condition_emb = torch.cat([self.condition_encoders[cond_name](
                cond_values) for cond_name, cond_values in conditions.items()], dim=1).view(transformer_output.shape[0], -1)

        output = {}
        decoder_output = self.decoder(
            transformer_output
            if not self.conditions
            else torch.cat(
                [
                    transformer_output,
                    condition_emb.unsqueeze(1).repeat(
                        1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=decoder_output["zero_probs"])
            full_preds = bernoulli.sample() * decoder_output["pred"]
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1]:]
        else:
            full_preds = decoder_output["pred"]  # (batch, seq_len)
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1]:]
        if self.explicit_zero_prob:
            output["zero_probs"] = decoder_output["zero_probs"]

        output = self._extend_output(
            output,
            transformer_output,
            condition_emb=condition_emb if self.conditions else None,
            MVC=MVC,
            do_sample=do_sample,
        )

        return output

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        MVC: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output

        Returns:
            dict of output Tensors.
        """
        transformer_output = self.encode(
            src, values, src_key_padding_mask, conditions
        )
        if self.conditions:
            condition_emb = torch.cat([self.condition_encoders[cond_name](
                cond_values) for cond_name, cond_values in conditions.items()], dim=1).view(transformer_output.shape[0], -1)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.conditions
            else torch.cat(
                [
                    transformer_output,
                    condition_emb.unsqueeze(1).repeat(
                        1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
        )
    
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        output = self._extend_output(
            output,
            transformer_output,
            condition_emb=condition_emb if self.conditions else None,
            MVC=MVC,
            do_sample=do_sample,
        )

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        conditions: Optional[Dict] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            if self.conditions:
                conditions_i = {}
                for cond_name, cond_values in conditions.items():
                    conditions_i[cond_name] = cond_values[i: i +
                                                          batch_size].to(device)
            raw_output = self.encode(
                src[i: i + batch_size].to(device),
                values[i: i + batch_size].to(device),
                src_key_padding_mask[i: i + batch_size].to(device),
                conditions_i
                if conditions
                else None,
            )

            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i: i + batch_size] = output

        return outputs

class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
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


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int,
        explicit_zero_prob: bool = False,
        conditions: Dict = None,
    ):
        super().__init__()
        d_in = d_model * (len(conditions)+1) if conditions else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, out_dim),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, out_dim),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)



class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        conditions: Dict = None,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        # Inner products don't work with output dimension > 1

        d_in = d_model * (len(conditions)+1) if conditions else d_model
        self.out_dim = out_dim
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
            if out_dim > 0:
                self.fc1 = nn.Linear(1, out_dim)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_in+64, 64)
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
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
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
            zero_logits = torch.bmm(self.W_zero_logit(
                query_vecs), cell_emb).squeeze(2)
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
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        x = grad_reverse(x, scale=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

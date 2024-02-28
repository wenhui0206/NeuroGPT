#!/usr/bin/env python3

from typing import Dict
import warnings
import torch
from transformers import GPT2Config, GPT2Model
import torch.nn as nn

class GPTModel(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        embed_dim: int = 768,
        intermediate_dim_factor: int = 4,
        n_positions: int = 512,
        hidden_activation: str = 'gelu',
        dropout: float = 0.1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'GPT'
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.intermediate_dim_factor = intermediate_dim_factor
        self.n_positions = n_positions
        self.hidden_activation = hidden_activation
        self.dropout_resid = dropout
        self.dropout_attn = dropout
        self.dropout_embd = dropout
        self.mse_loss = torch.nn.MSELoss()
        self.bxe_loss = torch.nn.BCEWithLogitsLoss() 
        self.config = GPT2Config(
            vocab_size=1,
            n_positions=self.n_positions,
            n_embd=self.embed_dim,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.embed_dim * self.intermediate_dim_factor,
            resid_pdrop=self.dropout_resid,
            attn_pdrop=self.dropout_attn,
            embd_pdrop=self.dropout_embd,
            activation_function=self.hidden_activation
        )
        self.transformer = GPT2Model(config=self.config)
        self.is_decoding_mode = False
        self.decoding_head = None
        self.num_decoding_classes = None
        self.pooler_layer = None
        self.add_pooler_layer()

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool=False,
        num_decoding_classes: int=None
        ) -> None:
        self.is_decoding_mode = is_decoding_mode
        if self.is_decoding_mode:
            if self.pooler_layer is None:
                self.add_pooler_layer()
            self.add_decoding_head(num_decoding_classes=num_decoding_classes)
        else:
            self.decoding_head = None

    def add_pooler_layer(self):
        if self.pooler_layer is not None:
            warnings.warn(
                    'Warning: overwriting existing pooler layer'
                )
        self.pooler_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(self.dropout_resid)
        )

    def add_decoding_head(
        self,
        num_decoding_classes: int
        ) -> None:
        if self.decoding_head is not None:
            if self.num_decoding_classes == num_decoding_classes:
                warnings.warn(
                    'Warning: not overwriting decoding head, as '
                    f'{num_decoding_classes}-class decoding head exists.'
                )
                return None
            else:
                warnings.warn(
                    f'Warning: overwriting existing {num_decoding_classes}-class decoding head.'
                )
        self.num_decoding_classes = num_decoding_classes
        # self.decoding_head = torch.nn.Sequential(
        #     torch.nn.Linear(
        #         in_features=self.embed_dim,
        #         out_features=self.num_decoding_classes
        #     )
        # )
        self.decoding_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_decoding_classes)
        )
        return None
    
    def decode(
        self,
        outputs: torch.tensor,
        attention_mask: torch.tensor,
        ) -> Dict[str, torch.tensor]:
        assert self.is_decoding_mode, 'GPTModel must be in decoding_mode.'
        assert self.pooler_layer is not None, 'pooler_layer head must be added.'
        assert self.decoding_head is not None, 'decoding head must be added.'
        batch_size = outputs.size()[0]
        sequence_lengths = attention_mask.sum(dim=1)-1
        decoding_outputs = {
            'pooler_outputs': self.pooler_layer(
                outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
            )
        }
        decoding_outputs['decoding_logits'] = self.decoding_head(decoding_outputs['pooler_outputs'])
        return decoding_outputs

    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        transformer_outputs = self.transformer.forward(
            inputs_embeds=batch['inputs_embeds'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
            return_dict=True
        )
        outputs = {'outputs': transformer_outputs['last_hidden_state']}

        if not self.is_decoding_mode:
            return outputs

        outputs.update(
            self.decode(
                outputs=outputs['outputs'],
                attention_mask=batch['attention_mask']
            )
        )
        return outputs


class PretrainedGPT2(GPTModel):
    
    def __init__(
        self,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.name = 'PretrainedGPT2'
        self.config = GPT2Config()
        self.n_positions = self.config.n_positions
        self.embed_dim = self.config.n_embd
        self.num_hidden_layers = self.config.n_layer
        self.num_attention_heads = self.config.n_head
        self.intermediate_dim_factor = 4
        self.dropout_resid = self.config.resid_pdrop
        self.dropout_attn = self.config.attn_pdrop
        self.dropout_embd = self.config.embd_pdrop
        self.hidden_activation = self.config.activation_function
        self.transformer = GPT2Model.from_pretrained("gpt2")
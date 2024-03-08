#/usr/bin/env python3

import pdb
import torch
from typing import Dict
from einops import rearrange

class EmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: int = 0.1,
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack = []
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.in_dim,
                        out_features=self.embed_dim
                    ),
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(p=self.dropout)
                ]
            )
        layer_stack.extend(
            [
                torch.nn.Linear(
                    in_features=self.embed_dim if self.num_hidden_layers>1 else self.in_dim,
                    out_features=self.embed_dim
                ),
                torch.nn.LayerNorm(self.embed_dim),
                torch.nn.Dropout(p=self.dropout)
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def _stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def _unstack_inputs(
        self,
        tensor,
        b
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='(b s) e -> b s e',
            b=b
        )

    def forward(
        self,
        inputs,
        **kwargs
        ) -> torch.tensor:
        inputs_stacked = self._stack_inputs(tensor=inputs)
        
        return self._unstack_inputs(
            tensor=self.model(inputs_stacked),
            b=inputs.size()[0]
        )


class BaseEmbedder(torch.nn.Module):
    def __init__(self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'BaseEmbedder'
        self.training_style = 'base'
        self._root_training_style = 'base'
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.xe_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bxe_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.l2_loss = torch.nn.MSELoss(reduction='mean') # for L2 loss
        # self.huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0) # for Huber loss
        
        self.embed_model = EmbeddingModel(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout
        )
        self.is_decoding_mode = False

    def switch_decoding_mode(self, is_decoding_mode: bool=False) -> None:
        self.is_decoding_mode = is_decoding_mode
        
        if self.is_decoding_mode:
            self.training_style = 'decoding'
        else:
            self.training_style = self._root_training_style
    
    @staticmethod
    def _pad_tensor_left_by_n(
        tensor,
        n,
        pad_value
        ) -> torch.tensor:
        filling = torch.ones(
            (
                tensor.size()[0],
                n,
                *tensor.size()[2:]
            ),
            device=tensor.device
        ) * pad_value
        
        return torch.cat(
            [
                filling,
                tensor
            ],
            dim=1
        ).to(torch.long)

    @staticmethod
    def _round_to_precision(
        x: torch.tensor,
        precision: float,
        ) -> torch.tensor:
        return torch.round(x / precision) * precision


    def embed_inputs(
        self,
        inputs: torch.tensor
        ) -> torch.tensor:
        return self.embed_model(inputs)
    
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        
        if self.in_dim == self.embed_dim:
            inputs_embeds = batch[inputs_key]
        else:
            inputs_embeds = self.embed_inputs(inputs=batch[inputs_key])
        
        return inputs_embeds

    def decoding_loss(
        self,
        decoding_logits,
        labels,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        # pdb.set_trace()
        return {
            'decoding_loss': self.xe_loss(
                input=decoding_logits,
                target=labels.to(dtype=torch.long)
            )
        }
    
    def reconstruction_loss(
        self,
        input,
        target,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return {
            'reconstruction_loss': self.l2_loss(
                input=input,
                target=target
            )
        }

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict:
        batch_out = {}
        
        for key in batch:
            
            if (
                torch.is_tensor(batch[key])
                and key != 'labels'
            ):
                batch_out[key] = batch[key].to(torch.float)
            
            elif key == 'labels':
                batch_out[key] = batch['labels'].to(torch.int)

            else:
                batch_out[key] = torch.clone(batch[key])
        
        # dummy copy of inputs to be used in forward pass
        batch_out['inputs_embeds'] = torch.clone(batch_out['inputs'])
        
        return batch_out

    def _root_loss(
        self,
        inputs,
        outputs,
        attention_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        attention_mask = torch.unsqueeze(attention_mask, -1).repeat(1,1,self.in_dim)
        
        return  self.reconstruction_loss(
            input=torch.masked_select(outputs, attention_mask.to(torch.bool)),
            target=torch.masked_select(inputs, attention_mask.to(torch.bool))
        )

    def loss(
        self,
        batch,
        outputs
        ) -> Dict[str, torch.tensor]:

        if self.is_decoding_mode:
            losses = self.decoding_loss(
                **batch,
                **outputs
            )
        
        else:
            losses = self._root_loss(
                **batch,
                **outputs
            )

        if 'loss' not in losses:
            losses['loss'] = sum(losses.values())

        return losses
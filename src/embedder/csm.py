
#/usr/bin/env python3

import pdb
from typing import Dict
import torch
from embedder.base import BaseEmbedder


class CSMEmbedder(BaseEmbedder):
    
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'CSMEmbedder'
        self.training_style = 'CSM'
        assert self.training_style in {'CSM', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'CSM'
        ##=========
        self.in_dim_for_mask = self.in_dim
        self.msk_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim_for_mask)
            )
        )
        self.cls_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim_for_mask)
            )
        )
        self._embeds = [
            self.msk_embed,
            self.cls_embed
        ]
        self._init_embeds()

    def _init_embeds(self):
        
        for embed in self._embeds:
            torch.nn.init.normal_(
                tensor=embed,
                mean=0.0,
                std=1.0,
            )

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor],
        ) -> Dict[str, torch.tensor]:
        batch_out = dict(batch)
        labels =  torch.clone(batch['labels']) if 'labels' in batch else None

        if self.training_style != 'decoding':
            return self.mask_inputs(batch=batch_out)

        batch_out =  self.add_cls_embed(batch=batch_out)
        
        if labels is not None:
            batch_out['labels'] = labels
        
        return batch_out
        
    def mask_inputs(
        self,
        batch: Dict[str, torch.tensor],
        ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        input_shape = batch[inputs_key].size()
        device = batch[inputs_key].device
        masking_i = torch.cat(
            [
                torch.randint(
                    low=1, # at least one seq value before mask!
                    high=sum(batch['attention_mask'][i]==1), # high is exclusive, so this accounts for 0-indexing
                    size=(1,),
                    device=device
                )
                for i in range(input_shape[0])
            ],
            dim=0
        )
        print("masking id", masking_i)
        modelling_mask = torch.zeros_like(
            batch[inputs_key],
            device=device
        )
        modelling_mask[torch.arange(input_shape[0]), masking_i] = 1
        batch['modelling_mask'] = modelling_mask.to(torch.long)
        batch['masked_inputs'] = torch.masked_select(
            input=batch[inputs_key],
            mask=batch['modelling_mask'].to(torch.bool)
        ).detach().clone()
        batch['inputs_embeds'] = torch.where(
            batch['modelling_mask']==1,
            self.msk_embed.repeat(
                input_shape[0],
                input_shape[1],
                1
            ),
            batch[inputs_key].to(torch.float)
        )
        batch['attention_mask'] = torch.cat(
            [
                torch.cat(
                    (
                        torch.ones(
                            (
                                1,
                                i+1 # to account for 0-indexing in python
                            ),
                            device=device
                        ),
                        torch.zeros(
                            (
                                1,
                                input_shape[1]-i-1 # to account for 0-indexing in python
                            ),
                            device=device
                        )
                    ),
                    dim = 1
                )
                for i in masking_i
            ],
            dim = 0
        ).to(torch.long)
        # re-mask inputs
        attention_mask_expanded = torch.unsqueeze(
            batch['attention_mask'],
            dim=2
        ).repeat(
            1,
            1,
            self.in_dim_for_mask
        )
        batch["inputs_embeds"] = torch.where(
            attention_mask_expanded == 1,
            batch['inputs_embeds'],
            torch.zeros_like(batch['inputs_embeds'])
        )

        return batch

    def add_cls_embed(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        batch_size = batch[inputs_key].size()[0]
        sequence_lengths = batch['attention_mask'].sum(dim=1)
        inputs_embeds = []
        
        if 't_rs' in batch:
            t_rs = []
        
        for i in range(len(sequence_lengths)):
            inputs_embeds.append(
                torch.cat(
                    [
                        batch[inputs_key][i, :sequence_lengths[i], :],
                        self.cls_embed[0],
                        batch[inputs_key][i, sequence_lengths[i]:, :]
                    ],
                    dim=0
                )
            )
            
            if 't_rs' in batch:
                t_rs.append(
                    torch.cat(
                        [
                            batch['t_rs'][i, :sequence_lengths[i]],
                            torch.ones(1, device=batch['t_rs'].device) * -1,
                            batch['t_rs'][i, sequence_lengths[i]:]
                        ],
                        dim=0
                    )
                )

        batch['inputs_embeds'] = torch.stack(
            inputs_embeds,
            dim=0
        )

        if 't_rs' in batch:
            batch['t_rs'] = torch.stack(
                t_rs,
                dim=0
            )

        if 'token_type_ids' in batch:
            batch['token_type_ids'] = self._pad_tensor_left_by_n(
                tensor=batch['token_type_ids'],
                n=1,
                pad_value=0
            )

        if 'modelling_mask' in batch:
            batch['modelling_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['modelling_mask'],
                n=1,
                pad_value=0
            )

        if 'attention_mask' in batch:
            batch['attention_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['attention_mask'],
                n=1,
                pad_value=1
            )
        return batch

    def masking_loss(
        self,
        masked_inputs,
        outputs,
        modelling_mask
        ) -> Dict[str, torch.tensor]:
        
        return {
            'masking_loss': self.reconstruction_loss(
                input=torch.masked_select(outputs, modelling_mask.to(torch.bool)),
                target=masked_inputs
            )['reconstruction_loss']
        }

    def _root_loss(
        self,
        masked_inputs,
        outputs,
        modelling_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return self.masking_loss(
            masked_inputs=masked_inputs,
            outputs=outputs,
            modelling_mask=modelling_mask
        )
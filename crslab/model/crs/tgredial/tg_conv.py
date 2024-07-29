# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2021/1/7, 2020/12/15, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou, Yuanhang Zhou
# @Email  : wxl1999@foxmail.com, sdzyh002@gmail, sdzyh002@gmail.com

r"""
TGReDial_Conv
=============
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources


class TGConvModel(BaseModel):
    """
        
    Attributes:
        context_truncate: A integer indicating the length of dialogue context.
        response_truncate: A integer indicating the length of dialogue response.
        pad_id: A integer indicating the id of padding token.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']

        language = dataset_language_map[opt['dataset']]
        resource = resources['gpt2'][language]
        dpath = os.path.join(PRETRAIN_PATH, 'gpt2', language)
        super(TGConvModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        """build model"""
        self.model = GPT2LMHeadModel.from_pretrained(self.dpath)
        self.loss = CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, batch, mode):
        if mode == 'test' or mode == 'infer':
            enhanced_context = batch[1]
            return self.generate(enhanced_context)
        else:
            enhanced_input_ids = batch[0]
            # torch.tensor's shape = (bs, seq_len, v_s); tuple's length = 12
            lm_logits = self.model(enhanced_input_ids).logits

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits[:, -self.response_truncate:-1, :],
                enhanced_input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits, dim=2)[1]  # [bs, seq_len]
            pred = pred[:, -self.response_truncate:]

            return loss, pred

    def generate(self, context):
        """
        Args:
            context: torch.tensor, shape=(bs, context_turncate)

        Returns:
            generated_response: torch.tensor, shape=(bs, reponse_turncate-1)
        """
        generated_response = []
        former_hidden_state = None
        context = context[..., -self.response_truncate + 1:]

        for _ in range(self.response_truncate - 1):
            outputs = self.model(context, former_hidden_state)  # (bs, c_t, v_s),
            last_hidden_state, former_hidden_state = outputs.logits, outputs.past_key_values

            next_token_logits = last_hidden_state[:, -1, :]  # (bs, v_s)
            preds = next_token_logits.argmax(dim=-1).long()  # (bs)

            context = preds.unsqueeze(1)
            generated_response.append(preds)

        generated_response = torch.stack(generated_response).T

        return generated_response

    def generate_bs(self, context, beam=4):
        context = self._truncate_context(context)
        context_former = context
        batch_size = context.shape[0]
        sequences = self._initialize_sequences(batch_size)

        for _ in range(self.response_truncate - 1):
            context = self._update_context(context, context_former, sequences, batch_size)
            
            next_token_probs = self._compute_next_token_probs(context)
            probs, preds = self._get_top_k_probs_and_preds(next_token_probs, beam, batch_size)
            
            sequences = self._update_sequences(sequences, probs, preds, batch_size, beam)

        return self._prepare_result(sequences, batch_size)

    def _truncate_context(self, context):
        return context[..., -self.response_truncate + 1:]

    def _initialize_sequences(self, batch_size):
        return [[[list(), 1.0]]] * batch_size

    def _update_context(self, context, context_former, sequences, batch_size):
        if sequences != self._initialize_sequences(batch_size):
            context = [
                torch.cat((context_former[i], torch.tensor(cand[0]).to(self.device)))
                for i in range(batch_size)
                for cand in sequences[i]
            ]
            context = torch.stack(context)
        return context

    def _compute_next_token_probs(self, context):
        with torch.no_grad():
            outputs = self.model(context)
        last_hidden_state = outputs.logits
        next_token_logits = last_hidden_state[:, -1, :]
        return torch.nn.functional.softmax(next_token_logits)

    def _get_top_k_probs_and_preds(self, next_token_probs, beam, batch_size):
        topk = torch.topk(next_token_probs, beam, dim=-1)
        probs = topk.values.reshape([batch_size, -1, beam])
        preds = topk.indices.reshape([batch_size, -1, beam])
        return probs, preds

    def _update_sequences(self, sequences, probs, preds, batch_size, beam):
        new_sequences = []
        for j in range(batch_size):
            candidates = self._generate_candidates(sequences[j], probs[j], preds[j], beam)
            ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)
            new_sequences.append(ordered[:beam])
        return new_sequences

    def _generate_candidates(self, sequence, probs, preds, beam):
        return [
            [seq + [preds[n][k]], prob * probs[n][k]]
            for n, (seq, prob) in enumerate(sequence)
            for k in range(beam)
        ]

    def _prepare_result(self, sequences, batch_size):
        return torch.stack([torch.stack(sequences[i][0][0]) for i in range(batch_size)])

    def calculate_loss(self, logit, labels):
        """
        Args:
            preds: torch.FloatTensor, shape=(bs, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(bs, response_truncate)

        """

        loss = self.loss(logit.reshape(-1, logit.size(-1)), labels.reshape(-1))
        return loss

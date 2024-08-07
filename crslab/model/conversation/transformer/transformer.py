# @Time   : 2020/12/17
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
Transformer
===========
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.transformer import TransformerEncoder, TransformerDecoder


class TransformerModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_word: A integer indicating the number of words.
        n_entity: A integer indicating the number of entities.
        pad_word_idx: A integer indicating the id of word padding.
        pad_entity_idx: A integer indicating the id of entity padding.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the drouput rate.
        attention_dropout: A integer indicating the drouput rate of attention layer.
        relu_dropout: A integer indicating the drouput rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = vocab['n_word']
        self.n_entity = vocab['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.longest_label = opt.get('longest_label', 1)
        # encoder
        self.transformer_config = {
            'n_heads': opt.get('n_heads', 2),
            'n_layers': opt.get('n_layers', 2),
            'embedding_size': self.token_emb_dim,
            'ffn_size': opt.get('ffn_size', 300),
            'vocabulary_size': self.vocab_size,
            'embedding': self.token_embedding,
            'dropout': opt.get('dropout', 0.1),
            'attention_dropout': opt.get('attention_dropout', 0.0),
            'relu_dropout': opt.get('relu_dropout', 0.1),
            'padding_idx': self.pad_token_idx,
            'learn_positional_embeddings': opt.get('learn_positional_embeddings', False),
            'embeddings_scale': opt.get('embedding_scale', True),
            'reduction': opt.get('reduction', False),
            'n_positions': opt.get('n_positions', 1024)
        }

        super(TransformerModel, self).__init__(opt, device)

    def build_model(self):
        self._init_embeddings()
        self._build_conversation_layer()

    def _init_embeddings(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        logger.debug('[Finish init embeddings]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.conv_encoder = TransformerEncoder(
            self.transformer_config
        )

        self.conv_decoder = TransformerDecoder(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )

        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        logger.debug('[Finish build conv layer]')

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, response):
        batch_size, _ = response.shape # batch_size, seq_len
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        dialog_latent, _ = self.conv_decoder(inputs, token_encoding)  # (bs, seq_len, dim)

        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        preds = gen_logits.argmax(dim=-1)
        return gen_logits, preds

    def _decode_greedy_with_kg(self, token_encoding):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        for _ in range(self.longest_label):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)

            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            preds = gen_logits.argmax(dim=-1).long()
            logits.append(gen_logits)
            inputs = torch.cat((inputs, preds), dim=1)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break
        logits = torch.cat(logits, dim=1)
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, beam=4):
        batch_size = token_encoding[0].shape[0]
        xs, sequences, incr_state = self._initialize_beam_search(batch_size)

        for i in range(self.longest_label):
            if i == 1:
                token_encoding = self._repeat_token_encoding(token_encoding, beam)
            
            if i != 0:
                xs = self._get_xs_from_sequences(sequences, batch_size, beam)

            logits, probs, preds = self._compute_logits_and_probs(xs, token_encoding, incr_state, sequences, batch_size, beam)
            
            sequences = self._update_sequences(sequences, xs, logits, probs, preds, batch_size, beam)

            if self._is_generation_finished(xs, batch_size):
                break

        return self._get_final_outputs(sequences)

    def _initialize_beam_search(self, batch_size):
        xs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        sequences = [[[list(), list(), 1.0]]] * batch_size
        incr_state = None
        return xs, sequences, incr_state

    def _repeat_token_encoding(self, token_encoding, beam):
        return (token_encoding[0].repeat(beam, 1, 1), token_encoding[1].repeat(beam, 1, 1))

    def _get_xs_from_sequences(self, sequences, batch_size, beam):
        xs = [sequences[j][d][0] for d in range(len(sequences[0])) for j in range(batch_size)]
        return torch.stack(xs).reshape(beam, batch_size, -1)

    def _compute_logits_and_probs(self, xs, token_encoding, incr_state, sequences, batch_size, beam):
        dialog_latent, incr_state = self.conv_decoder(xs.reshape(len(sequences[0]) * batch_size, -1),
                                                    token_encoding,
                                                    incr_state)
        dialog_latent = dialog_latent[:, -1:, :]
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
        logits = gen_logits.reshape(len(sequences[0]), batch_size, 1, -1)
        probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)
        return logits, probs, preds

    def _update_sequences(self, sequences, xs, logits, probs, preds, batch_size, beam):
        new_sequences = []
        for j in range(batch_size):
            all_candidates = self._generate_candidates(sequences[j], xs, logits, probs, preds, j, beam)
            ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
            new_sequences.append(ordered[:beam])
        return new_sequences

    def _generate_candidates(self, sequence, xs, logits, probs, preds, j, beam):
        candidates = []
        for n in range(len(sequence)):
            for k in range(beam):
                prob = sequence[n][2]
                logit = sequence[n][1]
                logit_tmp = logits[n][j][0].unsqueeze(0) if logit == [] else torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                seq_tmp = torch.cat((xs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                candidates.append(candidate)
        return candidates

    def _is_generation_finished(self, xs, batch_size):
        return ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == batch_size

    def _get_final_outputs(self, sequences):
        logits = torch.stack([seq[0][1] for seq in sequences])
        xs = torch.stack([seq[0][0] for seq in sequences])
        return logits, xs

    def forward(self, batch, mode):
        context_tokens, _, _, response = batch # context_tokens, context_entities, context_words, response

        # encoder-decoder
        tokens_encoding = self.conv_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self._decode_forced_with_kg(tokens_encoding,
                                                        response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else:
            logits, preds = self._decode_greedy_with_kg(tokens_encoding)
            return preds

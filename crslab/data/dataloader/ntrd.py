# @Time   : 2021/10/06
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, merge_utt_replace, padded_tensor, get_onehot, truncate, merge_utt


class NTRDDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        self._initialize_attributes(opt, vocab)

    def _initialize_attributes(self, opt, vocab):
        entity_attrs = ['n_entity', 'pad_entity_idx']
        token_attrs = ['pad_token_idx', 'start_token_idx', 'end_token_idx', 'pad_word_idx']
        truncate_attrs = ['context_truncate', 'response_truncate', 'entity_truncate', 'word_truncate']
        
        for attr in entity_attrs + token_attrs:
            setattr(self, attr, vocab[attr.replace('_idx', '')])
        
        for attr in truncate_attrs:
            setattr(self, attr, opt.get(attr, None))
        
        self.replace_token = opt.get('replace_token', None)
        self.replace_token_idx = vocab[self.replace_token]

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch):
        context_data = self._extract_context_data(batch)
        return (
            padded_tensor(context_data['words'], self.pad_word_idx, pad_tail=False),
            get_onehot(context_data['entities'], self.n_entity)
        )

    def _extract_context_data(self, batch):
        return {
            'entities': [self._truncate_context(conv['context_entities'], self.entity_truncate) for conv in batch],
            'words': [self._truncate_context(conv['context_words'], self.word_truncate) for conv in batch]
        }

    def _truncate_context(self, context, truncate_size):
        return truncate(context, truncate_size, truncate_tail=False)

    def rec_process_fn(self):
        return [
            self._create_augmented_dict(conv_dict, movie)
            for conv_dict in tqdm(self.dataset)
            if conv_dict['role'] == 'Recommender'
            for movie in conv_dict['items']
        ]

    def _create_augmented_dict(self, conv_dict, movie):
        augmented = deepcopy(conv_dict)
        augmented['item'] = movie
        return augmented

    def rec_batchify(self, batch):
        context_data = self._extract_context_data(batch)
        items = [conv['item'] for conv in batch]
        
        return (
            padded_tensor(context_data['entities'], self.pad_entity_idx, pad_tail=False),
            padded_tensor(context_data['words'], self.pad_word_idx, pad_tail=False),
            get_onehot(context_data['entities'], self.n_entity),
            torch.tensor(items, dtype=torch.long)
        )

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        context_data = self._extract_full_context_data(batch)
        responses = self._process_responses(batch)
        
        if not any(resp.count(self.replace_token_idx) for resp in responses):
            return False

        movies = [
            truncate(conv['items'], resp.count(self.replace_token_idx), truncate_tail=False)
            for conv, resp in zip(batch, responses)
        ]

        return (
            padded_tensor(context_data['tokens'], self.pad_token_idx, pad_tail=False),
            padded_tensor(context_data['entities'], self.pad_entity_idx, pad_tail=False),
            padded_tensor(context_data['words'], self.pad_word_idx, pad_tail=False),
            padded_tensor(responses, self.pad_token_idx),
            padded_tensor(movies, self.pad_entity_idx, pad_tail=False)
        )

    def _extract_full_context_data(self, batch):
        return {
            'tokens': [truncate(merge_utt(conv['context_tokens']), self.context_truncate, truncate_tail=False) for conv in batch],
            'entities': [self._truncate_context(conv['context_entities'], self.entity_truncate) for conv in batch],
            'words': [self._truncate_context(conv['context_words'], self.word_truncate) for conv in batch]
        }

    def _process_responses(self, batch):
        return [
            add_start_end_token_idx(
                truncate(conv['response'], self.response_truncate - 2),
                start_token_idx=self.start_token_idx,
                end_token_idx=self.end_token_idx
            )
            for conv in batch
        ]

    def policy_batchify(self, *args, **kwargs):
        pass
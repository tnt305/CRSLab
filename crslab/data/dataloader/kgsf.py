# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt


class KGSFDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        self._setup_attributes(opt, vocab)

    def _setup_attributes(self, opt, vocab):
        for key in ['n_entity', 'pad', 'start', 'end', 'pad_entity', 'pad_word']:
            setattr(self, f"{key}_{'idx' if key != 'n_entity' else ''}", vocab[key])
        
        for attr in ['context', 'response', 'entity', 'word']:
            setattr(self, f"{attr}_truncate", opt.get(f'{attr}_truncate', None))

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch):
        entities, words = self._extract_context_data(batch)
        return (
            padded_tensor(words, self.pad_word_idx, pad_tail=False),
            get_onehot(entities, self.n_entity)
        )

    def _extract_context_data(self, batch):
        entities = [self._truncate_data(conv['context_entities'], self.entity_truncate) for conv in batch]
        words = [self._truncate_data(conv['context_words'], self.word_truncate) for conv in batch]
        return entities, words

    def _truncate_data(self, data, max_len):
        return truncate(data, max_len, truncate_tail=False)

    def rec_process_fn(self):
        return [
            self._create_augmented_dict(conv, movie)
            for conv in tqdm(self.dataset)
            if conv['role'] == 'Recommender'
            for movie in conv['items']
        ]

    def _create_augmented_dict(self, conv, movie):
        augmented = deepcopy(conv)
        augmented['item'] = movie
        return augmented

    def rec_batchify(self, batch):
        entities, words = self._extract_context_data(batch)
        items = [conv['item'] for conv in batch]
        
        return (
            padded_tensor(entities, self.pad_entity_idx, pad_tail=False),
            padded_tensor(words, self.pad_word_idx, pad_tail=False),
            get_onehot(entities, self.n_entity),
            torch.tensor(items, dtype=torch.long)
        )

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        context_data = self._prepare_conversation_data(batch)
        
        return (
            padded_tensor(context_data['tokens'], self.pad_token_idx, pad_tail=False),
            padded_tensor(context_data['entities'], self.pad_entity_idx, pad_tail=False),
            padded_tensor(context_data['words'], self.pad_word_idx, pad_tail=False),
            padded_tensor(context_data['responses'], self.pad_token_idx)
        )

    def _prepare_conversation_data(self, batch):
        return {
            'tokens': [self._truncate_data(merge_utt(conv['context_tokens']), self.context_truncate) for conv in batch],
            'entities': [self._truncate_data(conv['context_entities'], self.entity_truncate) for conv in batch],
            'words': [self._truncate_data(conv['context_words'], self.word_truncate) for conv in batch],
            'responses': [self._prepare_response(conv['response']) for conv in batch]
        }

    def _prepare_response(self, response):
        truncated = truncate(response, self.response_truncate - 2)
        return add_start_end_token_idx(truncated, start_token_idx=self.start_token_idx, end_token_idx=self.end_token_idx)

    def policy_batchify(self, *args, **kwargs):
        # Optional policy function.
        pass

from torch.utils.data import Dataset
import numpy as np


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class ABSA_Dataset_(Dataset):
    def __init__(self, text: str, named_entity: str, tokenizer):
        self.data = []
        text_left, _, text_right = [s.lower().strip() for s in text.partition("$T$")]  # két string lista nélkül
        aspect = named_entity.lower().strip()
        polarity = '0'  # dummy value, NOT used

        text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

        text_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'polarity': polarity,
        }

        self.data.append(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

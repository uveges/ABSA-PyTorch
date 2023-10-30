import hu_core_news_lg
import pandas as pd
from transformers import AutoModel
import torch
from argparse import Namespace

import config
from models.bert_spc import BERT_SPC
from pathlib import Path
from data_utils import Tokenizer4Bert
import numpy as np
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def main():
    # test_text = 'DR. $T$ ANDRÁS (LMP): Köszönöm a szót, elnök úr.'
    # test_ne = 'schiffer'
    #
    # test_text_list = ['DR. $T$ ANDRÁS (LMP): Köszönöm a szót, elnök úr.'] * 10
    # test_ne_list = ['schiffer'] * 10

    df = pd.read_excel("../datasets/parl_speech_7_segmented_part_0.xlsx")
    texts = df[config.text_column].values.tolist()[:11]


    predictor = Predictor(state_dict='../state_dict/bert_spc_validated_val_acc_0.7159')
    for text, ne in zip(test_text_list, test_ne_list):
        class_ = predictor.predict(text=text, named_entity=ne)
        print(class_)


class Predictor(object):
    def __init__(self, state_dict: str, verbose: bool = False):
        self.verbose = verbose
        self.state_dict = state_dict
        self.model = self.__load_model()
        self.tokenizer = Tokenizer4Bert(max_seq_len=config.model_parameters['max_seq_len'],
                                        pretrained_bert_name=config.model_parameters['bert_model_name'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def __load_model(self):
        if self.verbose:
            print('Loading BERT model...')

        bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = {'dropout': config.model_parameters['dropout'],
             "bert_dim": config.model_parameters['bert_dim'],
             "polarities_dim": config.model_parameters['polarities_dim']}
        opt = Namespace(**x)
        model = BERT_SPC(bert=bert, opt=opt).to(device=device)

        if self.verbose:
            print('Done!')

        checkpoint = torch.load(Path(self.state_dict))

        if self.verbose:
            print('Loading state-dict to model...')

        model.load_state_dict(checkpoint)

        if self.verbose:
            print('Done!')

        return model

    def predict(self, text: str, named_entity: str) -> List[int]:
        """
        Given a Text - Named Entity pair, returns the predicted label.

        :param text: Text to predict, where the Named Entity's lemma replaced by a '$T$' character sequence.
        :param named_entity: Lemma of the Named Entity.
        """

        prepeared_data_for_prediction = ABSA_Dataset_(text, named_entity, self.tokenizer)
        test_data_loader = DataLoader(dataset=prepeared_data_for_prediction, batch_size=1, shuffle=False)

        predictions = []
        with torch.no_grad():
            for i_batch, t_batch in enumerate(test_data_loader):
                t_inputs = [t_batch[col].to(self.device) for col in ['concat_bert_indices', 'concat_segments_indices']]
                t_outputs = self.model(t_inputs)
                predicted_classes = torch.argmax(t_outputs, -1).tolist()
                predictions.extend(predicted_classes)
        return predictions


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
        concat_bert_indices = tokenizer.text_to_sequence(
            '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

        text_bert_indices = tokenizer.text_to_sequence(
            "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
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

if __name__ == '__main__':
    main()
    
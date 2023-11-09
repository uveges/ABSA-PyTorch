import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModel

import config
from data_utils import Tokenizer4Bert
from models.bert_spc import BERT_SPC
from tqdm import tqdm


def main():

    for part in range(14):
        file = f"../datasets/parl_speech_7_segmented_part_{part}.xlsx"
        df = pd.read_excel(file)
        preparator = DataPreparator(dataframe=df)
        data_dict = preparator.start()

        predictor = Predictor(state_dict=config.checkpoint)
        predictions = []
        print("Generating predictions...")
        for sent, aspect in tqdm(zip(data_dict[config.text_column], data_dict[config.NE_column])):
            prediction = predictor.predict(text=sent, named_entity=aspect)
            predictions.extend(prediction)
        data_dict[config.predictions_column] = predictions
        result_frame = pd.DataFrame.from_dict(data_dict)
        result_frame.to_excel(f'../resources/RESULTS_napirend_elotti_2006_2010_segmented_part_{part}.xlsx')


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

        bert = AutoModel.from_pretrained(config.model_parameters['bert_model_name'])
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


class DataPreparator(object):

    def __init__(self, dataframe: pd.DataFrame, huspacy_model_name: str = "hu_core_news_lg"):
        self.dataframe = dataframe
        self.original_data_list_per_column = {}     # {column_name: [original values]}
        self.column_names = []
        self.result_data_list_per_column = {config.NE_column: []}       # {column_name: [original values]} --> ready for prediction
        self.model_name = huspacy_model_name
        self.nlp = None
        self.PATHS = {
            "hu_core_news_lg": "pip install https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.whl",
            "hu_core_news_trf": "pip install https://huggingface.co/huspacy/hu_core_news_trf/resolve/v3.5.2/hu_core_news_trf-any-py3-none-any.whl"
        }
        try:
            if self.model_name == "hu_core_news_lg":
                import hu_core_news_lg
                self.nlp = hu_core_news_lg.load()
            if self.model_name == "hu_core_news_trf":
                import hu_core_news_trf
                self.nlp = hu_core_news_trf.load()
        except (OSError, IOError) as e:
            print(f"Error! Language model not installed. You can install it by 'pip install {self.PATHS[self.model_name]}'")
            sys.exit(e)

    def start(self) -> Dict:
        self.column_names = self.dataframe.columns.values.tolist()
        for c in self.column_names:
            if c not in self.original_data_list_per_column:
                self.original_data_list_per_column[c] = []
                self.result_data_list_per_column[c] = []
            self.original_data_list_per_column[c] = self.dataframe[c].values.tolist()
        print("Preprocess data...")
        for i, t in tqdm(enumerate(self.original_data_list_per_column[config.text_column])):
            sents, aspects = self.__preprocess_with_spacy(t)
            repetitions = len(sents)
            for column in self.column_names:
                if column == config.text_column:
                    for rep in range(repetitions):
                        self.result_data_list_per_column[column].append(sents[rep])
                else:
                    for rep in range(repetitions):
                        self.result_data_list_per_column[column].append(self.original_data_list_per_column[column][i])
            for rep in range(repetitions):
                self.result_data_list_per_column[config.NE_column].append(aspects[rep])

        return self.result_data_list_per_column

    def __preprocess_with_spacy(self, text: str) -> Tuple[List[str], List[str]]:

        preprocessed_sentences, named_entities = ([] for i in range(2))
        doc = self.nlp(text)
        for ent in doc.ents:
            lemma = self.nlp(ent.text)[0].lemma_
            start_index = ent.start_char
            end_index = start_index + len(lemma)
            preprocessed_sentences.append(text[:start_index] + "$T$" + text[end_index:])
            named_entities.append(lemma)

        return preprocessed_sentences, named_entities


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
    
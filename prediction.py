import sys
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel
from sklearn.metrics import classification_report

from data_utils import Tokenizer4Bert
from models.bert_spc import BERT_SPC

PATHS = {
    "hu_core_news_lg": "pip install https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.whl",
    "hu_core_news_trf": "pip install https://huggingface.co/huspacy/hu_core_news_trf/resolve/v3.5.2/hu_core_news_trf-any-py3-none-any.whl"
}


class DataPreparatorForPrediction(object):

    def __init__(self,
                 data_for_prediction: str,
                 text_column_name: str = "text",
                 huspacy_model_name: str = "hu_core_news_lg",
                 max_seq_len: int = 85,
                 bert_model_name: str = "SZTAKI-HLT/hubert-base-cc"):

        self.model_name = huspacy_model_name
        self.data_for_prediction = data_for_prediction
        self.text_column_name = text_column_name
        self.prediction_data_as_list: List[str] = []
        self.tokenizer = Tokenizer4Bert(max_seq_len=max_seq_len, pretrained_bert_name=bert_model_name)
        self.Dataset_: ABSA_Dataset_ = None
        self.model = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        self.sentences = []
        self.aspects = []
        try:
            if self.model_name == "hu_core_news_lg":
            # self.nlp = spacy.load(self.model_name)
                import hu_core_news_lg
                self.nlp = hu_core_news_lg.load()
            if self.model_name == "hu_core_news_trf":
                import hu_core_news_trf
                self.nlp = hu_core_news_trf.load()
        except (OSError, IOError) as e:
            print(f"Error! Language model not installed. You can install it by 'pip install {PATHS[self.model_name]}'")
            sys.exit(e)

    def start(self):
        if self.data_for_prediction.endswith('.xlsx'):
            dataframe = pd.read_excel(self.data_for_prediction)
            self.sentences = dataframe[self.text_column_name].values.tolist()
            self.prediction_data_as_list = self.create_train_format(self.sentences)
            self.sentences = self.prediction_data_as_list[0::3]
            self.aspects = self.prediction_data_as_list[1::3]
            self.aspects = [a.strip() for a in self.aspects]
        elif self.data_for_prediction.endswith('.txt'):
            with open(self.data_for_prediction, 'r', encoding='utf8') as i_:
                lines = i_.readlines()
            self.sentences = lines[::3]
            self.sentences = [s.strip() for s in self.sentences]
            self.prediction_data_as_list = lines
            self.aspects = lines[1::3]
            self.aspects = [a.strip() for a in self.aspects]
        self.Dataset_ = ABSA_Dataset_(self.prediction_data_as_list, self.tokenizer)
        return self.Dataset_

    def get_dataset_as_list(self):
        return self.sentences

    def get_aspects_as_list(self):
        return self.aspects

    def create_train_format(self, sentences: List[str]) -> List[str]:
        results_ = []
        print("Prepearing data for prediction (huspaCy)...")
        for s in tqdm(sentences):
            doc = self.nlp(s)
            for ent in doc.ents:
                lemma = self.nlp(ent.text)[0].lemma_
                start_index = ent.start_char
                end_index = start_index + len(lemma)
                results_.append(s[:ent.start_char] + "$T$" + s[end_index:])
                results_.append(lemma)
                results_.append('0')
        print("Done!")
        return results_


class ABSA_Dataset_(Dataset):
    def __init__(self, lines, tokenizer):
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

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
            polarity = int(polarity) + 1

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

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Predictor(object):

    def __init__(self, dataset: ABSA_Dataset_, state_dict: str):
        self.state_dict = state_dict
        self.dataset = dataset
        self.model = self.load_model()
        # self.model.load_state_dict(torch.load('../state_dict/bert_spc_validated_val_acc_0.7159'))


    def load_model(self):
        print('Loading BERT model...')
        bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = {'dropout': 0.01, "bert_dim": 768, "polarities_dim": 3}
        opt = Namespace(**x)
        model = BERT_SPC(bert=bert, opt=opt).to(device=device)
        print('Done!')
        checkpoint = torch.load(Path(self.state_dict))
        print('Loading state-dict to model...')
        model.load_state_dict(checkpoint)
        print('Done!')
        # self.model = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-05, weight_decay=0.01)
        # checkpoint = torch.load('../state_dict/bert_spc_validated_val_acc_0.7159')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # self.model.load_state_dict(torch.load('../state_dict/bert_spc_validated_val_acc_0.7159'))
        return model

    def evaluate(self):
        predictions = []
        self.model.eval()
        test_data_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Getting predictions...')
        with torch.no_grad():
            for i_batch, t_batch in enumerate(tqdm(test_data_loader)):
                t_inputs = [t_batch[col].to(device) for col in ['concat_bert_indices', 'concat_segments_indices']]
                t_outputs = self.model(t_inputs)
                predicted_classes = torch.argmax(t_outputs, -1).tolist()
                predictions.extend(predicted_classes)
        print('Done!')
        return predictions

    def start(self):
        return self.evaluate()


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

    # kiszerdónk egyesével minden sort a bemenetben
    # a sorból megkeressük a text -oszlopot
    # megkeressük a névelemeket - annyi példányban lemásoljuk az eredeti sort, ahány NE volt benne
    # ezeket a sorokat egyesével előkészítjük megfelelő formába
    # prediktálunk rá soronként
    # az eredményt kimentjük

    for part in range(14):
        # part = 7
        file = f"../datasets/parl_speech_7_segmented_part_{part}.xlsx"
        text_column = "text"

        p = DataPreparatorForPrediction(data_for_prediction=file,
                                        text_column_name=text_column,
                                        # huspacy_model_name="hu_core_news_trf"
                                        )
        prepared_dataset = p.start()

        predictor = Predictor(prepared_dataset, state_dict='../state_dict/bert_spc_validated_val_acc_0.7159')
        predictions = predictor.start()
        predictions = [p-1 for p in predictions]            # szentimentre, a címke számozás miatt

        data_as_list = p.get_dataset_as_list()
        aspects = p.get_aspects_as_list()

        # with open(file, 'r', encoding='utf8') as tmp:
        #     lines_ = tmp.readlines()
        #     GS_labels = lines_[2::3]
        #     GS_labels = [int(g) for g in GS_labels]

        # results = pd.DataFrame(list(zip(data_as_list, aspects, predictions, GS_labels)), columns=['text', 'aspect', 'prediction', 'Gold Standard'])
        # print(classification_report(y_true=GS_labels, y_pred=predictions))

        results = pd.DataFrame(list(zip(data_as_list, aspects, predictions)), columns=['Sentence', 'Aspect', 'Label'])
        results.to_excel(f'../resources/RESULTS_napirend_elotti_2006_2010_SEGMENTED_part_{part}.xlsx')

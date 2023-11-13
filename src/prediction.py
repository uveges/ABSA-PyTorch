from argparse import Namespace
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

import config
from data_utils import Tokenizer4Bert
from models.bert_spc import BERT_SPC
from preprocessors.dataset import ABSA_Dataset_


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

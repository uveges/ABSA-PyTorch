import logging
import math
import os
import random
import sys
from argparse import Namespace
from time import strftime, localtime

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, AutoModel

import config
from data_utils import Tokenizer4Bert, ABSADataset
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Trainer(object):
    """
    Performs fine-tuning for the selected BERT model.
    Keep in mind, that hyperparameters are scraped from `config.py` file.
    """

    def __init__(self):
        self.MODEL_OPTIONS = config.model_parameters
        self.start_train_setting()
        self.opt = Namespace(**self.MODEL_OPTIONS)

    def train(self, train_dataset_path: str, test_dataset_path: str, bert_model_name: str = config.model_parameters['bert_model_name']):

        ins = Instructor(self.opt, bert_model_name, train_dataset_path, test_dataset_path)
        ins.run()

    def start_train_setting(self):
        seed = config.model_parameters['seed']
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(seed)

        model_classes = {
            'bert_spc': BERT_SPC
        }

        input_colses = {
            'bert_spc': ['concat_bert_indices', 'concat_segments_indices']
        }
        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal_,
            'orthogonal_': torch.nn.init.orthogonal_,
        }
        optimizers = {
            'adadelta': torch.optim.Adadelta,   # default lr=1.0
            'adagrad': torch.optim.Adagrad,     # default lr=0.01
            'adam': torch.optim.Adam,           # default lr=0.001
            'adamax': torch.optim.Adamax,       # default lr=0.002
            'asgd': torch.optim.ASGD,           # default lr=0.01
            'rmsprop': torch.optim.RMSprop,     # default lr=0.01
            'sgd': torch.optim.SGD,
        }

        self.MODEL_OPTIONS['model_class'] = model_classes[config.model_parameters['model_name']]
        self.MODEL_OPTIONS['inputs_cols'] = input_colses[config.model_parameters['model_name']]
        self.MODEL_OPTIONS['initializer'] = initializers[config.model_parameters['initializer']]
        self.MODEL_OPTIONS['optimizer'] = optimizers[config.model_parameters['optimizer']]
        self.MODEL_OPTIONS['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if \
        config.model_parameters['device'] is None else torch.device(config.model_parameters['device'])

        log_file = '{}-{}.log'.format(config.model_parameters['model_name'], strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))


class Instructor:
    def __init__(self, opt, bert_model_name: str, train_dataset_path, test_dataset_path):
        self.opt = opt

        tokenizer = Tokenizer4Bert(opt.max_seq_len, bert_model_name)
        bert = AutoModel.from_pretrained(bert_model_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.trainset = ABSADataset(train_dataset_path, tokenizer)
        self.testset = ABSADataset(test_dataset_path, tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()

            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        classification_report = metrics.classification_report(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2])
        print(classification_report)
        return acc, f1

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        print(f"Best: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
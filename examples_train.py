import config

from src.training import Trainer


def start_train_setting():

    trainer = Trainer()
    trainer.train(train_dataset_path=config.train_dataset,
                  test_dataset_path=config.test_dataset,
                  bert_model_name=config.model_parameters['bert_model_name'])


if __name__ == '__main__':
    start_train_setting()

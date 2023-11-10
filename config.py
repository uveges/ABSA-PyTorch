# Train - Test set creation parameters

dateset_name = "Validated"
test_size = 0.2

text_column = "text"
NE_column = "Named Entity"
predictions_column = "Prediction"

checkpoint = '../state_dict/HUN_bert_spc_validated_val_acc_0.7159'

train_dataset = '../datasets/english_Train.txt'
test_dataset = '../datasets/english_Test.txt'

model_parameters = {
    'dropout': 0.01,
    "bert_dim": 768,
    "polarities_dim": 3,
    'max_seq_len': 85,
    # 'bert_model_name': "SZTAKI-HLT/hubert-base-cc"  # HUN
    'bert_model_name': "bert-base-cased",           # ENG
    'model_name': 'bert_spc',
    'dataset': 'validated',
    'optimizer': 'adam',                            # FROM: ['adadelta', 'adagrad', 'adam', 'adamax', 'asgd', 'rmsprop', 'sgd']
    'initializer': 'xavier_uniform_',               # FROM: ['xavier_uniform_', 'xavier_normal_', 'orthogonal_']
    'lr': 2e-5,
    'l2reg': 0.01,
    'num_epoch': 20,
    'batch_size': 16,
    'log_step': 10,
    'embed_dim': 300,
    'hidden_dim': 300,
    'pretrained_bert_name': 'bert-base-uncased',
    'hops': 3,
    'patience': 5,
    'device': None,
    'seed': 1234,
    'valset_ratio': 0
}

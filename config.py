# Train - Test set creation parameters

dateset_name = "Validated"
test_size = 0.2

text_column = "text"                                # must be already in the data
NE_column = "Named Entity"                          # to be generated by the code
NE_type_column = "NE type"
predictions_column = "Prediction"

#English case: 
checkpoint = 'state_dict/bert_spc_validated_val_acc_0.4659'        # ENG!
train_dataset = 'datasets/train_english.txt'
test_dataset = 'datasets/test_english.txt'
bert_model = "bert-base-cased"                                     # FROM: ["bert-base-cased", "SZTAKI-HLT/hubert-base-cc"]
spacy_model_name = "en_core_web_lg"

# Hungarian case:
# checkpoint = 'state_dict/bert_spc_validated_val_acc_0.6932'      # HUN!
# train_dataset = 'datasets/Validated_Train.txt'
# test_dataset = 'datasets/Validated_Test.txt'
# bert_model = "SZTAKI-HLT/hubert-base-cc"
# spacy_model_name = "hu_core_news_lg"                             # FROM: ["hu_core_news_lg", "hu_core_news_trf", "en_core_web_lg"]             

prediction_results_folder = "results/"


########################################################################################################################

model_parameters = {
    'dropout': 0.01,
    "bert_dim": 768,
    "polarities_dim": 3,
    'max_seq_len': 85,
    'bert_model_name': bert_model,
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
    'valset_ratio': 0                               # validation set (in terms of proportion of the test set)
}

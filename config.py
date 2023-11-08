# Traning configuration
model_name = "bert_spc"

# Train - Test set creation parameters

dateset_name = "Validated"
test_size = 0.2

text_column = "text"
NE_column = "Named Entity"
predictions_column = "Prediction"

checkpoint = '../state_dict/HUN_bert_spc_validated_val_acc_0.7159'

model_parameters = {
    'dropout': 0.01,
    "bert_dim": 768,
    "polarities_dim": 3,
    'max_seq_len': 85,
    # 'bert_model_name': "SZTAKI-HLT/hubert-base-cc"    # HUN
    'bert_model_name': "bert-base-cased"              # ENG
}

# ABSA-PyTorch

Currently used python files:

### Prediction:
- Main script: `./examples/examples_predict.py `
- Dependencies: 
    - Data preparation from 'raw' xlsx to prediction ready format (NER): `./preprocessors/prepeare_data_for_prediction.py` (`DataPreparator` class)
    - Prediction: `./src/prediction.py` (`Predictor` class)
    - Configurations (for all the above): `./config.py` 

### Training:
- Main script: `./examples/examples_train.py`
- Dependencies: 
    - `./src/training.py` (`Trainer` class)
    - Configurations: `./config.py` 

## Licence

MIT

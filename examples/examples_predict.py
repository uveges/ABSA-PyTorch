import os

import pandas as pd
from tqdm import tqdm

import config
from preprocessors.prepeare_data_for_prediction import DataPreparator
from src.prediction import Predictor


def simplest_case_usage():
    file_path = '../datasets/parl_speech_7_segmented_part_0.xlsx'

    ################################################ Keep intact #######################################################
    df = pd.read_excel(file_path)

    preparator = DataPreparator(dataframe=df)
    data_dict = preparator.start()

    predictor = Predictor(state_dict=config.checkpoint)
    predictions = []

    for sent, aspect in tqdm(zip(data_dict[config.text_column], data_dict[config.NE_column])):
        prediction = predictor.predict(text=sent, named_entity=aspect)
        predictions.extend(prediction)

    data_dict[config.predictions_column] = predictions
    result_frame = pd.DataFrame.from_dict(data_dict)
    filename = file_path.split("/")[-1]
    result_path = os.path.join(config.prediction_results_folder, filename.replace('.xlsx', '_predictions.xlsx'))
    result_frame.to_excel(result_path)


def more_files():

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
        filename = file.split("/")[-1]
        result_path = os.path.join(config.prediction_results_folder, filename.replace('.xlsx', '_predictions.xlsx'))
        result_frame.to_excel(result_path)


if __name__ == '__main__':
    simplest_case_usage()
    # more_files()
    
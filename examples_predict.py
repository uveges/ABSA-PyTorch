import os

import pandas as pd
from tqdm import tqdm

import config
from preprocessors.prepeare_data_for_prediction import DataPreparator
from src.prediction import Predictor


def simplest_case_usage():
    file_path = 'resources/parl_speech_7_segmented_part_13.xlsx'

    ################################################ Keep intact #######################################################
    df = pd.read_excel(file_path)

    preparator = DataPreparator(dataframe=df, huspacy_model_name="hu_core_news_lg")
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

    folder = 'resources'

    for file in os.listdir(folder):
        f = os.path.join(folder, file)
        if os.path.isfile(f) and f.endswith('.xlsx'):
            df = pd.read_excel(file)
            print(f'File read for preprocess: {file}')

            preparator = DataPreparator(dataframe=df)
            data_dict = preparator.start()
            print('Dictionary format from preprocess created')

            predictor = Predictor(state_dict=config.checkpoint)
            predictions = []
            for sent, aspect in tqdm(zip(data_dict[config.text_column], data_dict[config.NE_column])):
                prediction = predictor.predict(text=sent, named_entity=aspect)
                predictions.extend(prediction)
            print(f"predictions created. File length: {len(df.index)}, predictions: {len(predictions)}")

            data_dict[config.predictions_column] = predictions
            result_frame = pd.DataFrame.from_dict(data_dict)
            print("Result dataframe created")

            filename = f.split("/")[-1]
            result_path = os.path.join(config.prediction_results_folder, filename.replace('.xlsx', '_predictions.xlsx'))
            print(f'Path for results: {result_path}')

            result_frame.to_excel(result_path)
            print("Predictions written into file")

    # for part in range(14):
    #     file = f"../datasets/parl_speech_7_segmented_part_{part}.xlsx"
    #     df = pd.read_excel(file)
    #
    #     preparator = DataPreparator(dataframe=df)
    #     data_dict = preparator.start()
    #
    #     predictor = Predictor(state_dict=config.checkpoint)
    #     predictions = []
    #     print("Generating predictions...")
    #
    #     for sent, aspect in tqdm(zip(data_dict[config.text_column], data_dict[config.NE_column])):
    #         prediction = predictor.predict(text=sent, named_entity=aspect)
    #         predictions.extend(prediction)
    #
    #     data_dict[config.predictions_column] = predictions
    #     result_frame = pd.DataFrame.from_dict(data_dict)
    #     filename = file.split("/")[-1]
    #     result_path = os.path.join(config.prediction_results_folder, filename.replace('.xlsx', '_predictions.xlsx'))
    #     result_frame.to_excel(result_path)


if __name__ == '__main__':
    # simplest_case_usage()
    more_files()
    
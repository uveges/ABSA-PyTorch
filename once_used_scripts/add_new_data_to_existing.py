import pandas as pd
from preprocessors.dataset_preparator import *
from typing import Dict

def main():
    date_ = "20230102"

    # HUN:
    # new_path = f"/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/absa_training_{date_}.xlsx"
    # previous_train = "/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/Validated_Train.txt"
    # previous_test = "/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/Validated_Test.txt"
    # new_combined_txt = f"/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/{date_}_train_test_before_stratified_split.txt"
    # result_file_name = f"Dataset_{date_}"

    # ENG:
    new_path = f"/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/ENG_absa_training_{date_}.xlsx"
    previous_train = "/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/train_english.txt"
    previous_test = "/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/test_english.txt"
    new_combined_txt = f"/home/istvanu/PycharmProjects/ABSA-PyTorch/datasets/ENG_{date_}_train_test_before_stratified_split.txt"
    result_file_name = f"ENG_Dataset_{date_}"

    new_data_df = pd.read_excel(new_path)

    new_excel_columns = {
        'labels_col': 'Label',
        'aspect_col': 'Named Entity',
        'sentence_col': 'text'
    }

    combined_df = merge_existed_txt_with_new_excel(previous_train=previous_train,
                                                   previous_test=previous_test,
                                                   new_excel=new_data_df,
                                                   new_data_columns=new_excel_columns)


    serialize_txt_from_df(path=new_combined_txt,
                          pandas_dataframe=combined_df,
                          column_names=new_excel_columns)

    preparator = DatasetPreparator(dataset_path=new_combined_txt,
                                   result_file_name=result_file_name,
                                   test_size=config.test_size)
    preparator.stratified_split(verbose=True)


def serialize_txt_from_df(path: str, pandas_dataframe: pd.DataFrame, column_names: Dict):
    """
    Sentence
    Aspect
    Label
    """
    result_lines = []
    for index, row in pandas_dataframe.iterrows():
        result_lines.append(row[column_names['sentence_col']])
        result_lines.append(row[column_names['aspect_col']])
        result_lines.append(int(row[column_names['labels_col']]) -1)

    with open(path, 'w', encoding='utf8') as output_:
        for l in result_lines:
            output_.write(str(l) + '\n')


def merge_existed_txt_with_new_excel(previous_train: str, previous_test: str, new_excel: pd.DataFrame, new_data_columns: Dict):
    sentences, aspects, labels = ([] for i in range(3))
    for path_ in [previous_test, previous_train]:
        with open(path_, 'r', encoding='utf8') as input_:
            lines = input_.readlines()
            for label_index in range(3, len(lines), 3):
                sentences.append(lines[label_index])
                aspects.append(lines[label_index-2])
                labels.append(int(lines[label_index-1].strip().replace('\n', '')) + 1)

    for index_, line in new_excel.iterrows():
        sentences.append(line[new_data_columns['sentence_col']])
        aspects.append(line[new_data_columns['aspect_col']])
        labels.append(line[new_data_columns['labels_col']])

    aspects = [l.strip().replace('\n', '') for l in aspects]
    sentences = [l.strip().replace('\n', '') for l in sentences]

    result_dict = {}
    for v in new_data_columns.values():
        result_dict[v] = []
    result_dict[new_data_columns['sentence_col']] = sentences
    result_dict[new_data_columns['aspect_col']] = aspects
    result_dict[new_data_columns['labels_col']] = labels

    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df


if __name__ == '__main__':
    main()

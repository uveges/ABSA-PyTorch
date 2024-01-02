import pandas as pd
from sklearn.model_selection import train_test_split

import config


def merge_earlier_not_stratified_train_test_files(train_path: str, test_path: str,
                                                  combined_path: str) -> None:
    with open(train_path, 'r', encoding='utf8') as train_input, open(test_path, 'r', encoding='utf8') as test_input:
        train_lines = train_input.readlines()
        test_lines = test_input.readlines()
    with open(combined_path, 'w', encoding='utf8') as combined_output:
        for l in train_lines:
            combined_output.write(l)
        for l in test_lines:
            combined_output.write(l)


def create_short_test_data(data_file: str, test_examples_amount: int) -> None:
    with open(data_file, 'r', encoding='utf8') as train_:
        content = train_.readlines()
    # if len(content) % 3 != 0:
    #     raise AttributeError(f"Potentially corrupted file. The number of lines ({len(content)}) is not divisible by 3 (Text, Aspect, Label)!")
    test_lines = test_examples_amount * 3
    if test_lines > len(content):
        test_lines = len(content)

    test_set = content[:test_lines]

    with open("../datasets/test_file.txt", 'w', encoding='utf8') as test_results:
        for l in test_set:
            test_results.write(l)


def print_statistics(train_data, train_labels, test_data, test_labels):
    train_data['Label'] = train_labels['Label'].values.tolist()
    test_data['Label'] = test_labels['Label'].values.tolist()

    train_metadata = train_data.groupby('Label')
    test_metadata = test_data.groupby('Label')
    train_keys = [k.replace('\n', '') for k in train_metadata.groups.keys()]
    test_keys = [k.replace('\n', '') for k in test_metadata.groups.keys()]
    print(f"Train set: {train_keys}: {train_metadata.size().values.tolist()}, Test set: {test_keys}: {test_metadata.size().values.tolist()}")


def create_traning_file(result_path: str, dataset: pd.DataFrame) -> None:
    file_content = []
    for index, row in dataset.iterrows():
        file_content.append(row['Sentence'])
        file_content.append(row['Aspect'])
        file_content.append(row['Label'])

    with open(result_path, 'w', encoding='utf8') as output:
        for l in file_content:
            output.write(l)
    print(f"File created: {result_path}")


class DatasetPreparator(object):

    def __init__(self, dataset_path: str, result_file_name: str = "Dataset", test_size: float = 0.2):
        """
        dataset_path: .txt file, where a masked sentence, aspect (str), label (int) convention is followed.
        """
        self.dataset_path = dataset_path
        self.result_file_name = result_file_name
        self.test_size = test_size

    def stratified_split(self, verbose: bool = False) -> None:
        with open(self.dataset_path, 'r', encoding='utf8') as test_data:
            test_data_lines = test_data.readlines()
        start, end = 0, 3
        test_frame = pd.DataFrame(columns=['Sentence', 'Aspect', 'Label'])
        while end < len(test_data_lines):
            part = test_data_lines[start:end]
            test_frame.loc[len(test_frame)] = part
            start += 3
            end += 3

        data = test_frame[['Sentence', 'Aspect']].copy()
        labels = test_frame[['Label']].copy()

        train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                            labels,
                                                                            test_size=self.test_size,
                                                                            random_state=42,
                                                                            stratify=labels)
        if verbose:
            print_statistics(train_data, train_labels, test_data, test_labels)

        create_traning_file(result_path=f"../datasets/semeval14/{self.result_file_name}_Train.txt", dataset=train_data)
        create_traning_file(result_path=f"../datasets/semeval14/{self.result_file_name}_Test.txt", dataset=test_data)


if __name__ == '__main__':
    # train_file = "../datasets/OpinHuBank_Train.txt"
    # test_file = "../datasets/OpinHuBank_Test.txt"
    # combined_temporary = "../datasets/merged.txt"
    # short_test_file = "../datasets/test_file.txt"

    # merge_earlier_not_stratified_train_test_files(train_path=train_file,
    #                                               test_path=test_file,
    #                                               combined_path=combined_temporary)
    #
    # create_short_test_data(data_file=combined_temporary,
    #                        test_examples_amount=500)



    short_test_file = "../datasets/HunEmPoli_8_ner_valid_txt_RECODED.txt"

    preparator = DatasetPreparator(dataset_path=short_test_file,
                                   result_file_name=config.dateset_name,
                                   test_size=config.test_size)
    preparator.stratified_split(verbose=True)

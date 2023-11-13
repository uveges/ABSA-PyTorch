import config
import pandas as pd
import sys
from typing import List, Tuple, Dict
from tqdm import tqdm


class DataPreparator(object):
    """
    Class that creates data format ready for prediction. Input: any kind of excel file, that contains at least a
    Column with texts for prediction. Column name must be specified at `config.text_column`

    Attributes
    ----------
    dataframe : pd.DataFrame
        The original excel file loaded into a Pandas Dataframe
    huspacy_model_name : str
        Name of the Spacy model to be used for Named Entity Recognition

    Methods
    -------
    start():
        Retruns a Python dictionary that can be directly turned into a Pandas Dataframe again.

    """
    def __init__(self, dataframe: pd.DataFrame, huspacy_model_name: str = "hu_core_news_lg"):
        self.dataframe = dataframe
        self.original_data_list_per_column = {}                         # {column_name: [original values]}
        self.column_names = []
        self.result_data_list_per_column = {config.NE_column: []}       # {column_name: [original values]} --> ready for prediction
        self.model_name = huspacy_model_name
        self.nlp = None
        self.PATHS = {
            "hu_core_news_lg": "pip install https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.whl",
            "hu_core_news_trf": "pip install https://huggingface.co/huspacy/hu_core_news_trf/resolve/v3.5.2/hu_core_news_trf-any-py3-none-any.whl"
        }
        try:
            if self.model_name == "hu_core_news_lg":
                import hu_core_news_lg
                self.nlp = hu_core_news_lg.load()
            if self.model_name == "hu_core_news_trf":
                import hu_core_news_trf
                self.nlp = hu_core_news_trf.load()
        except (OSError, IOError) as e:
            print(f"Error! Language model not installed. You can install it by 'pip install {self.PATHS[self.model_name]}'")
            sys.exit(e)

    def start(self) -> Dict:
        """
        Creates the prediction-ready format in dictionary.
        First Named Entities recognised, then a sentence will have as many instances in the output as many NE it contained.
        Each of these sentences have exactly one NE masked out in it with '$T$'.
        Every original cells in a given line will be kept!

        """

        self.column_names = self.dataframe.columns.values.tolist()
        for c in self.column_names:
            if c not in self.original_data_list_per_column:
                self.original_data_list_per_column[c] = []
                self.result_data_list_per_column[c] = []
            self.original_data_list_per_column[c] = self.dataframe[c].values.tolist()
        print("Preprocess data...")
        for i, t in tqdm(enumerate(self.original_data_list_per_column[config.text_column])):
            sents, aspects = self.__preprocess_with_spacy(t)
            repetitions = len(sents)
            for column in self.column_names:
                if column == config.text_column:
                    for rep in range(repetitions):
                        self.result_data_list_per_column[column].append(sents[rep])
                else:
                    for rep in range(repetitions):
                        self.result_data_list_per_column[column].append(self.original_data_list_per_column[column][i])
            for rep in range(repetitions):
                self.result_data_list_per_column[config.NE_column].append(aspects[rep])

        return self.result_data_list_per_column

    def __preprocess_with_spacy(self, text: str) -> Tuple[List[str], List[str]]:

        preprocessed_sentences, named_entities = ([] for i in range(2))
        doc = self.nlp(text)
        for ent in doc.ents:
            lemma = self.nlp(ent.text)[0].lemma_
            start_index = ent.start_char
            end_index = start_index + len(lemma)
            preprocessed_sentences.append(text[:start_index] + "$T$" + text[end_index:])
            named_entities.append(lemma)

        return preprocessed_sentences, named_entities

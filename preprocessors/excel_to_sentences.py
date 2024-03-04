import os

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from preprocessors.sentence_splitter import SentenceSplitter

num_processes = mp.cpu_count() - 2 if mp.cpu_count() > 4 else mp.cpu_count()

splitter = SentenceSplitter(language="hu")


def main(excel_path: str, id_column: str, text_column: str) -> None:
    dataframe = open_excel(excel_path)
    print(len(dataframe))
    segmented_dataframe = segment_dataframe_into_sentences(
        dataframe, id_column, text_column
    )
    segmented_dataframe.to_csv(excel_path.replace(".csv", "_SEGMENTED.csv"))
    print(
        f"Exported {len(segmented_dataframe)} sentences to {excel_path.replace('.csv', '_SEGMENTED.csv')}"
    )


def segment_dataframe_into_sentences(
    dataframe: pd.DataFrame, id_column: str, text_column: str
) -> pd.DataFrame:
    result_frame = pd.DataFrame(columns=[id_column, text_column])
    result_list = []
    skipped_line_count = 0
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        id = row[id_column]
        text = row[text_column]
        if not text or not isinstance(text, str):
            # print(id, text)
            # print(f"Error at id: {id}. Line is empty or text is not a str!")
            # raise ValueError(f"Error at id: {id}. Line is empty or text is not a str!")
            skipped_line_count += 1
            result_list.append([id, ""])
            continue
        sentences = splitter.split(text.replace("\n", " ").replace("\r", " "))
        if len(sentences) > 1:
            id_postfixes = list(range(0, len(sentences)))
            for s, i in zip(sentences, id_postfixes):
                id_to_write = str(id) + "_" + str(i)
                result_list.append([id_to_write, s])
        else:
            for s in sentences:
                result_list.append([id, s])
    if result_list:
        result_frame = pd.DataFrame(result_list, columns=[id_column, text_column])
    else:
        print("No results!")
    return result_frame


def open_excel(path: str) -> pd.DataFrame:
    """Despite the name it opens a csv file."""
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    id_column = "speech_id_new"
    text_column = "text"
    excel_path = "../data/corpus_speeches_croatia_eng_v_0_1.csv"

    main(excel_path, id_column, text_column)

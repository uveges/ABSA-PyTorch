import re

import pandas as pd
import hu_core_news_lg


def main():
    path = '../datasets/Validated_Test.txt'
    output = '/home/istvanu/DATA/Validated_Test_to_translate.txt'
    nlp = hu_core_news_lg.load()

    with open(path, 'r', encoding='utf8') as eredeti:
        lines = eredeti.readlines()

    sentences, aspects, labels = ([] for x in range(3))
    for i in range(0, len(lines), 3):
        sentences.append(lines[i])
    for i in range(1, len(lines), 3):
        aspects.append(lines[i])
    for i in range(2, len(lines), 3):
        labels.append(lines[i])

    for sentence, aspect, label in zip(sentences, aspects, labels):
        doc = nlp(aspect)
        l_list = []
        for token in doc:
            l_list.append(token.lemma_)
        for i_orig, i_lemma in zip(aspect, " ".join(l_list)):
            print()




# emotion_dictionary = {
#     "Neutral": 0,
#     "0": 0,
#     "Fear": 1,
#     "fear": 1,
#     "Sadness": 2,
#     "sadness": 2,
#     "Anger": 3,
#     "anger": 3,
#     "Disgust": 4,
#     "disgust": 4,
#     "Success": 5,
#     "success": 5,
#     "Joy": 6,
#     "joy": 6,
#     "Trust": 7,
#     "trust": 7
# }

# emotion_dictionary = {
#     "Neutral": 0,
#     "0": 0,
#     "fear": -1,
#     "sadness": -1,
#     "anger": -1,
#     "disgust": -1,
#     "success": 1,
#     "joy": 1,
#     "trust": 1
# }
#
#
# def main():
#     file_path = "../resources/absa_training set_english_v1.xlsx"
#     output_path = "../resources/ENG_HunEmPoli_8_ner_valid.txt"
#     df = pd.read_excel(file_path)
#
#     column_names = df.columns.values.tolist()
#     column_names.remove('id')
#     column_names.remove('text')
#
#     named_entity_column_pat = re.compile(r'NE_[0-9]+')
#     label_column_pat = re.compile(r'NE_emotion_[0-9]+')
#     # serial_pat = re.compile(r'[0-9]+')
#
#     ne_columns = list(filter(named_entity_column_pat.match, column_names))
#     emotion_columns = list(filter(label_column_pat.match, column_names))
#
#     for rowIndex, row in df.iterrows():
#         labels = []
#         NE_s = []
#         for potential_ne, potential_label in zip(ne_columns, emotion_columns):
#             if not isinstance(row[potential_ne], str):
#                 continue
#             ne = (row[potential_ne].replace('(per)', '').replace('(PER)', '').replace('(loc)', '').replace('(LOC)', '').replace('(org)', '').replace('(ORG)', '').replace('(misc)', '').replace('(MISC)', '')).strip()
#             label = str(row[potential_label]).lower()
#             if label in emotion_dictionary:
#                 label = emotion_dictionary[label]
#                 # print(row['text'], '\r\n', ne, label)
#                 labels.append(label)
#                 NE_s.append(ne)
#         if NE_s:
#             # print(row['text'], NE_s, labels)
#             print(len(labels))



if __name__ == '__main__':
    main()

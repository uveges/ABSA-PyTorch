from tqdm import tqdm

coding_scheme = {
    '0': '0',           # Neutral
    '1': '-1',          # Fear
    '2': '-1',          # Sadness
    '3': '-1',          # Anger
    '4': '-1',          # Disgust
    '5': '1',           # Success
    '6': '1',           # Joy
    '7': '1',           # Trust
}


def main(label_statistics:bool = False):

    if label_statistics:
        emotions = dict.fromkeys(range(0, 8))
        sentiments = dict.fromkeys(range(-1, 2))

    emotion_coded_txt_path = "../resources/HunEmPoli_8_ner_valid_txt.txt"
    recoded = []
    with open(emotion_coded_txt_path, 'r', encoding='utf8') as original_:
        lines = original_.readlines()
    for i in tqdm(range(0, len(lines) - 3, 3)):
        polarity = lines[i + 2].strip()
        try:
            p = int(polarity)
        except Exception as e:
            print(i, polarity)
            continue

        if label_statistics:
            if not emotions[int(polarity)]:
                emotions[int(polarity)] = 0
            emotions[int(polarity)] = emotions[int(polarity)] + 1

        recoded.append(lines[i])
        recoded.append(lines[i+1])
        polarity = coding_scheme[polarity]

        if label_statistics:
            if not sentiments[int(polarity)]:
                sentiments[int(polarity)] = 0
            sentiments[int(polarity)] = sentiments[int(polarity)] + 1

        recoded.append(polarity + '\n')

    if label_statistics:
        print(emotions, '\n', sentiments)

    with open(emotion_coded_txt_path.replace('.txt', '_RECODED.txt'), 'w', encoding='utf8') as result_:
        for l in recoded:
            result_.write(l)


if __name__ == '__main__':
    main(label_statistics=True)

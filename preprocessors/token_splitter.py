from transformers import AutoTokenizer
from typing import List, Union


class HuBERTtruncator():

    def __init__(self, max_length: int = 505):
        self.tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        self.max_length = max_length

    def start(self, sentences: Union[str, List]) -> List:

        if isinstance(sentences, str):
            sentences = [sentences]
        # print(sentences)

        results = []
        for sentence in sentences:
            encoded = self.tokenizer.encode_plus(
                text=sentence,
                truncation=True,
                max_length=self.max_length,
            ).get('input_ids')

            # print(len(encoded), len(sentence))

            if len(encoded) < self.max_length:
                encoded = encoded[:self.max_length]

            # print(len(encoded), len(self.tokenizer.decode(encoded, skip_special_tokens=True)))
            results.append(self.tokenizer.decode(encoded, skip_special_tokens=True))

        return results

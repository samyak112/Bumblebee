import re
from typing import List, Optional


class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]):
        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
        self.sentences = list_of_sentences
        if not list_of_sentences:
            return
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

        print(self.token2index)
        print(self.index2token)

    def add_tokens(self, tokens: List[str]) -> None:

        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Split on all tokens and punctuation. Optionally adds BOS and EOS tokens.
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token2index[token] for token in tokens]

    def batch_encode(
        self, padding=True, add_special_tokens: bool = False
    ) -> List[List[int]]:
        sentences = self.sentences
        tokenized_sentences = [
            self.encode(sentence, add_special_tokens) for sentence in sentences
        ]
        if padding:
            max_length = max([len(tokens) for tokens in tokenized_sentences])
            tokenized_sentences = [
                s + ((max_length - len(s)) * [self.token2index[self.PAD]])
                for s in tokenized_sentences
            ]
        return tokenized_sentences


vocab = Vocabulary(list_of_sentences=["Hello world!", "How are you? asd"])
print(vocab.batch_encode())
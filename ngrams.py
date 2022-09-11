from pathlib import Path
import pickle

import numpy as np


class N_grams():
    """
    Trigram model class that contains methods for prepocessing,
    training, prediction, saving and loading.
    """

    def __init__(self, weights: dict = None):
        self.bigrams = {}
        if weights is None:
            self.weights = {}
        else:
            self.weights = weights

    def tokenize(self, text: str) -> list:
        """Process the text into a list of tokens."""
        processed = text.lower()
        processed = [t.strip('….,-!:;“„”—–«»?()""') for t in processed.split()]
        processed = [t for t in processed if t]
        return processed

    def fit_processed(self, processed: list, preprev: str, prev: str):
        """Train on a list of tokens."""
        for i in range(len(processed)):
            current = processed[i]
            if (preprev, prev) not in self.weights:
                self.weights[(preprev, prev)] = [{}, [], []]
            tok2id, toks, count = self.weights[(preprev, prev)]
            if current not in tok2id:
                tok2id[current] = len(toks)
                toks.append(current)
                count.append(1)
            else:
                count[tok2id[current]] += 1
            preprev, prev = prev, current
        return preprev, prev

    def fit(self, path: str = None):
        """
        Train trigrams on a directory of text files or an input string.
        """
        if not path:
            text = input('Enter the text to train on: ')
            text = self.tokenize(text)
            self.fit_processed(text, '<eos>', '<bos>')
        else:
            path = Path(path)
            for p in path.glob('*.txt'):
                with open(p, 'r') as file:
                    preprev, prev = '<eos>', '<bos>'
                    for line in file:
                        processed = self.tokenize(line)
                        preprev, prev = self.fit_processed(
                            processed, preprev, prev)
        # Normalize to get probabilities
        for w in self.weights.values():
            probs = np.array(w[-1])
            probs = probs / probs.sum()
            w[-1] = probs

    def build_bigrams(self):
        """
        Populate self.bigrams from self.weights for one-word prediction.
        """
        for k, v in self.weights.keys():
            if k in self.bigrams:
                self.bigrams[k].append(v)
            else:
                self.bigrams[k] = [v]

    def predict(self, first: str = None, second: str = None) -> str:
        """Predict the next word for a 0-to-2-word prompt"""
        if first is None and second is None:
            first, second = '<eos>', '<bos>'
        elif second is None:
            first, second = '<bos>', first
        elif first is None:
            first, second = '<bos>', second
        if (first, second) in self.weights:
            _, toks, probs = self.weights[(first, second)]
            return np.random.choice(toks, 1, p=probs)[0]
        else:
            self.build_bigrams()
            if second in self.bigrams:
                return np.random.choice(self.bigrams[second], 1)[0]
            else:
                return np.random.choice(list(self.bigrams.keys()), 1)[0]

    def save(self, path: Path):
        """Save the weights of the model to path."""
        with open(path, 'wb') as file:
            pickle.dump(self.weights, file)

    @classmethod
    def load(cls, path: Path):
        """Create a model with weights from path."""
        with open(path, 'rb') as file:
            weights = pickle.load(file)
        return cls(weights)

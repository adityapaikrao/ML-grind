import numpy as np
from typing import List, Tuple, Any
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

rng = np.random.default_rng(seed=0)
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words("english"))


def preprocess_text(doc: str) -> str:
    """
    Preprocess a word doc
    """
    doc = doc.lower()  # lowercase
    doc = doc.translate(str.maketrans("", "", string.punctuation))  # remove punctuations
    tokens = doc.split()  # remove extra whitespaces
    tokens = [token for token in tokens if token not in stopwords_set]  # remove stopwords

    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # lemmatize tokens
    return " ".join(tokens)


class SkipGram:
    def __init__(
        self,
        corpus: List[str],
        emb_dim: int = 300,
        context_size: int = 2,
    ):

        self.vocab = {}  # word -> idx
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.dataset: List[Any] = []  # (input_word, target_words) pairs
        idx = 0
        for doc in corpus:
            cleaned_doc = preprocess_text(doc).split()
            for i, word in enumerate(cleaned_doc):
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
                if context_size <= i < len(cleaned_doc) - context_size:
                    input_word = cleaned_doc[i]
                    target_words = cleaned_doc[i - context_size : i] + cleaned_doc[i + 1 : i + context_size + 1]
                    self.dataset.append([input_word, target_words])

        self.vocab_size = len(self.vocab)
        self.embeddings_Vd = rng.normal(size=(self.vocab_size, emb_dim)) * 0.01
        self.Wo_dV = rng.normal(size=(emb_dim, self.vocab_size)) * 0.01

    def train(self, epochs: int = 100, learning_rate: float = 1e-2, batch_size: int = 32) -> None:
        C = self.context_size * 2
        input_one_hot_NV = np.array(
            [np.eye(self.vocab_size)[self.vocab[pair[0]]] for pair in self.dataset]
        )
        target_one_hot_NCV = np.array(
            [
                [np.eye(self.vocab_size)[self.vocab[word]] for word in pair[1]]
                for pair in self.dataset
            ]
        )

        def softmax(x: np.ndarray) -> np.ndarray:
            xmax = x.max(axis=2, keepdims=True)
            z = np.exp(x - xmax)  # scale to avoid out overflow
            z /= z.sum(axis=2, keepdims=True) + 1e-6
            return z

        for _ in range(epochs):
            for start in range(0, input_one_hot_NV.shape[0], batch_size):
                end = min(start + batch_size, input_one_hot_NV.shape[0])
                input_BV, target_BCV = input_one_hot_NV[start:end], target_one_hot_NCV[start:end]

                # forward pass
                input_emb_Bd = input_BV @ self.embeddings_Vd  # (B, d)
                input_emb_BCd = np.repeat(input_emb_Bd[:, None, :], C, axis=1)  # (B, C, d)

                yhat_BCV = softmax(input_emb_BCd @ self.Wo_dV)  # (B, C, V)

                # backprop
                d_error2_BCV = (yhat_BCV - target_BCV) / (input_BV.shape[0] * C)  # (B, C, V)
                d_wograd = input_emb_BCd.reshape(-1, self.emb_dim).T @ d_error2_BCV.reshape(-1, self.vocab_size)  # (d, V)

                d_error1_BCd = d_error2_BCV @ self.Wo_dV.T  # (B, C, d)
                d_error1_Bd = d_error1_BCd.mean(axis=1)  # (B, d)
                d_emb_Vd = input_BV.T @ d_error1_Bd  # (V, d)

                # grad updates
                self.Wo_dV -= learning_rate * d_wograd
                self.embeddings_Vd -= learning_rate * d_emb_Vd


# Pytorch version
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SkipGramtorch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embeddings_Vd = nn.Embedding(self.vocab_size, self.emb_dim)
        self.Wo_dV = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X_emb_BD = self.embeddings_Vd(input)  # (B, d)
        return self.Wo_dV(X_emb_BD)  # (B, V)


class SkipGramDataset(Dataset):
    def __init__(self, corpus: List[str], context_size: int = 2):
        self.vocab = {}  # word -> idx

        self.dataset: List[Any] = []  # (input_word, target_words) pairs
        idx = 0
        for doc in corpus:
            cleaned_doc = preprocess_text(doc).split()
            for i, word in enumerate(cleaned_doc):
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
                if context_size <= i < len(cleaned_doc) - context_size:
                    input_word = cleaned_doc[i]
                    target_words = cleaned_doc[i - context_size : i] + cleaned_doc[i + 1 : i + context_size + 1]
                    self.dataset.append([input_word, target_words])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_word, target_words = self.dataset[idx]
        input_indices = torch.tensor([self.vocab[input_word]])
        target_indices = torch.tensor([self.vocab[w] for w in target_words])
        return input_indices, target_indices

    def get_vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    corpus = [""]
    dataset = SkipGramDataset(corpus)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    model = SkipGramtorch(dataset.get_vocab_size())
    loss_fn = F.cross_entropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    epochs = 100
    for _ in range(epochs):
        total_loss = 0
        for input_word, target_words in dataloader:
            optimizer.zero_grad()

            input_word = input_word.squeeze(1)
            logits = model(input_word)  # (B, V)
            loss = sum(loss_fn(logits, target_words[:, i]) for i in range(target_words.shape[1])) / target_words.shape[1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

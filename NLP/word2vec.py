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
    doc = doc.lower() # lowercase
    doc = doc.translate(str.maketrans("", "", string.punctuation)) # remove punctuations
    tokens = doc.split() # remove extra whitespaces
    tokens = [token for token in tokens if token not in stopwords_set] # remove stopwords

    tokens = [lemmatizer.lemmatize(token) for token in tokens] # lemmatize tokens
    return " ".join(tokens)     

class CBOW:
    def __init__(self, 
                 corpus: List[str],
                 emb_dim: int = 300,
                 context_size: int = 2
                 ):
        
        self.vocab = {} # word -> idx
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.dataset: List[Any] = [] # (target, context) pairs
        idx = 0
        for doc in corpus:
            cleaned_doc = preprocess_text(doc).split()
            for i, word in enumerate(cleaned_doc):
                if word not in self.vocab:
                    self.vocab[word] = idx 
                    idx += 1
                if context_size <= i < len(cleaned_doc) - context_size:
                    context = cleaned_doc[i-context_size:i] + cleaned_doc[i + 1: i + context_size + 1]
                    target = cleaned_doc[i]
                    self.dataset.append([target, context])
        
        self.vocab_size = len(self.vocab)
        self.embeddings_Vd = rng.normal(size=(self.vocab_size, emb_dim)) * 0.01
        self.Wo_dV = rng.normal(size=(emb_dim,self.vocab_size)) * 0.01
    
    def train(self, epochs: int = 100, learning_rate: float = 1e-2, batch_size: int = 32) -> None:
            C = self.context_size * 2
            X_one_hot_NCV = np.array(
                [
                    [np.eye(self.vocab_size)[self.vocab[word]] for word in pair[1]]
                    for pair in self.dataset
                ]
            )
            Y_one_hot_NV = np.array(
                [np.eye(self.vocab_size)[self.vocab[pair[0]]] for pair in self.dataset]
            )

            def softmax(x: np.ndarray) -> np.ndarray:
                xmax = x.max(axis=1, keepdims=True)
                z = np.exp(x - xmax) # scale to avoid out overflow
                z /= z.sum(axis=1, keepdims=True) + 1e-6
                return z

            for _ in range(epochs):
                 for start in range(0, X_one_hot_NCV.shape[0], batch_size):
                    end = min(start + batch_size, X_one_hot_NCV.shape[0])            
                    X_BCV, Y_BV = X_one_hot_NCV[start:end], Y_one_hot_NV[start:end] 
                    
                    # forward pass
                    X_emb_Bd = X_BCV @ self.embeddings_Vd # (B, C, d)
                    X_emb_Bd = X_emb_Bd.mean(axis=1) # (B, d)

                    yhat_BV = softmax(X_emb_Bd @ self.Wo_dV) # (B, V)

                    # backprop 
                    d_error2_BV = (yhat_BV - Y_BV) / X_BCV.shape[0] # (B, V)
                    d_wograd = X_emb_Bd.T @ d_error2_BV  # (d, V)
                    
                    d_error1_Bd = d_error2_BV @ self.Wo_dV.T # (B, d)
                    # (B, C, V) @ (B, d) -> (B * C, V).T @ (B * C, d) -> (V, d)
                    X_in = X_BCV.reshape(X_BCV.shape[0] * C, self.vocab_size)
                    d_error1 = np.repeat(d_error1_Bd, C, axis=0)
                    d_emb_Vd = X_in.T @ d_error1 / C # (V, B*C) @ (B*C, d)
                    
                    # grad updates
                    self.Wo_dV -= learning_rate * d_wograd
                    self.embeddings_Vd -= learning_rate * d_emb_Vd



# Pytorch version
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
class CBOWtorch(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 emb_dim: int = 300,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embeddings_Vd = nn.Embedding(self.vocab_size, self.emb_dim)
        self.Wo_dV = nn.Linear(self.emb_dim, self.vocab_size, bias=False)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X_emb_BD = self.embeddings_Vd(input) # (B, C, d)
        X_emb_BD = X_emb_BD.mean(dim=1)
        
        return self.Wo_dV(X_emb_BD)

class CBOWDataset(Dataset):
    def __init__(self, corpus: List[str], context_size: int = 2):
        self.vocab = {} # word -> idx

        self.dataset: List[Any] = [] # (target, context) pairs
        idx = 0
        for doc in corpus:
            cleaned_doc = preprocess_text(doc).split()
            for i, word in enumerate(cleaned_doc):
                if word not in self.vocab:
                    self.vocab[word] = idx 
                    idx += 1
                if context_size <= i < len(cleaned_doc) - context_size:
                    context = cleaned_doc[i-context_size:i] + cleaned_doc[i + 1: i + context_size + 1]
                    target = cleaned_doc[i]
                    self.dataset.append([target, context])
        
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target, context = self.dataset[idx]
        target_indices = torch.tensor([self.vocab[target]])
        context_indices = torch.tensor([self.vocab[w] for w in  context])
        return target_indices, context_indices
    def get_vocab_size(self) -> int:
        return len(self.vocab)

corpus = [""]
dataset = CBOWDataset(corpus)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model = CBOWtorch(dataset.get_vocab_size())
loss_fn = F.cross_entropy
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
epochs = 100
for _ in range(epochs):
    total_loss = 0
    for target, context in dataloader:
        optimizer.zero_grad()
        
        loss = loss_fn(model(context), target.squeeze(1)) # squeeze to convert (B, 1) -> (B,); squeeze(1) 
                                                          # to avoid it becoming scalar
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

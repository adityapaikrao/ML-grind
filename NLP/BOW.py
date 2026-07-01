import numpy as np
from typing import List, Tuple, Any
import heapq
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stopwords = set(stopwords.words("english"))

class BOWSim:
    def __init__(self, 
                 corpus: List[str]):
        self.vocab = {} # word -> index

        idx = 0
        for doc in corpus:
            for word in self._preprocess_doc(doc).split():
                if word not in self.vocab:
                    self.vocab[word] = idx 
                    idx += 1
        
        self.vocab_size = len(self.vocab)
        self.docs = corpus
        self.doc_embeddings: List[np.ndarray] = [self.get_bow_embeddings(self._preprocess_doc(doc)) for doc in corpus] # D * V 
    
    def get_bow_embeddings(self, doc: str) -> np.ndarray:
        embedding = np.zeros((1, self.vocab_size))
        for word in self._preprocess_doc(doc).split():
            if word in self.vocab:
                embedding[0, self.vocab[word]] += 1
        return embedding
    
    def _preprocess_doc(self, text:str) -> str:
        """
        Preprocesses a piece of text. Lowercase, whitespace/punctuation removal, 
        """
        text = text.lower() # convert string to lowercase
        text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctuations replace "" with ""
        tokens = text.split() # remove extra whitespace

        # stemming vs lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return "".join(tokens)


    def get_topk_similar(self, query: str, k: int = 4) -> List[str]:
        query = self._preprocess_doc(query)
        query_vector = self.get_bow_embeddings(query)
        min_heap: Tuple[float, int] | Any = [] # (sim_score, other_doc_index)

        for idx, other_doc_vector in enumerate(self.doc_embeddings):
            sim_score = self._compute_cosine_similarity(query_vector, other_doc_vector)
            heapq.heappush(min_heap, (sim_score, idx))

            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return [self.docs[idx] for _, idx in sorted(min_heap, reverse=True)]

    def _compute_cosine_similarity(self, doc: np.ndarray, other_doc: np.ndarray) -> float:
        denom = (np.linalg.norm(doc) * np.linalg.norm(other_doc))
        if denom == 0.0:
            return 0.0
        
        return float((doc @ other_doc).item() / denom)
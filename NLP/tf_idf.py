import numpy as np
from typing import List

class TFIDF:
    def __init__(self, 
                 corpus: List[str]): 
        
        self._idf = {}
        self.docs = [self._preprocess_doc(doc) for doc in corpus]
    
    def _preprocess_doc(self, string: str) -> str:
        """
        Preprocess a given doc by removing punctuation, lowercasing etc
        """
        string = string.lower()

        return string


    def tf(self, term: str, doc: str) -> float:
        """
        Returns the term frequency of term in the given document.
        """
        words = self._preprocess_doc(doc).split()
        if len(words) == 0:
            raise ValueError("doc cannot be empty")
    
        count = 0
        for word in words: 
            if word == self._preprocess_doc(term): count += 1

        return count / len(words)

    def idf(self, term: str) -> float:
        """
        Returns the inverse document frequency (IDF) value for a term in the corpus
        """
        term = self._preprocess_doc(term)

        if term not in self._idf:
            N = len(self.docs)
            count_present = 0
            for doc in self.docs:
                words = doc.split()
                if term in words: count_present += 1

            idf = np.log(N/count_present)        
            self._idf[term] = idf

        return self._idf[term]

    def tf_idf(self, term: str, doc: str) -> float:
        """
        Returns the tf-idf score of a term for a given doc
        """
        tf = self.tf(term, doc)
        idf = self.idf(term)
        return tf * idf


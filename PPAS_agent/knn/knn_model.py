import numpy as np
import faiss


class KNeighbors:
    def __init__(self, k=1, nlist=1000, nprobe=1):
        self.index = None
        self.y = None
        self.k = 1
        self.nlist = nlist
        self.nprobe = nprobe

    def fit(self, X, y):
        quantizer = faiss.IndexFlatL2(X.shape[1])  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, X.shape[1], self.nlist)
        # Train:
        assert not self.index.is_trained
        self.index.train(X.astype(np.float32))
        assert self.index.is_trained
        # Fit:
        self.index.add(X.astype(np.float32))
        self.index.nprobe = self.nprobe
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=1)
        index = indices[0][0]
        return self.y[index]

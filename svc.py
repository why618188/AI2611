import numpy as np
from tqdm import tqdm
from itertools import combinations
from svm import SVM
from mksvm import MKSVM


class SVC:
    def __init__(self, kernel, penalty=3.0, max_iter=100, epsilon=1e-3, gamma=1.0, degree=3, bias=1.0,
                 heuristic=True, strategy='OvsO', multi_kernel=False):
        valid_strategies = ['OvsO', 'OvsR']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid kernel. Expected one of {valid_strategies}, got {strategy}.")

        self.penalty = penalty
        self.kernel = kernel
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.bias = bias
        self.heuristic = heuristic
        self.multi_kernel = multi_kernel
        self.strategy = strategy
        self.classifiers = []
        self.labels = []


    def fit(self, X, y):
        self.labels = np.unique(y)
        if self.strategy == 'OvsR':
            self._build_model_OvsR(self.labels)
            for label, svm in tqdm(self.classifiers):
                y_binary = np.where(y == label, 1, -1)
                svm.fit(X, y_binary)
        elif self.strategy == 'OvsO':
            self._build_model_OvsO(self.labels)
            for label1, label2, svm in tqdm(self.classifiers):
                idx = np.where((y == label1) | (y == label2))
                X_pair = X[idx]
                y_pair = y[idx]
                y_pair = np.where(y_pair == label1, -1, 1)
                svm.fit(X_pair, y_pair)

    def _build_model_OvsR(self, labels):
        for label in labels:
            if self.multi_kernel:
                svm = MKSVM(kernels=self.kernel, penalty=self.penalty, max_iter=self.max_iter, epsilon=self.epsilon,
                            gamma=self.gamma, degree=self.degree, bias=self.bias)
                self.classifiers.append((label, svm))
            else:
                svm = SVM(kernel=self.kernel, penalty=self.penalty, max_iter=self.max_iter, epsilon=self.epsilon,
                           gamma=self.gamma, degree=self.degree, bias=self.bias)
                self.classifiers.append((label, svm))


    def _build_model_OvsO(self, labels):
        for (label1, label2) in combinations(labels, 2):
            if self.multi_kernel:
                svm = MKSVM(kernels=self.kernel, penalty=self.penalty, max_iter=self.max_iter, epsilon=self.epsilon,
                          gamma=self.gamma, degree=self.degree, bias=self.bias)
                self.classifiers.append((label1, label2, svm))
            else:
                svm = SVM(kernel=self.kernel, penalty=self.penalty, max_iter=self.max_iter, epsilon=self.epsilon,
                          gamma=self.gamma, degree=self.degree, bias=self.bias)
                self.classifiers.append((label1, label2, svm))


    def predict(self, X):
        if self.strategy == 'OvsR':
            confidences = np.zeros((X.shape[0], len(self.classifiers)))
            for label, svm in tqdm(self.classifiers):
                confidences[:, label] = svm.get_predict_value(X)
            return self.labels[np.argmax(confidences, axis=1)]
        elif self.strategy == 'OvsO':
            vote = np.zeros((X.shape[0], len(self.labels)))
            for label1, label2, svm in tqdm(self.classifiers):
                preds = svm.predict(X)
                vote[:, label1] += (preds == -1)
                vote[:, label2] += (preds == 1)
            return self.labels[np.argmax(vote, axis=1)]


    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


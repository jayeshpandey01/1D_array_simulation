# Random Forest algorithm on 1D signal data using pytorch

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# random forest using torch is not straightforward, we will use sklearn's RandomForestClassifier for this purpose
class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            max_depth=self.max_depth, 
                                            random_state=self.random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    def feature_importances(self):
        return self.model.feature_importances_

def main():
    df = pd.read_csv('../datasets/train')
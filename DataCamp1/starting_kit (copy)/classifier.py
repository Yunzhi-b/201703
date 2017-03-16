from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 10
        self.n_estimators = 1000
	self.max_depth = None
	self.max_features = "auto"
        self.clf = Pipeline([
#	     	('lda',LinearDiscriminantAnalysis(n_components=self.n_components)),
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,max_features=self.max_features,
                                           random_state=42))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
	

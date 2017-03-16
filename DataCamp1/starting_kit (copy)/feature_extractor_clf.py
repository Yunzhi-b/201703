import numpy as np
# import pandas as pd


class FeatureExtractorClf():
    	def __init__(self):
        	pass

    	def fit(self, X_df, y):
        	pass


	def transform(self, X_df):
		XX = np.array([np.array(dd) for dd in X_df['spectra']])
		return XX
"""
    	def transform(self, X_df):
        	XX = np.array([np.array(dd) for dd in X_df['spectra']])
		n,p = XX.shape
		a = []
		for i in np.arange(p-1)+1:
			a.append(XX[:,i]-XX[:,i-1])
		XX = np.concatenate((XX,np.array(a).T),axis=1)
        	return XX
"""	

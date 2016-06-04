from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from sklearn.preprocessing import binarize
import numpy as np
    
class NBmatrix(BaseEstimator, TransformerMixin):
   
    
    def __init__(self, alpha, bina, n_jobs = 1):
        self.alpha = alpha
        self.bina = bina
        self.n_jobs = n_jobs
        self.r = []

    def fit(self, X, y):
        alpha = self.alpha
        nb_doc, voc_length = X.shape
        pos_idx = [y==1][0].astype(int)
        neg_idx = [y==0][0].astype(int)
        #Store the indicator vectors in sparse format to accelerate the computations
        pos_idx = sp.csr_matrix(pos_idx.T)
        neg_idx = sp.csr_matrix(neg_idx.T)
        #Use sparse format dot product to get a weightning vector stored in sparse format
        alpha_vec = sp.csr_matrix(alpha*np.ones(voc_length))
        p = (alpha_vec + pos_idx.dot(X)) 
        norm_p = p.sum()
        p = p.multiply(1/norm_p)
        #print p.toarray()
        q = (alpha_vec + neg_idx.dot(X))
        norm_q = q.sum()
        q = q.multiply(1/norm_q)
        #print q.toarray()
        
        ratio = sp.csr_matrix(np.log((p.multiply(sp.csr_matrix(np.expand_dims(q.toarray()[0]**(-1),axis=0)))).data))
        #print ratio.toarray()
        self.r = ratio #Stock the ratio vector to re-use it for transforming unlablled data
        return self

    def transform(self, X):
        #If the binarize option is set to true, we need now to recompute "f", our binarized word counter
        if(self.bina == True):
            f_hat = binarize(X, threshold = 0.0)
        else :
            f_hat = X
        
        f_tilde = f_hat.multiply(self.r)
        return f_tilde
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)


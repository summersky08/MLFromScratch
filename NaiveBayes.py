import numpy as np
from numpy.linalg import det,inv

def normalND_pdf(x, x_mean=None, Sigma=None):
    '''
    # 行列演算をするので、多数のxをまとめて入力するのには対応できない。
    >>>normalND_pdf([.5, .1, .2], [0, 0, 0], np.eye(3))
    0.05464947890082988
    '''
    x = np.array(x)
    n_dim = len(x)
    if np.all(x_mean == None):
        x_mean = np.zeros(n_dim)
    else:
        x_mean = np.array(x_mean)
    if np.all(np.array(Sigma) == None):
        Sigma = np.eye(n_dim)
    else:
        Sigma = np.array(Sigma)
    term1 = np.sqrt((2*np.pi)**n_dim*det(Sigma))
    term2 = -.5*(x - x_mean).T @ inv(Sigma) @ (x - x_mean)
    return 1/term1*np.e**term2

class NaiveBayes:
    def __init__(self,pdf=normalND_pdf):
        self.pdf = pdf
        
    def fit(self,X,y):
        self.data = {}
        for x,y_ in zip(X,y):
            if y_ in self.data.keys(): self.data[y_].append(x)
            else: self.data[y_] = [x]
        self.stats = {}
        for k in self.data.keys():
            self.data[k] = np.array(self.data[k])
            self.stats[k] = self.data[k].mean(0),self.data[k].std(0)
            
    def predict(self,X):
        return self.predict_proba(X).argmax(1)
    
    def predict_proba(self,X):
        res = np.zeros((X.shape[0],len(self.stats)))
        for i,x in enumerate(X):
            for k,(mean,std) in self.stats.items():
                res[i,k] = normalND_pdf(x,mean,np.diag(std))
        return res
import numpy as np
from scipy import linalg
import cvxopt
from scipy import stats
from itertools import combinations

def qp(H, e, A, b, C=np.inf, l=1e-8, verbose=True):
    # Gram matrix
    n = H.shape[0]
    H = cvxopt.matrix(H)
    A = cvxopt.matrix(A, (1, n),'d')
    e = cvxopt.matrix(-e)
    b = cvxopt.matrix(0.0)
    if C == np.inf:
        G = cvxopt.matrix(np.diag(np.ones(n) * -1))
        h = cvxopt.matrix(np.zeros(n))
    else:
        G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1),
                                         np.diag(np.ones(n))], axis=0))
        h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(H, e, G, h, A, b)
 
    # Lagrange multipliers
    mu = np.ravel(solution['x'])
    return mu

def svm_solver(K, y, C=np.inf):
    n = y.shape[0]
    H = y[:,None]*K*y[None,:]  # GG.T = yi*yj*np.dot(xi,xj)
    e = np.ones(n) 
    A = y
    b = 0.
    mu = qp(H, e, A, b, C, l=1e-8, verbose=False)
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support


def compute_b(K, y, mu_support, idx_support):
    y_support = y[idx_support]
    K_support = K[idx_support][:, idx_support]
    
    g = np.dot(y_support*mu_support, (K[idx_support][:, idx_support])[:,0])
    b = 1./y_support[0] - g
    return b

class SVMClassifier():
    def __init__(self, C=None, kernel=None, kernel_coef=None, param=None):
        self.C = C
        self.kernel = kernel
        self.param = param
        self.kernel_coef =kernel_coef

    def fit(self, X, y):
        self.X = X
        self.y = y
        K_tmp = [tmp(self.X,self.X) for tmp in self.kernel]
        K = 0
        for i in range(len(K_tmp)):
            K = K + K_tmp[i]*self.kernel_coef[i]
        self.mu_support, self.idx_support = svm_solver(K, self.y, self.C)
        self.b = compute_b(K, self.y, self.mu_support, self.idx_support)
        self.w = np.sum((self.mu_support * self.y[self.idx_support])[: , None] * self.X[self.idx_support], axis=0)
        self.X_support = self.X[self.idx_support]
        return self

    def predict(self, X_test):
        K_tmp = [tmp(X_test,self.X_support) for tmp in self.kernel]
        G = 0
        for i in range(len(K_tmp)):
            G = G + K_tmp[i]*self.kernel_coef[i]
        # Calcul de la fonction de décision
        decision = G.dot(self.mu_support * self.y[self.idx_support]) + self.b

        # Calcul du label prédit
        y_pred = np.sign(decision)
        return y_pred

    def predict_prob(self, X_test):
        K_tmp = [tmp(X_test,self.X_support) for tmp in self.kernel]
        G = 0
        for i in range(len(K_tmp)):
            G = G + K_tmp[i]*self.kernel_coef[i]
        # Calcul de la fonction de décision
        decision = G.dot(self.mu_support * self.y[self.idx_support]) + self.b

        # Calcul du label prédit
        y_pred_prob = 1./(1.+np.exp(-decision))
        return y_pred_prob


    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class KernelMultiClassifier_OneVsOne():
    def __init__(self, baseClassifier=None,C=None, kernel=None, kernel_coef=None, param=None):
        self.C = C
        self.kernel = kernel
        self.kernel_coef =kernel_coef
        self.param = param
        self.baseClassifier = baseClassifier

    def fit(self, X, y):
        datasets_dict = {}
        models_dict = {}
        l = np.unique(y)
        for (class1,class2) in combinations(l, 2):
            y1 = y[y==class1]
            y1[:] = -1
            X1 = X[y==class1]
            y2 = y[y==class2]
            y2[:] = 1
            X2 = X[y==class2]
            datasets_dict[(class1,class2)] = (np.concatenate((X1,X2)),np.concatenate((y1,y2)))
        for tmp in datasets_dict:
            (X_tmp,y_tmp) = datasets_dict[tmp]
            model = self.baseClassifier(self.C,self.kernel,self.kernel_coef)
            model = model.fit(X_tmp, y_tmp)
            models_dict[tmp] = model
        self.models_dict = models_dict
        return self


    def predict(self, X):
        y = []
        prob_dict = {}
        n = X.shape[0]
        for tmp in self.models_dict:
            y_pred_prob = self.models_dict[tmp].predict_prob(X)
            if tmp[1] not in prob_dict:
                prob_dict[tmp[1]] = y_pred_prob
            else:
                prob_dict[tmp[1]] = prob_dict[tmp[1]]*y_pred_prob
                
            if tmp[0] not in prob_dict:
                prob_dict[tmp[0]] = 1 - y_pred_prob
            else:
                prob_dict[tmp[0]] = prob_dict[tmp[0]]*(1-y_pred_prob)
        
        labels = []
        probs = []
        for label in prob_dict:
            labels.append(label)
            probs.append(prob_dict[label])
        labels = np.array(labels)
        probs = np.array(probs)
        labels_index = np.argmax(probs,axis=0)
        return labels[labels_index]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

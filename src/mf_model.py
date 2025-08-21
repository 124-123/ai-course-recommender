import numpy as np, joblib, os
from tqdm import trange
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
MF_MODEL_PATH = os.path.join(MODELS_DIR, 'mf_model.joblib')

class MF:
    def __init__(self, n_users, n_items, n_factors=32, lr=0.01, reg=0.02, n_epochs=30, random_state=42):
        self.n_users = n_users; self.n_items = n_items; self.n_factors = n_factors
        self.lr = lr; self.reg = reg; self.n_epochs = n_epochs; self.random_state = random_state
        rng = np.random.RandomState(random_state)
        self.P = 0.01 * rng.randn(n_users, n_factors)
        self.Q = 0.01 * rng.randn(n_items, n_factors)
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.global_mean = 0.0

    def fit(self, interactions):
        if len(interactions) == 0: return
        self.global_mean = np.mean([r for (_,_,r) in interactions])
        for epoch in trange(self.n_epochs, desc="MF epochs"):
            np.random.shuffle(interactions)
            for (u,i,r) in interactions:
                pred = self.predict_single(u,i); e = r - pred
                self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                Pu = self.P[u,:].copy(); Qi = self.Q[i,:].copy()
                self.P[u,:] += self.lr * (e * Qi - self.reg * Pu)
                self.Q[i,:] += self.lr * (e * Pu - self.reg * Qi)

    def predict_single(self,u,i):
        return self.global_mean + self.bu[u] + self.bi[i] + self.P[u,:].dot(self.Q[i,:])

    def predict_user(self,u):
        return self.global_mean + self.bu[u] + self.bi + self.P[u,:].dot(self.Q.T)

    def save(self,path=MF_MODEL_PATH): joblib.dump(self,path)
    @staticmethod
    def load(path=MF_MODEL_PATH): return joblib.load(path)

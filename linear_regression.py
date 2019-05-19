import numpy as np
import matplotlib.pyplot as plt
import pickle

def read_t(t,root="/home/samknu/MyRepos/MLProjectIsingModel/data/IsingData/"):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)

temperatures = np.arange(0.25, 4., step=0.25)

test = read_t(temperatures[0])
dings = np.einsum('bi,bo->bio',test,test)

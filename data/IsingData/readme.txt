Ok this one I checked using Onsager's magnetization.
It is also resampled to take into account Z2 symmetry !
Samples are decorrelated and correctly distributed.

To read data, use following function:

import pickle
def read_t(t,root="./"):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)



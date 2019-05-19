import numpy as np
import pickle
from matplotlib import pyplot as plt

def read_t(t=0.25,root="./"):
    if t > 0.:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    else:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)

stack = []
for i,t in enumerate(np.arange(0.25,4.01,0.25)):
    y = np.ones(10000,dtype=int)
    if t > 2.25:
        y*=0
    stack.append(y)

pickle.dump(np.vstack(y),open('labels_all.pkl','wb'))

#X = read_t(-1)
''' for i in range(16):
    print(np.arange(0.25,4.01,0.25)[i])
    x = X[i*10000].reshape(40,40)
    plt.imshow(x)
    plt.show() '''


''' stack = []
for t in np.arange(0.25,4.01,0.25):
    stack.append(read_t(t))

X = np.vstack(stack)
pickle.dump(np.packbits(X), open('Ising2DFM_reSample_L40_T=All.pkl','wb')) '''






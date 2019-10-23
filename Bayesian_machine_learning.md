
# Bayesian Machine Learning
By Samuel Knudsen 

Contact: 
<br>
samuelknudsen@hotmail.com
<br>
github.com/kodemannen


### Overview

This project is about the Bayesian approach to machine learning. More specifically we go through the Bayesian formulation of regression and neural networks for classification. We apply these methods to data from simulations of the 1 and 2 dimensional Ising models in similar style as Mehta et al. [4].

This text is split up into 4 parts.

- Part 1 - General intro to Bayesian statistics
- Part 2 - Bayesian regression
- Part 3 - Bayesian Convolutional Neural Network
- Part 4 - Using Bayesian reasoning on the probability of life

The Python code for part 2 is written from scratch in raw NumPy, while the code for part 3 uses the TensorFlow Probability library.

# Part 1 - Bayesian Statistics

Bayesian statistics is an alternative approach to the more common classical (frequentist) school of thought in statistics. It differs in the way that it views probability as our subjective uncertainty about the world, instead of as there being something inherently random in nature.


The most notable difference in Bayesian statistics is the use of a __prior__, which is a way of incorporating the prior knowledge one often has about the problem into the inference. In Bayesian machine learning, using various priors are in many cases mathematically equivalent to specific regularization schemes that one often sees in classical machine learning.


A very useful property of Bayesian inference is that we don't just get point estimates of our model parameters, but we will instead get a full distribution of our probability estimate in parameter space. This means that we can get knowledge about how points in the neighbourhood of our best estimate compares. This has the useful property of letting us define __credible intervals__, which we will see in part 2, but it can in addition be used to do probabilistic estimation, which we will see in part 3. The Bayesian approach also provide solutions for some of the inherent pathologies that exist in classical statistics -- so it can for example do inference from a one-off event, which we will see in part 4.

### Derivation of Bayes Theorem

Everything starts with Bayes theorem.

We have two parameters $A$ and $B$. 
For any two values we have $p(A,B)$ as the probability that both of those values are the true values of A and B. 


We start with the intuitive statement $$p(A,B) = p(A|B)p(B).$$

Where $p(A|B)$ means "the probability of A being true given that B is true". The above equation can then ber understood intuitively as evaluating the truth estimate of A and B sequentially.


Then since $p(A,B) = p(B,A)$ it must follow that

$$p(A|B)p(B) = p(B|A)p(A),$$

which leads to Bayes theorem

$${p(A|B) = \frac{p(B|A)p(A)}{p(B)}},$$


Usually written as 
    

$$\boxed{p(A|B)  \propto p(B|A)p(A)}$$
    
$p(B)$ is as a normalization constant making sure that $\int_A p(A'|B)dA' = 1$

### Bayesian Inference

Say we have a dataset $D = \{d_1, d_2, .., d_N\}$ that are measurements of value $y$ that is a function of a parameter vector $\vec{x}$. In other words $d_i = y(\vec{x}_i | \boldsymbol{\theta})$.

$D$ and $X=[\vec{x}_1, \vec{x}_2, .., \vec{x}_N ]^T$ are known, and we want to find the function $y$, meaning we need to find its parameters $\boldsymbol{\theta}$ (if the shape/form of $y$ is assumed, otherwise we'd need to find the shape as well). 

Any parameter configuration $\boldsymbol{\theta}$ is a unique hypothesis for the model.
For any given $\boldsymbol{\theta}$, we want to know the probability of that hypothesis being true from the data, described as

$$
p(\boldsymbol{\theta}|D).
$$

We can then use Bayes theorem to get
$$ 
\boxed{
p(\boldsymbol{\theta}|D)  \propto {p(D|\boldsymbol{\theta})p(\boldsymbol{\theta})}
}.$$

The factor $p(D|\boldsymbol{\theta})$ is called the __likelihood function__ and describes the probability of getting the data $D$ if the given hypothesis $\boldsymbol{\theta}$ is true. The factor $p(\boldsymbol{\theta})$ is called the __prior distribution__  for the hypothesis, meaning the probability distribution for various hypotheses $\boldsymbol{\theta}$ being true prior to seeing the data. If we have the likelihood and the prior, then we can create $p(\boldsymbol{\theta}|D)$ which is known as the __posterior distribution__.


### Comparison to classical inference

With classical statistical inference, one is only interested in the value for $\boldsymbol{\theta}$ that maximizes the probability of getting the obtained data, i.e.

$$
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\text{argmax}} p(D|\boldsymbol{\theta})
$$

$\hat{\boldsymbol{\theta}}$ is known as the MLE (maximum likelihood estimate).  But this is just a point estimate and gives no information about the robustness of the estimate, i.e. how much the probability changes by moving to other points that are close to $\hat{\boldsymbol{\theta}}$ in parameter space.

This is something we can get with Bayesian inference and herein lies much of its power. 

# Part 2 - Bayesian Regression on the 1D Ising model (with Noise)
### The 1D Ising Model (with noise)

We randomly generate a set of $N$ states $\{\vec{x}^i\}$ of the 1D ising model (meaning N 1D vectors consisting of -1s and 1s) and calculate their energies using the following Hamiltonian:

$$
H[\vec{\vec{x}^i}] = \sum_{j=1}^L\sum_{k=1}^L J_{jk}\vec{x}_j^i\vec{x}_{k}^i + \epsilon
$$

Where $J_{jk} =-0.5 \cdot \max \big(\delta_{j,k-1}, \delta_{j,k+1}\big)$.
In other words, each element only interacts with its neighbour.

All values of the matrix $J$ will be zero except form the elements on the two diagonal next to the main diagonal which will have value -0.5 (plotted below).


$\vec{x}_j^i$ is the j'th element of the i'th state $\vec{\vec{x}^i}$. The max energy is 40 so $\epsilon \sim \mathcal{N}(0,2.5)$ seems like a good choice.

We will then try to see if we can re-extract this Hamiltonian from the data using Bayesian Linear regression.


```python
import numpy as np
import scipy.sparse as sp
np.random.seed(13)

import warnings
# Comment this to turn on warnings
warnings.filterwarnings('ignore')

### define Ising model aprams
# system size
L=40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(1400,L))

def ising_energies(states_, plot_true=False):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L): 
        J[i,(i+1)%L]=-0.5 # interaction between nearest-neighbors
        J[(i+1)%L,i]=-0.5
    # compute energies
    E = np.einsum('...i,ij,...j->...',states_,J,states_)
    
    if plot_true:
        import matplotlib.pyplot as plt
        %matplotlib inline
        import seaborn as sns

        sns.heatmap(J)
        plt.title("Parameters of the True Hamiltonian")
        plt.show()
    return E

# calculate Ising energies
energies=ising_energies(states,plot_true=True)

# Adding noise:
noise_variance = 2.5
energies += np.random.normal(0,scale=np.sqrt(noise_variance), size=energies.shape)


```


![png](Bayesian_machine_learning_files/Bayesian_machine_learning_7_0.png)


### Remapping data for regression

We pretend that we're ignorant about the Hamiltonian used to generate the above data. That means that the values aren't the only unknowns, but the shape of it as well. So we need to consider the __all-to-all Hamiltonian__

$$
H_{model}[\vec{\vec{x}^i}] = \sum_{j=1}^L\sum_{k=1}^L J_{jk}\vec{x}_j^i\vec{x}_{k}^i + \epsilon
$$

We see that the actual Hamiltonian we used above is just a special case of this, with $J_{jk} =-0.5 \cdot \max \big(\delta_{j,k-1}, \delta_{j,k+1}\big)$.



Taking the outer product

$\vec{{x}} \rightarrow \phi(\vec{{x}})=\vec{{x}}\otimes \vec{{x}}$

then we make the vector $\phi(\vec{x})$ one-dimensional.
But we'll just write $\phi(\vec{x})$ as $\vec{x}$ for simplicity.


```python
new_states = np.einsum('bi,bo->bio',states,states)
new_states = new_states.reshape(new_states.shape[0],-1)
```

## Performing the regression

Bayesian regression is just a special type of Bayesian inference.
The goal is to create the posterior from the likelihood and the prior using

$$ p(\boldsymbol{\theta}|D)  \propto {p(D|\boldsymbol{\theta})p(\boldsymbol{\theta})}.$$

We thus need to specify the likelihood and the prior. How this is done is of course problem dependent.

#### Choosing the Likelihood
It is common to make the assumption that the data is __iid__ (identically and independently distributed), which it is in our case.

The likelihood can then be modelled as 
$$
p(D|\boldsymbol{\theta}) = p(d_1|\boldsymbol{\theta})p(d_2|\boldsymbol{\theta})..p(d_N|\boldsymbol{\theta})
$$
where 
$$
\begin{align}
p(d_i|\boldsymbol{\theta}) & = \mathcal{N}(\vec{w}^T\vec{x}_i, \sigma^2) \\ 
                           & \propto \exp \Big[-\dfrac{1}{2\sigma^2} (d_i-\vec{w}^T\vec{x}_i)^2\Big]
\end{align}
$$

and $\boldsymbol{\theta} = \{\vec{w}, \sigma^2\}$. 
The product $\vec{w}^T \vec{x}$ is just a weighted sum of the input parameters.
The Gaussian is commonly used because this is the probability distribution with the highest entropy for iids. In other words, if the data is iid, the Gaussian is the _most probable way for the data to be distributed_. Here we assume that the noise variation $\sigma^2$ does not change with $\vec{x}$ (which is not always a correct assumption, but it is in this case).


The full likelihood is then
$$
\begin{align}
p(D|\boldsymbol{\theta}) &\propto \exp \Big[-\sum_i^N \dfrac{1}{\sigma^2} (d_i-\vec{w}^T\vec{x}_i)^2\Big]\\
& = \exp \Big[ - \dfrac{1}{2\sigma^2}(\vec{y}-X\vec{w})^T(\vec{y}-X\vec{w}) \Big].
\end{align}
$$



#### Choosing the Prior
We need to decide a shape for our prior 
$$
p(\boldsymbol{\theta}) = p(\vec{w},\sigma^2).
$$


We will assume that the noise variance $\sigma^2$ is known, since we could in practice measure it by holding $\vec{x}$ constant and measure it multiple times.

Our prior to find is therefore just 
$$
p(\boldsymbol{\theta}) = p(\vec{w}).$$


A common choice is the zero mean Gaussian. 
This gives a higher prior probaility to functions with small, even parameters, i.e. smoother / less complex functions. 
This in a way captures the idea of Occam's Razor that we should prefer the simplest hypothesis that explains the data (although other zero zentered, symmetric distributions would capture this this as well).

It also makes it easier mathematically to pick a Gaussian when the likelihood is Gaussian as well (called conjugate prior). Therefore

$$
\begin{align}
p(\vec{w}) &= \mathcal{N}(\vec{w} | \vec{w}_0, V_0)\\
& \propto \exp \Big[ - \frac{1}{2}(\vec{w}- \vec{w}_0)^T V_0^{-1} (\vec{w}- \vec{w}_0) \Big].
\end{align}
$$

#### The Posterior
Now that we have the likelihood and prior, we can find the posterior as

$$
\begin{align}
p(\vec{w}|D) & \propto {p(D|\vec{w})p(\vec{w})} \\
             & \propto \exp \Big[  -\dfrac{1}{2\sigma^2}(\vec{y}-X\vec{w})^T(\vec{y}-X\vec{w}) - \frac{1}{2}(\vec{w}- \vec{w}_0)^T V_0^{-1} (\vec{w}- \vec{w}_0) \Big]
\end{align}
$$

By doing some algebra this can be rewritten as a multivariate normal distribution (MVN)


$$
\boxed{
\begin{align}
p(\vec{w}|D) = \mathcal{N}(\vec{w}|\vec{w}_N, V_N)
\end{align}},
$$
where
$$
\boxed{
\begin{align}
\vec{w}_N &= V_N V_0^{-1} + \frac{1}{\sigma^2}V_N X^T \vec{y}, \\
V_N^{-1}  &= V_0^{-1} + \frac{1}{\sigma^2}X^TX,\\
V_N       &= \sigma^2(\sigma^2V_0^{-1} + X^T X)^{-1} 
\end{align}}.
$$




#### The Special Case when $\vec{w}_0=\vec{0}$ and $V_0 = \tau^2I$
The prior is then
$$
\begin{align}
p(\vec{w}) &= \prod_j^M \mathcal{N}(w_j | 0, \tau^2)\\
& \propto \exp \Big[- \frac{1}{2\tau^2}\sum_j^M {w_j^2} \Big]
\end{align}
$$
where $1/\tau^2$ controls the strength of the prior.


We now have
$$
\begin{align}
p(\vec{w}|D) & \propto {p(D|\vec{w})p(\vec{w})} \\
                         & \propto \exp \Big[- \Big( \sum_i^N \dfrac{1}{\sigma^2} (d_i-\vec{w}^T\vec{x}_i)^2 +\sum_j^M w_j^2 / \tau^2\Big) \Big]
\end{align}
$$
The MAP estimate is the value of $\vec{w}$ that maximizes $p(\vec{w}|D)$, which means the value that minimizes the exponent, i.e.

$$
\begin{align}
\vec{w}_{MAP} & = \underset{\vec{w}}{\text{argmin}} \sum_i^N \dfrac{1}{\sigma^2} (d_i-\vec{w}^T\vec{x}_i)^2 +\sum_j^M w_j^2 / \tau^2 \\
\end{align}
$$

where $\vec{y}$ is the vector containing the data $D$. We can see that this is equivalent to regular regression with L2 regularization.
This has an analytical solution, which we can find by rewriting to matrix formulation

$$
\vec{w}_{MAP} = \underset{\vec{w}}{\text{argmin}} \ (\vec{y}-X\vec{w})^T(\vec{y}-X\vec{w}) + \lambda \vec{w}^T\vec{w}
$$

and we can then differentiate the right side with respect to $\vec{w}$ and set equal to zero to find the solution as

$$
\boxed{\vec{w}_{MAP} = (\lambda I_M + {X}^T{X})^{-1}{X}^T\vec{y}}
$$

which is equivalent to ridge regression in classical statistics.

### The Code

Reminder: $\sigma^2$ is assumed to be known. We also have no prior reason to believe that the elements in $\vec{w}$ are correlated, nor have we any prior reason to know whether they are positive or negative, so the special case of $\vec{w}_0=\vec{0}$ and $V_0 = \tau^2I$ seems to apply here.


```python
import time
from sys import exit
t0 = time.time()


n = new_states.shape[0]   # number of data
D = new_states.shape[1]   # data dimension

# Prior:
variance = 2.5
w0 = np.zeros(D)
tau = 1 # 1 means unitary gaussian, determines the strength of the prior
V0 = tau**2*np.identity(D)  # precision matrix of prior
V0_inv = np.linalg.inv(V0)

mean_x = np.mean(new_states,axis=0,keepdims=True)

X = new_states #- mean_x # data matrix with data as rows, centered


y = energies - np.mean(energies)


VN_inv = V0_inv + np.dot(X.T,X) / variance
VN = np.linalg.inv(VN_inv)

wN = np.dot(np.dot(VN,V0_inv),w0) + np.dot(np.dot(VN,X.T),y) / variance
t1 = time.time()-t0

```

### Plotting $\vec{w}_{MAP}$


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

sns.heatmap(wN.reshape(L,L))
plt.title("Estimated Hamiltonian")
plt.show()
```


![png](Bayesian_machine_learning_files/Bayesian_machine_learning_18_0.png)


### The Posterior Distribution

Since we now have the full posterior $P(\vec{w}|D)$, we can see how the probability changes as we move in parameter space away from the MAP estimate, i.e. how confident we would be in points near $\vec{w}_{MAP}$. We only show the posterior for four of the parameters.



```python
dw = 0.001
w_range = np.arange(-1.,1., dw)

#print(w_range)
def Pw(index1,index2):
    
    index = index1*L + index2
    vec = wN.copy()
    
    logs = np.zeros(len(w_range))
    for k in range(len(w_range)):
        w = w_range[k]
        vec[index] = w
        logs[k] = -0.5 * np.dot(np.dot((vec - wN).T, VN_inv),vec - wN)
    
    logs -= np.max(logs)
    P = np.exp(logs)
    return P 

def plot_w_distribution(ax, index1,index2,show=False,grid=True):
    P = Pw(index1,index2)
    ax.plot(w_range,P, label="$P(w_{%.i,%.i}|D)$" % (index1,index2))
    ax.legend()
    ax.grid() if grid else None
    if show:
        plt.show()

fig, axes = plt.subplots(2,2,sharex=False, sharey=True)
fig.set_size_inches(18.5*0.75, 10.5*0.7)
plot_w_distribution(axes[0,0], 0,0)
plot_w_distribution(axes[0,1],0,1)
plot_w_distribution(axes[1,0],1,0)
plot_w_distribution(axes[1,1],1,1)
plt.show()

```


![png](Bayesian_machine_learning_files/Bayesian_machine_learning_20_0.png)


### Credible Intervals

We will show the 95 % HDI (Highest Density Interval) which means the region that contains 95 % of the probability mass where all points in the region are higher than the ones outside. 
This area is not necessarily contiguous if the PDF is multimodal. Since the posterior here is gaussian, the HDI is the same as the Central Interval.

The algorithm used to find the HDI region can be easily derived by thinking of it as turning the curve upside down and filling it with water drop by drop.


```python
def credible_interval(ax, index1, index2):
    P_ = Pw(index1,index2)
    # normalize
    P_normed = P_ / np.sum(P_)
    
    ############################
    # Water filling algorithm: #
    ############################
    #points = np.zeros_like(P_normed, dtype=np.int)
    points_taken= []
    points = []
    done = False
    t = 0
    while not done:
        best=0
        bestindex=0
        for i in range(len(P_normed)-1):
            if i not in points_taken:
                val = P_normed[i]
                if val > best:
                    best = val
                    bestindex = i
        points_taken.append(bestindex)
        points.append(best)
        if np.sum(points) >= 0.95:
            done=True
    
    points_taken = np.array(points_taken, dtype=np.int)
    argsorted = np.argsort(points_taken)

    points_taken = points_taken[argsorted]
    
    
    plot_w_distribution(ax, index1,index2,show=False,grid=False)
    first_lastw = [w_range[points_taken[0]], w_range[points_taken[-1]]]
    first_lastP = [P_[points_taken[0]], P_[points_taken[-1]]]

    
    fill = np.zeros(len(points_taken)+2)
    fill[1:-1] = P_[points_taken]
    
    w_range_fill = np.zeros_like(fill)
    w_range_fill[1:-1] = w_range[points_taken]
    w_range_fill[0] = w_range_fill[1]
    w_range_fill[-1] = w_range_fill[-2]
    
    ax.fill(w_range_fill,fill,facecolor="red",alpha=0.5)
    
    line = [P_[points_taken[0]],P_[points_taken[-1]]] 
    line = np.ones(2)*P_[points_taken[0]] # looks better, but not actually totally correct
    ax.plot(first_lastw,line, "k", alpha=0.5)

    
fig, axes = plt.subplots(2,2,sharex=False, sharey=True)
fig.set_size_inches(18.5*0.75, 10.5*0.75)
credible_interval(axes[0,0], 0,0)
credible_interval(axes[0,1],0,1)
credible_interval(axes[1,0],1,0)
credible_interval(axes[1,1],1,1)
plt.suptitle("95 % Credible Interval")
plt.show()

```


![png](Bayesian_machine_learning_files/Bayesian_machine_learning_22_0.png)


### Test data

We can evaluate the performance by calculationg the __coefficient of determination__, given by

$$
\begin{align}
R^2 &=  \big(1-\frac{u}{v}\big),\\
u &= \big(y_{\text{predicted}} - y_{\text{true}}\big)^2 \\
v &= \big(y_{\text{predicted}} - \langle y_{\text{true}}\rangle\big)^2
\end{align}
$$

The best possible score is then $R^2=1$, but it can also be negative. A constant model that always predicts the expected value of $y$, $ \langle y_{\text{true}}\rangle$, disregarding the input features, would get a $R^2$ score of 0 [4].


```python
test_states=np.random.choice([-1, 1], size=(1000,L))
# calculate Ising test energies
test_energies=ising_energies(test_states)

# remapping states:
test_states = np.einsum('bi,bo->bio',test_states,test_states)
test_states = test_states.reshape(test_states.shape[0],-1)

predicted_energies = np.dot(test_states, wN)


### R^2 - coefficient of determination
y_true_avg = np.mean(test_energies)
residuals = predicted_energies - test_energies
u = np.dot(residuals,residuals)
v = test_energies - y_true_avg
v = np.dot(v,v)

R_squared = 1 - u/v

print(R_squared)

```

    0.92679016596241


# Part 3 - Bayesian Neural Networks
### The Essence
A general classical neural network is a function on the form

\begin{equation}
    \mathcal{F}(\vec{x}) = g_L(\boldsymbol{W}^L \ g_{L-1}(\boldsymbol{W}^{L-1} \dots g_1(\boldsymbol{W}^1\vec{x}) \dots  )),
\end{equation}

where the weights are elements in the matrices $\boldsymbol{W}^l$ and $g_l: \mathbb{R}^k \rightarrow \mathbb{R}^k$ are activation functions. For any arbitrary architecture we denote the total number of parameters in such a network by $N$.

In a Bayesian Neural Network with the same architecture, the number of parameters is instead $2N$. Instead of the parameters (weights) being a point estimate, each weight $w_{ij}$ in the classical neural net is instead switched out with two parameters $\mu_{ij}$ and $\sigma_{ij}$ which are the mean and standard deviation in a normal distribution. When we do a forward pass and need the weight, we just sample it from this distribution, i.e.

$$
w_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)
$$

where the trainable parameters are now $\theta = \{ \mu, \sigma^2 \}$


The full posterior is in other words not just useful for finding credible intervals, but for sampling! It gives us only twice the number of parameters for in principle an infinite ensemble of networks.

The figure below is an illustration of the difference between a frequentist and a Bayesian CNN. 

![bayescnn.png](attachment:bayescnn.png)
Source: Shridhar et al. [1].

### The Math 
We want to find the posterior
$$
p(\theta|D) = \frac{p(D|w)p(w)}{p(D)}
$$

so that we can use this to do inference 
$$
p(y^i|x^i,D) = \int p(y^i|x^i,w)p(w|D)d\theta.
$$

This can be understood as checking how much we believe datapoint $x^i$ belongs to class $y^i$ for all possible hypothesises $\theta$ while weighing for how much we believe in each $\theta$, based on the data. $D = \{d_i\}$ are the class labels for $X=\{x_i\}$.


Let our Bayesian Neural Network model $q_\theta$ be a multivariate Gaussian approximation to the true posterior $p$.

$$
q_\theta(w|D) \approx p(w|D)
$$

and then minimize the Kullback-Leibler (KL) divergence between the two distributions


$$
\theta_{opt} = \underset{\theta}{\text{argmin}} \ \text{KL} \ \big[q_\theta(w|D)||p(w|D)\big].
$$

The KL divergence is a measure of how close two distributions are to each other, and is defined as

$$
\text{KL} \ \big[q_\theta(w|D)||p(w|D)\big] = \int q_\theta(w|D) \log \frac{q_\theta(w|D)}{p(w|D)}dw.
$$

This can also be seen as the expectation value of $\log \dfrac{q_\theta(w|D)}{p(w|D)}$ with respect to $q_\theta(w|D)$, i.e.

$$
\text{KL} \ \big[q_\theta(w|D)||p(w|D)\big] = \mathbb{E}_{q_\theta(w|D)}\big[\log \frac{q_\theta(w|D)}{p(w|D)}\big].
$$

The right side can be approximated as a discrete sum 

$$
\mathbb{E}_{q_\theta(w|D)}\big[\log \frac{q_\theta(w|D)}{p(w|D)}\big] \approx \frac{1}{m}\sum_i^m \log \frac{q_\theta(w^i|D)}{p(w^{i}|D)}.
$$

We then substitute $p(w^i|D) = \dfrac{p(D|w^i)p(w^i)}{p(D)}$ and use the rule for the logarithm of fractions, i.e. $\log \frac{a}{b} = \log a - \log b$, so that we get

$$
\mathbb{E}_{q_\theta(w|D)}\big[\log \frac{q_\theta(w|D)}{p(w|D)}\big] \approx \frac{1}{m}\sum_i^m \log {q_\theta(w^i|D)} - \log p(w^i) - \log p(D|w^i)   + \log p(D)
$$

This is a tractable objective function that can be minimized with respect to $\theta = (\mu, \sigma^2)$ by variational methods, Monte Carlo, evolutionary algorithms or other optimizing schemes. The term $\log p(D)$ is just a constant, so we can remove that. 

The optimum can now be found as

\begin{equation}
\boxed{
\begin{align}
\theta_{opt} & = \underset{\theta}{\text{argmin}} \frac{1}{m}\sum_i^m \log {q_\theta(w^i|D)} - \log p(w^i) - \log p(D|w^i) \\
& = \underset{\theta}{\text{argmin}} \ \text{KL} \ \big[q_\theta(w|D)||p(w)\big] - \mathbb{E}_{q_\theta(w|D)}\big[\log p(D|w^i)\big]
\end{align}
}
\end{equation}


by sampling $w^i$ from $q_\theta(w|D)$. This is also known as the __evidence lower bound__ (ELBO).

NB!
For simplicity we have skipped the conditioning on the input data $X$.
So e.g. $\log P(D|w^i)$ should really be understood as $\log P(D|w^i, X)$, meaning that it looks at the probability of setting the correct labels for the data $X$.

### Training a Bayesian Convolutional Neural Net on the 2D Ising model

The 2D Ising model undergoes a phase transition around a critical temperature $T_c$ where it switches from the disordered to the ordered state. The idea is to train a classifier on states that we know are ordered or disordered, and then after use that classifier on states from the transition state / critical region.

![states.png](attachment:states.png)
Source: Mehta et al. [4] 

The following code trains a Bayesian Neural Network to classify states of the 2 dimensional Ising model by minimizing the ELBO, using a minimizing scheme called Flipout [1]. The architecture is LeNet-5 [3]. 

It is written using TensorFlow Probability, a library built on TensorFlow for doing probabilistic machine learning. The script is built on the code at the following Github repository https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py, modified for using the Ising model data.


```python
"""Trains a Bayesian neural network to classify data from the 2D Ising model.
The architecture is LeNet-5 [1].
#### References
[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     Gradient-based learning applied to document recognition.
     _Proceedings of the IEEE_, 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.contrib.learn.python.learn.datasets import mnist



# TODO(b/78137893): Integration tests currently fail with seaborn imports.
warnings.simplefilter(action="ignore")

try:
    import seaborn as sns  # pylint: disable=g-import-not-at-top
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

tfd = tfp.distributions

ISING = True

IMAGE_SHAPE = [40,40,1] if ISING else [28, 28, 1] 

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=6000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=400,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.
    Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(qm.flatten(), ax=ax, label=n)
    ax.set_title("weight means")
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(qs.flatten(), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, 1.])

    fig.tight_layout()
    
    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, label_vals, probs,
                            fname, n=10, title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
    Args:
    input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    indices = np.random.randint(low=0,high=input_vals.shape[0],size=n)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[indices[i], :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(2) if ISING else np.arange(10), prob_sample[indices[i], :], alpha=1/FLAGS.num_monte_carlo, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(2) if ISING else np.arange(10), np.mean(probs[:, indices[i], :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs, correct=%.i" % label_vals[indices[i]] )
        
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))



def plot_test_prediction(input_vals, probs,
                            fname, n=10, title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
    Args:
    input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    indices = np.random.randint(low=0,high=input_vals.shape[0],size=n)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[indices[i], :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(2) if ISING else np.arange(10), prob_sample[indices[i], :], alpha=1/FLAGS.num_monte_carlo, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(2) if ISING else np.arange(10), np.mean(probs[:, indices[i], :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs, test set")
        
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))

def build_input_pipeline(mnist_data, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))

    print(mnist_data.train.images.shape)
    training_batches = training_dataset.shuffle(
        50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
        np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
    heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)


    test_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.test.images,
        np.int32(mnist_data.test.labels)))
    test_frozen = (test_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_frozen)


    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator, test_iterator


def test_data_pipeline(mnist_data, batch_size):
    """Build an Iterator switching between train and heldout data."""


    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.test.images))
    heldout_frozen = (heldout_dataset.take(batch_size).
                    repeat().batch(batch_size))
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)

    # Combine these into a feedable iterator that can switch between training
    # and test inputs.
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, heldout_dataset.output_types, heldout_dataset.output_shapes)
    images = feedable_iterator.get_next()

    return images, handle, test_iterator


def Get_ising_data():
    import pickle
    
    
    def read_t(t,root="/home/samknu/MyRepos/Bayesian-Machine-Learning/data/IsingData/"):
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
        return np.unpackbits(data).astype(int).reshape(-1,1600)
    
    temperatures = np.arange(0.25, 4., step=0.25)
    
    ordered = np.zeros(shape=(np.sum(temperatures<2.0),10000,1600))
    disordered = np.zeros(shape=(np.sum(temperatures>2.5),10000,1600))
    critical = np.zeros(shape=(np.sum((temperatures>=2.0)*(temperatures<=2.5)),10000,1600))
    
    ordered_index = 0
    disordered_index = 0
    crit_index = 0
    for i in range(len(temperatures)):
        T = temperatures[i]
        if T < 2.0:
            ordered[ordered_index] = read_t(T)
            ordered_index += 1
        elif T > 2.5:
            disordered[disordered_index] = read_t(T)
            disordered_index += 1
        else:
            critical[crit_index] = read_t(T)
            crit_index += 1

    ordered = ordered.reshape(-1,1600)       # 70000
    disordered = disordered.reshape(-1,1600) # 50000
    critical = critical.reshape(-1,1600)     # 30000

    # Shuffling before separating into training, validation and test set
    np.random.shuffle(ordered)
    np.random.shuffle(disordered)
    np.random.shuffle(critical)

    training_data = np.zeros((6000*12,1600))
    validation_data = np.zeros((2000*12,1600))
    test_data = np.zeros((2000*12 + 10000*3,1600))

    training_data[:round(0.6*70000)] = ordered[:round(0.6*70000)]
    training_data[round(0.6*70000):] = disordered[:round(0.6*50000)]

    validation_data[:round(0.2*70000)] = ordered[round(0.6*70000):round(0.6*70000)+round(0.2*70000)]
    validation_data[round(0.2*70000):] = disordered[round(0.6*50000):round(0.6*50000)+round(0.2*50000)]

    test_data[:round(0.2*70000)] = ordered[round(0.6*70000)+round(0.2*70000):round(0.6*70000)+2*round(0.2*70000)]
    test_data[round(0.2*70000):round(0.2*70000)+round(0.2*50000)] = disordered[round(0.6*50000)+round(0.2*50000):round(0.6*50000)+2*round(0.2*50000)]
    test_data[round(0.2*70000)+round(0.2*50000):] = critical

    training_labels = np.zeros(6000*12)
    training_labels[round(0.6*70000):] = np.ones(round(0.6*50000))

    validation_labels = np.zeros(2000*12)
    validation_labels[round(0.2*70000):] = np.ones(round(0.2*50000))

    # Class 0 is ordered, class 1 is disordered

    ############################################################
    # Reshaping since we want them as matrices for convolution #
    ############################################################
    training_data = training_data.reshape(-1,40,40)
    training_data = training_data[:,:,:,np.newaxis]

    validation_data = validation_data.reshape(-1,40,40)
    validation_data = validation_data[:,:,:,np.newaxis]
    
    test_data = test_data.reshape(-1,40,40)
    test_data = test_data[:,:,:,np.newaxis]
    

    del ordered
    del disordered
    del critical
    del temperatures

    
    #############################
    # Shuffling data and labels #
    #############################
    indices = np.random.permutation(np.arange(training_data.shape[0]))
    training_data = training_data[indices]
    training_labels = training_labels[indices]
    
    indices = np.random.permutation(np.arange(validation_data.shape[0]))
    validation_data = validation_data[indices]
    validation_labels = validation_labels[indices]
    
    indices = np.random.permutation(np.arange(test_data.shape[0]))
    test_data = test_data[indices]
    #test_labels = test_labels[indices]
    
    cut_train = 20000   
    cut_val = 5000
    cut_test = 1000
    training_data = training_data[:cut_train]
    training_labels = training_labels[:cut_train]

    validation_data = validation_data[:cut_val]
    validation_labels = validation_labels[:cut_val]
    
    test_data = test_data[:cut_test]

    class Dummy(object):
        pass
    ising_data = Dummy()
    ising_data.train=Dummy()
    ising_data.train.images = training_data
    ising_data.train.labels = training_labels
    ising_data.train.num_examples = training_data.shape[0]

    ising_data.validation=Dummy()
    ising_data.validation.images = validation_data
    ising_data.validation.labels = validation_labels
    ising_data.validation.num_examples = validation_data.shape[0]


    ising_data.test=Dummy()
    ising_data.test.images = test_data
    ising_data.test.labels = np.zeros(test_data.shape[0])   # dummy labels
    ising_data.test.num_examples = test_data.shape[0]

    return ising_data


def main(argv):
    del argv  # unused

    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.compat.v1.logging.warning(
            "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)



    if ISING:
        the_data = Get_ising_data()
    else:
        the_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

    
    (images, labels, handle, training_iterator, heldout_iterator, test_iterator) = build_input_pipeline(
           the_data, FLAGS.batch_size, the_data.validation.num_examples)  


    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
    with tf.compat.v1.name_scope("bayesian_neural_net", values=[images]):
        neural_net = tf.keras.Sequential([
            tfp.layers.Convolution2DFlipout(6,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                            strides=[2, 2],
                                            padding="SAME"),
            tfp.layers.Convolution2DFlipout(16,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                            strides=[2, 2],
                                            padding="SAME"),
            tfp.layers.Convolution2DFlipout(120,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(2) if ISING else tfp.layers.DenseFlipout(10)
            ])
    
    logits = neural_net(images)
    labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(
        input_tensor=labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / the_data.train.num_examples     # 72000 is the size of the training set
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for validation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for layers with weight distributions
    # for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
        try:
            q = layer.kernel_posterior
        except AttributeError:
            continue
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())


    with tf.compat.v1.name_scope("train"):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        
        for step in range(FLAGS.max_steps):
            #for step in range(0):
            _ = sess.run([train_op, accuracy_update_op],
                        feed_dict={handle: train_handle})
            if step % 100 == 0:
                loss_value, accuracy_value = sess.run(
                    [elbo_loss, accuracy], feed_dict={handle: train_handle})
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                    step, loss_value, accuracy_value))

            if (step+1) % FLAGS.viz_steps == 0:
                # Compute log prob of heldout set by averaging draws from the model:
                # p(heldout | train) = int_model p(heldout|model) p(model|train)
                #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
                # where model_i is a draw from the posterior p(model|train).
                probs = np.asarray([sess.run((labels_distribution.probs),
                                            feed_dict={handle: heldout_handle})
                                    for _ in range(FLAGS.num_monte_carlo)])
                mean_probs = np.mean(probs, axis=0)

                image_vals, label_vals = sess.run((images, labels),
                                                feed_dict={handle: heldout_handle})
                
                
                probs_test = np.asarray([sess.run((labels_distribution.probs),
                                            feed_dict={handle: test_handle})
                                    for _ in range(FLAGS.num_monte_carlo)])
                mean_probs_test = np.mean(probs_test, axis=0)
                image_vals_test = sess.run((images),
                                                feed_dict={handle: test_handle})

                heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                    label_vals.flatten()]))
                                                    
                print(" ... Held-out nats: {:.3f}".format(heldout_lp))

                qm_vals, qs_vals = sess.run((qmeans, qstds))

                if HAS_SEABORN:
                    plot_weight_posteriors(names, qm_vals, qs_vals,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_weights.png".format(step)))

                    plot_heldout_prediction(image_vals, label_vals, probs,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_pred.png".format(step)),
                                            title="mean heldout logprob {:.2f}"
                                            .format(heldout_lp))

                    plot_test_prediction(image_vals_test, probs_test,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_test_pred.png".format(step)))


if __name__ == "__main__":
    
    tf.compat.v1.app.run()      # this thing will run the main(argv) function with sys.argv as argument
```

    WARNING:tensorflow:
    
      TensorFlow's `tf-nightly` package will soon be updated to TensorFlow 2.0.
    
      Please upgrade your code to TensorFlow 2.0:
        * https://www.tensorflow.org/beta/guide/migration_guide
    
      Or install the latest stable TensorFlow 1.X release:
        * `pip install -U "tensorflow==1.*"`
    
      Otherwise your code may be broken by the change.
    
      
    WARNING:tensorflow:Warning: deleting old log directory at /tmp/bayesian_neural_network/


    W1017 18:50:07.396682 140424763520832 <ipython-input-1-9cdff932cc21>:382] Warning: deleting old log directory at /tmp/bayesian_neural_network/


    (20000, 40, 40, 1)
    WARNING:tensorflow:From <ipython-input-1-9cdff932cc21>:222: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_types(dataset)`.


    W1017 18:50:12.770762 140424763520832 deprecation.py:323] From <ipython-input-1-9cdff932cc21>:222: DatasetV1.output_types (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_types(dataset)`.


    WARNING:tensorflow:From <ipython-input-1-9cdff932cc21>:222: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_shapes(dataset)`.


    W1017 18:50:12.771799 140424763520832 deprecation.py:323] From <ipython-input-1-9cdff932cc21>:222: DatasetV1.output_shapes (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_shapes(dataset)`.


    WARNING:tensorflow:From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_probability/python/layers/util.py:104: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.


    W1017 18:50:12.780434 140424763520832 deprecation.py:323] From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_probability/python/layers/util.py:104: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.


    WARNING:tensorflow:From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.


    W1017 18:50:12.789917 140424763520832 deprecation.py:506] From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.


    Step:   0 Loss: 138.866 Accuracy: 0.555
    Step: 100 Loss: 129.560 Accuracy: 0.938
    Step: 200 Loss: 124.809 Accuracy: 0.966
    Step: 300 Loss: 119.960 Accuracy: 0.976
    WARNING:tensorflow:From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_probability/python/distributions/categorical.py:232: Categorical._probs_deprecated_behavior (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-10-01.
    Instructions for updating:
    The `probs` property will return `None` when the distribution is parameterized with `probs=None`. Use `probs_parameter()` instead.


    W1017 18:52:05.901764 140424763520832 deprecation.py:323] From /home/samknu/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_probability/python/distributions/categorical.py:232: Categorical._probs_deprecated_behavior (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-10-01.
    Instructions for updating:
    The `probs` property will return `None` when the distribution is parameterized with `probs=None`. Use `probs_parameter()` instead.


     ... Held-out nats: -0.001
    saved /tmp/bayesian_neural_network/step00399_weights.png
    saved /tmp/bayesian_neural_network/step00399_pred.png
    saved /tmp/bayesian_neural_network/step00399_test_pred.png
    Step: 400 Loss: 115.113 Accuracy: 0.982
    Step: 500 Loss: 110.279 Accuracy: 0.985
    Step: 600 Loss: 105.423 Accuracy: 0.987
    Step: 700 Loss: 100.600 Accuracy: 0.989
     ... Held-out nats: -0.001
    saved /tmp/bayesian_neural_network/step00799_weights.png
    saved /tmp/bayesian_neural_network/step00799_pred.png
    saved /tmp/bayesian_neural_network/step00799_test_pred.png
    Step: 800 Loss: 95.806 Accuracy: 0.990
    Step: 900 Loss: 91.051 Accuracy: 0.991
    Step: 1000 Loss: 86.354 Accuracy: 0.992
    Step: 1100 Loss: 81.685 Accuracy: 0.993
     ... Held-out nats: -0.001
    saved /tmp/bayesian_neural_network/step01199_weights.png
    saved /tmp/bayesian_neural_network/step01199_pred.png
    saved /tmp/bayesian_neural_network/step01199_test_pred.png
    Step: 1200 Loss: 77.086 Accuracy: 0.993
    Step: 1300 Loss: 72.554 Accuracy: 0.994
    Step: 1400 Loss: 68.109 Accuracy: 0.994
    Step: 1500 Loss: 63.750 Accuracy: 0.994
     ... Held-out nats: -0.000
    saved /tmp/bayesian_neural_network/step01599_weights.png
    saved /tmp/bayesian_neural_network/step01599_pred.png
    saved /tmp/bayesian_neural_network/step01599_test_pred.png
    Step: 1600 Loss: 59.483 Accuracy: 0.995
    Step: 1700 Loss: 55.313 Accuracy: 0.995
    Step: 1800 Loss: 51.256 Accuracy: 0.995
    Step: 1900 Loss: 47.308 Accuracy: 0.995
     ... Held-out nats: -0.001



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-1-9cdff932cc21> in <module>
        529 if __name__ == "__main__":
        530 
    --> 531     tf.compat.v1.app.run()      # this thing will run the main(argv) function with sys.argv as argument
    

    ~/anaconda3/envs/bayesianML/lib/python3.7/site-packages/tensorflow_core/python/platform/app.py in run(main, argv)
         38   main = main or _sys.modules['__main__'].main
         39 
    ---> 40   _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
    

    ~/anaconda3/envs/bayesianML/lib/python3.7/site-packages/absl/app.py in run(main, argv, flags_parser)
        297       callback()
        298     try:
    --> 299       _run_main(main, args)
        300     except UsageError as error:
        301       usage(shorthelp=True, detailed_error=error, exitcode=error.exitcode)


    ~/anaconda3/envs/bayesianML/lib/python3.7/site-packages/absl/app.py in _run_main(main, argv)
        248     sys.exit(retval)
        249   else:
    --> 250     sys.exit(main(argv))
        251 
        252 


    <ipython-input-1-9cdff932cc21> in main(***failed resolving arguments***)
        512                                             fname=os.path.join(
        513                                                 FLAGS.model_dir,
    --> 514                                                 "step{:05d}_weights.png".format(step)))
        515 
        516                     plot_heldout_prediction(image_vals, label_vals, probs,


    <ipython-input-1-9cdff932cc21> in plot_weight_posteriors(names, qm_vals, qs_vals, fname)
        100     ax = fig.add_subplot(1, 2, 2)
        101     for n, qs in zip(names, qs_vals):
    --> 102         sns.distplot(qs.flatten(), ax=ax)
        103     ax.set_title("weight stddevs")
        104     ax.set_xlim([0, 1.])


    ~/anaconda3/lib/python3.7/site-packages/seaborn/distributions.py in distplot(a, bins, hist, kde, rug, fit, hist_kws, kde_kws, rug_kws, fit_kws, color, vertical, norm_hist, axlabel, label, ax)
        213     if hist:
        214         if bins is None:
    --> 215             bins = min(_freedman_diaconis_bins(a), 50)
        216         hist_kws.setdefault("alpha", 0.4)
        217         if LooseVersion(mpl.__version__) < LooseVersion("2.2"):


    ~/anaconda3/lib/python3.7/site-packages/seaborn/distributions.py in _freedman_diaconis_bins(a)
         32     if len(a) < 2:
         33         return 1
    ---> 34     h = 2 * iqr(a) / (len(a) ** (1 / 3))
         35     # fall back to sqrt(a) bins if iqr is 0
         36     if h == 0:


    ~/anaconda3/lib/python3.7/site-packages/seaborn/utils.py in iqr(a)
        365     a = np.asarray(a)
        366     q1 = stats.scoreatpercentile(a, 25)
    --> 367     q3 = stats.scoreatpercentile(a, 75)
        368     return q3 - q1
        369 


    ~/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py in scoreatpercentile(a, per, limit, interpolation_method, axis)
       1725         a = a[(limit[0] <= a) & (a <= limit[1])]
       1726 
    -> 1727     sorted_ = np.sort(a, axis=axis)
       1728     if axis is None:
       1729         axis = 0


    ~/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py in sort(a, axis, kind, order)
        932     else:
        933         a = asanyarray(a).copy(order="K")
    --> 934     a.sort(axis=axis, kind=kind, order=order)
        935     return a
        936 


    KeyboardInterrupt: 


Weights near the beginning:
![step00399_weights.png](attachment:step00399_weights.png)
Weights near the end:
![step03999_weights.png](attachment:step03999_weights.png)

### Classification on the unseen data:
(The data includes both ordered and disordered in addition to the critical state)

Class 0 = ordered, Class 1 = disordered
![step04399_test_pred.png](attachment:step04399_test_pred.png)

![step04799_test_pred.png](attachment:step04799_test_pred.png)

# Part 4 - Bayesian Reasoning on the Probability of Life

A frequentist will claim that we cannot say anything about the probability that life can arise, because we have only observed a single example of it. Here I will give a Bayesian explanation of why this intuition is wrong.

In Bayesian logic we can use other types of information about life on Earth to do inference -- in this case the datapoint that life on Earth seems to have appeared very shortly after the planet cooled down enough to allow for complex molecules to exist. The reasoning goes like this:

Let us say that we _initially_ only know two facts:
    1. Life exist on Earth 
    2. We also have a modern understanding of biology, meaning that we know that life is essentially an extension of thermodynamics, but we have no information about the actual probability of life to spontaneously appear


We now ask ourselves if we can say something about the probability of life to occur or not. To simplify the analysis we assume a binary hypothesis space $\theta$ where 


$$
          \theta = 0 \ \text{is the hypothesis that life has a} \textbf{ low } \text{probability of occurring}\\
          \theta = 1 \ \text{is the hypothesis that life has a} \textbf{ high } \text{probability of occurring}.          
$$


The question is then which of these hypotheses is true. Since we are initially completely ignorant, meaning we have no reason to believe either hypothesis more than the other, we start with a uniform prior, i.e.

$$
p(\theta=0) = 0.5
$$
and
$$
p(\theta=1) = 0.5 .
$$


Let us then assume that we observe a datapoint 
$$D = \{\textrm{Life appeared shortly after it was possible}\}.$$ 

Using Bayes theorem we can then write our two posterior estimates as

$$
p(\theta=0|D) = \frac{p(D|\theta=0)p(\theta=0)}{p(D)}
$$
and
$$
p(\theta=1|D) = \frac{p(D|\theta=1)p(\theta=1)}{p(D)}
$$


The denominators are the same in both cases, and since the priors are $p(\theta=0)=p(\theta=1)$ we see that the only factors that differ between the two posterior hypotheses are the likelihood factors $p(D|\theta=1)$ and $p(D|\theta=0)$.


Further it must be true that $p(D|\theta=1)>p(D|\theta=0)$, since observing datapoint $D$ is more probable if $\theta=1$ than if $\theta=0$, so it follows that

$$
p(\theta=1|D) > p(\theta=0|D). 
$$

So we conclude that our posterior, based on this single datapoint $D$, says that we should give a higher probability estimate that $\theta=1$ is true. In other words, we should lend more credence to the hypothesis that life has a high probability of occurring.

We could also do this analysis with a continuous hypothesis space. Our prior would then not be uniform, but would fall to zero towards the right end, because we know that if life were extremely, extremely probable we would have seen it arise in experiments, which we have not.

## References


[1] Shridhar et al. Uncertainty Estimations by Softplus normalization in Bayesian Convolutional Neural Networks with Variational Inference https://arxiv.org/pdf/1806.05978.pdf

[2] Wen et al. Flipout: Efficient pseudo independent weight perturbations on mini-batches https://arxiv.org/pdf/1803.04386.pdf

[3] LeCun et al. Gradient-Based Learning Applied to Document Recognition http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

[4] Mehta et al. A high-bias, low-variance introduction to Machine Learning for physisits https://arxiv.org/pdf/1803.08823.pdf, https://physics.bu.edu/~pankajm/MLnotebooks.html

[5] Book: Data Analysis - A Bayesian Tutorial by Devinderjit Sivia

[6] Book: Machine Learning: A Probabilistic Perspective by Kevin P. Murphy



```python

```

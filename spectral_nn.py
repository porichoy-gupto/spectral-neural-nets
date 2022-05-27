import torch
from scipy.stats import ortho_group
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 14})
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

m, n, k = 60, 10, 10
s = 3
nlow = 10

num_iter = 10000

def dot(A, X):
    X = X.repeat(m, 1, 1)
    return torch.sum(A * X, dim=[1, 2])

# Generate data
U0 = torch.randn(n, n)
A0 = U0 @ U0.T
Phi, S0, PsiT = torch.linalg.svd(A0)
Xtrue = (torch.rand(n, n) * 2 - 1) * 0.25 + 5

A = []
for i in range(m):
    Ai = torch.randn(n, n)
    A.append(Ai)
A = torch.stack(A)
y = dot(A, Xtrue)

epsilon = 1

def rec_diag_matrix(a, m, n):
    # Return an m x n matrix with the vector a as the diagonal elements
    Q = torch.zeros(m, n)
    Q[:m, :m] = torch.diagflat(a)
    return Q

def diagonal_init(Phi, PsiT):
    G = torch.tensor(ortho_group.rvs(n), dtype=torch.float)
    Ubar, _ = torch.sort(torch.abs(torch.randn(n) * epsilon))
    Vbar, _ = torch.sort(torch.abs(torch.randn(n) * epsilon))
    U_init = Phi @ torch.diag(Ubar) @ G
    V_init = PsiT.T @ torch.diag(Vbar) @ G
    return U_init, V_init

def random_init():
    U_init = torch.eye(n) * 10e-2 #* U_init.mean()
    V_init = torch.eye(n) * 10e-2 #* V_init.mean()
    return U_init, V_init

def non_linear_activation(Q):
    # Apply a non-linear on the spectrum of Q
    U, S, Vt = torch.linalg.svd(Q + 1e-2 * Q.mean() * torch.rand_like(Q), full_matrices=False)
    S = torch.sigmoid(S)
    return U @ torch.diag(S) @ Vt

def non_linear_huber(Q):
    # Apply a non-linear on the spectrum of Q
    U, S, Vt = torch.linalg.svd(Q + 1e-2 * Q.mean() * torch.rand_like(Q), full_matrices=False)
    S = torch.clamp(S, min=0, max=0.8)
    return U @ torch.diag(S) @ Vt

def non_linear_sigmoid_shifted_and_scaled(Q):
    # Apply a non-linear on the spectrum of Q
    U, S, Vt = torch.linalg.svd(Q + 1e-2 * Q.mean() * torch.rand_like(Q), full_matrices=False)
    S = (torch.sigmoid(S) - 0.5) * 0.8
    return U @ torch.diag(S) @ Vt

def train_error(X, A, y):
    return torch.sqrt(torch.sum(torch.square(y - dot(A, X))) / torch.sum(torch.square(y)))

def test_error(X, Xtrue):
    return torch.norm(X - Xtrue, p='fro') / torch.norm(Xtrue, p='fro')
    

class SNN(torch.nn.Module):
    def __init__(self, A, y, Phi, PsiT, m, n, k, s, L=[s, 3]):
        super(SNN, self).__init__()
        self.Us = torch.nn.ParameterList()
        self.Vs = torch.nn.ParameterList()
        self.alpha = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(L[0], L[1]) * 1e0),
            torch.nn.Parameter(torch.ones(L[1]) * 1e0)
        ])
        self.L = L
        for i in range(L[0]):
            U, V = random_init()
            U = torch.nn.Parameter(U)
            V = torch.nn.Parameter(V)
            self.Us.append(U)
            self.Vs.append(V)
        
    def forward(self):
        Xs = []
        for j in range(self.L[1]):
            result = torch.zeros_like(self.Us[0])
            for i, (U, V) in enumerate(zip(self.Us, self.Vs)):
                result += self.alpha[0][i, j] * non_linear_activation(U @ V.T)
            Xs.append(result)
        result = torch.zeros_like(self.Us[0])
        for i, X in enumerate(Xs):
            result += self.alpha[1][i] * (X)
        return result

    def sv(self):
        Q = self.forward()
        return torch.diag(torch.inverse(Phi) @ Q @ torch.inverse(PsiT))

    def nuclear_norm(self):
        return torch.norm(self.sv())


model = SNN(A, y, Phi, PsiT, m, n, k, s)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

singular_value_traces = []
nuclear_norm_traces = []
train_error_traces = []
test_error_traces = []

for i in range(num_iter):
    if i == 0 or (i+1) % 30 == 0:
        singular_value_traces.append(model.sv())
        nuclear_norm_traces.append(model.nuclear_norm())
        train_error_traces.append(train_error(model.forward(), A, y))
        test_error_traces.append(test_error(model.forward(), Xtrue))
    optimizer.zero_grad()
    loss = 0.5 * torch.sum(torch.square(y - dot(A, model())))
    loss.backward()
    optimizer.step()

    if i == 0 or (i+1) % (num_iter // 10) == 0: 
        print('Loss at iter {}: {}'.format(i+1, loss.item()))

singular_value_traces = torch.stack(singular_value_traces).detach().numpy().T
nuclear_norm_traces = torch.stack(nuclear_norm_traces).detach().numpy().T
train_error_traces = torch.stack(train_error_traces).detach().numpy()
test_error_traces = torch.stack(test_error_traces).detach().numpy()
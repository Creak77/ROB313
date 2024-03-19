import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset


def MDL(n, r, k):
    loss = np.mean(np.square(r))
    return (n/2)*np.log(loss) + k/2*np.log(n)

D = []
p = 5
count = (200-p+1)//2
for i in range(p+1):
    D += [lambda x, i=i: x**i]
for i in range(1, count):
    D += [lambda x, i=i: np.sin(i*x*np.pi)]
for i in range(1, count):
    D += [lambda x, i=i: np.cos(i*x*np.pi)]

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
x_train = np.vstack([x_valid, x_train])
y_train = np.vstack([y_valid, y_train])
n = x_train.shape[0]
r = np.copy(y_train)
candidates = np.arange(len(D))
select = []
k = 0
best = np.inf
while len(candidates) > 0:
    k += 1
    J = []
    for i in candidates:
        x = np.array(D[i](x_train))
        J += [((x.T @ r) ** 2) / np.abs(x.T @ x)]
    select += [candidates[np.argmax(J)]]
    candidates = np.delete(candidates, np.argmax(J))
    Phi = np.empty((n, k))
    for i, j in enumerate(select):
        Phi[:, i] = D[j](x_train.T)
    U, S, V = np.linalg.svd(Phi, full_matrices=False)
    w = V.T @ np.diag(1/S) @ U.T @ y_train
    y_pred = Phi @ w

    plt.plot(x_train, y_train, label='actual')
    plt.plot(x_train, y_pred, label='prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()

    r = y_train - y_pred
    mdl = MDL(n, r, k)
    if mdl < best:
        best = mdl
    else:
        break

select = select[:-1]
w = w[:-1]

test = np.empty((x_test.shape[0], len(select)))
for i, j in enumerate(select):
    test[:, i] = D[j](x_test.T)
y_pred = test @ w

plt.plot(x_test, y_test, label='actual')
plt.plot(x_test, y_pred, label='prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
RMSE = np.sqrt(np.mean(np.square(y_test - y_pred)))
print('RMSE: {}'.format(RMSE))


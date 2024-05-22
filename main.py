import random
import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    f1 = 100 * (x2 - x1**2)**2 + 5 * (1 - x1)**2
    return f1

def f2(x1, x2):
    f2 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    return f2
(-2.805, 3.131)
def euclidian_norm(x1, x2):
    return (x1**2 + x2**2)**0.5

def grad_f(f, x1, x2, eps=1e-3, max_norm=25):
    g1 = (f(x1 + eps, x2) - f(x1, x2)) / eps
    g2 = (f(x1, x2 + eps) - f(x1, x2)) / eps

    if max_norm:
        norm = euclidian_norm(g1, g2)

        if norm > max_norm:
            k = max_norm / norm
            g1 *= k
            g2 *= k

    return g1, g2

def hessian_f1(x1, x2):
    d2f1_dx1dx1 = 1200 * x1**2 - 400 * x2 + 2
    d2f1_dx1dx2 = -400 * x1
    d2f1_dx2dx1 = -400 * x1
    d2f1_dx2dx2 = 200
    return np.array([[d2f1_dx1dx1, d2f1_dx1dx2], [d2f1_dx2dx1, d2f1_dx2dx2]])


def hessian_f2(x1, x2):
    ddf1x1 = 12 * x1**2 + 4 * x2 - 42
    ddf1x2 = 4 * x1 + 4 * x2
    ddf2x1 = 4 * x1 + 4 * x2
    ddf2x2 = 4 * x1 + 12 * x2**2 - 26
    return np.array([[ddf1x1, ddf1x2], [ddf2x1, ddf2x2]])

def newton_optimize(f, hessian_f, x10, x20, x_eps=1e-4, grad_eps=1e-4, deriv_eps=1e-3, max_iters=100):
    X1, X2 = [x10], [x20]
    x1, x2 = x10, x20
    good = True
    iterations = 0

    for i in range(max_iters):
        g = np.array(grad_f(f, x1, x2, deriv_eps))
        hes = hessian_f(x1, x2)

        if np.linalg.det(hes) < 1e-1:
            good = False
            break

        search_direction = np.linalg.solve(hes, -g)

        x1 = x1 + search_direction[0]
        x2 = x2 + search_direction[1]

        X1.append(x1)
        X2.append(x2)

        iterations += 1

        if ((x1 - X1[-2])**2 + (x2 - X2[-2])**2) < x_eps**2 or np.linalg.norm(search_direction) < grad_eps:
            break

    return (x1, x2), (X1, X2), iterations, good

random.seed(61)

N_maxs = [1000, 100]
eps = 1e-4
mins = [
    [(1, 1)],
    [
        (3, 2),
        (-3.779, -3.283),
        (-2.805, 3.131),
        (3.584, -1.848),
    ],
]
verbose = True

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i, ax, f, hessian_f in zip(range(2), axs, [f1, f2], [hessian_f1, hessian_f2]):
    x1 = np.linspace(-5, 5, 1000)
    x2 = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x1, x2)
    Z = f(X, Y)

    pcm = ax.contourf(X, Y, Z, levels=20)
    ax.set_title(f'function optimization f{i+1}')

    iterations_array = []

    # function optimization (start from different points)
    for col in 10 * ['w', 'c', 'm', 'y', 'k', 'r', 'g', 'b']:
        x0 = (random.random() * 10 - 5, random.random() * 10 - 5)

        x_opt, x_trace, iterations, good = newton_optimize(f, hessian_f, *x0, eps, max_iters=N_maxs[i])
        iterations_array.append(iterations)

        ax.plot(*x_trace, f'{col}-')
        ax.plot(*x0, f'go')
        ax.plot(*x_opt, f'{("w" if good else "r")}o')


    if verbose:
        for col, mp in zip(['c', 'm', 'y', 'k', 'g', 'r', 'b'], mins[i]):
            ax.scatter(*mp, c=col, s=200)

        min_it = min(iterations_array)
        mean_it = np.mean(iterations_array)
        max_it = max(iterations_array)
        print(f'number of iterations function{i+1}: {min_it, mean_it, max_it} ')

    axs[i].set_xlim(-5, 5)
    axs[i].set_ylim(-5, 5)

plt.show()

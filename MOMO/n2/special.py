import numpy as np
from scipy.linalg import orth


def grad_finite_diff(func, x, eps=1e-8):
    """
    Подсчет градиента с помощью конечных разностей

    Параметры:
    ----------
    func: callable func(x)
        Функция, градиент которой нужно вычислить
        Принимает:
            x: np.ndarray
                Аргумент функции, вектор размера n
        Возвращает:
            f: float
                Значение функции в точке x

    x: np.ndarray
        Точка, в которой нужно вычислить градиент, вектор размера n

    eps: float, опционально
        Число eps_1 в формуле (5)

    Возврат:
    ----------
    g: np.ndarray
        Оценка градиента по формуле (5), n-мерный вектор

    """

    """
    Didn't use func(x + eps * np.eye(x.size)), cause func takes 1dim vec
    """
    f = func(x)
    e = np.zeros(x.size)

    g = np.empty(x.size)
    for i in range(g.size):
        e[i] = 1
        g[i] = (func(x + eps * e) - f) / eps
        e[i] = 0

    return g


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Подсчет действия гессиана на вектор с помощью конечных разностей

    Параметры:
    ----------
    func: callable func(x)
        Функция, у которой нужно вычислить произведение гессиана на вектор
        Принимает:
            x: np.ndarray
                Аргумент функции, вектор размера n
        Возвращает:
            f: float
                Значение функции в точке x

    x: np.ndarray
        Точка, в которой нужно вычислить гессиан, n-мерный вектор

    v: np.ndarray
        Вектор, на который нужно умножить гессиан; размер вектора n

    eps: float, опционально
        Число eps_1 в формуле (5)

    Возврат:
    ----------
    hv: np.ndarray
        Оценка hessian(f(x))v по формуле (5), вектор размера n

    """
    f = func(x)
    e = np.zeros(x.size)

    g = np.empty(x.size)
    for i in range(g.size):
        e[i] = 1
        g[i] = (func(x + eps * (v + e)) -
                func(x + eps * v) -
                func(x + eps * e) + f) / (eps ** 2)
        e[i] = 0

    return g


def gen_symm_matr(kappa, n):
    """
    Генерация случайной симметричной матрицы с заданным числом обусловленности

    Параметры:
    ----------
    kappa: float
        Число обусловленности матрицы

    n: int
        Размер матрицы

    Возврат:
    ----------
    matr: np.ndarray, ndim=2
        Матрца A: A=A.T > 0

    """
    s = np.random.uniform(1, kappa, size=n)
    s[0] = 1
    s[-1] = kappa

    Q = orth(np.random.randn(n, n))
    S = np.diag(s)

    return np.dot(Q, S.dot(Q.T))

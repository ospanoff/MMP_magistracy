import numpy as np
from scipy.sparse import csr_matrix


def logistic(w, X, y, reg_coef):
    """
    Оракул функции потерь логистической регрессии (1)

    Параметры:
    ----------
    w: np.ndarray
        Точка вычисления, вектор размера d

    X: np.ndarray или sp.sparse.csr_matrix
        Матрица признаков размеров n × d

    y: np.ndarray
        Метки классов, вектор размера n, состоящий из элементов {−1, +1}

    reg_coef: float
        Коэффициент регуляризации λ > 0

    Возврат:
    ----------
    f: float
        Значение функции в точке w

    g: np.ndarray
        Градиент функции в точке w, вектор размера d

    """
    w = w[:, np.newaxis]
    if type(X) == np.ndarray:
        A = -y[:, np.newaxis] * X

    else:
        A = X.multiply(csr_matrix(-y).T)

    Aw = A.dot(w)

    s = 1 / (1 + np.exp(-Aw))
    f = np.logaddexp(0, Aw).mean() + reg_coef * np.dot(w.T, w) / 2

    return (
        f[0, 0],
        A.T.dot(s) / y.size + reg_coef * w,
    )


def logistic_hess_vec(w, v, X, y, reg_coef):
    """
    Умножение гессиана функции логистической регрессии (1) на произвольный
    вектор

    Параметры:
    ----------
    w: np.ndarray
        Точка вычисления, d-мерный вектор

    v: np.ndarray
        Вектор, на который умножается гессиан, d-мерный вектор

    X: np.ndarray или sp.sparse.csr_matrix
        Матрица признаков размеров n × d

    y: np.ndarray
        Метки классов, вектор размера n, состоящий из элементов {−1, +1}

    reg_coef: float
        Коэффициент регуляризации λ > 0

    Возврат:
    ----------
    hv: np.ndarray
        Вектор hessian(f(w))v

    """
    w = w[:, np.newaxis]

    if type(X) == np.ndarray:
        A = -y[:, np.newaxis] * X

    else:
        A = X.multiply(csr_matrix(-y).T)

    f = 1 / (1 + np.exp(A.dot(w)))
    s = f * (1 - f)

    v = v[:, np.newaxis]
    ret = A.dot(v)

    return (A.T.dot(s * ret) / y.size + reg_coef * v).ravel()

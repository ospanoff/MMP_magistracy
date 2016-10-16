import numpy as np


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
    x = -y * X.dot(w)

    s = 1 / (1 + np.exp(-x))

    return (
        np.logaddexp(0, x).mean() + reg_coef * w.dot(w) / 2,
        X.T.dot(-y * s) / y.size + reg_coef * w,
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
    x = -y * X.dot(w)

    f = 1 / (1 + np.exp(x))
    s = f * (1 - f)

    return X.T.dot(s * X.dot(v)) / y.size + reg_coef * v

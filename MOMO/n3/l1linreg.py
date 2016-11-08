import numpy as np


def barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-5, tol_inner=1e-8,
            max_iter=100, max_iter_inner=20, t0=1, gamma=10, c1=1e-4,
            disp=False, trace=False):
    """
    Метод барьеров для решения задачи (5)

    Параметры:
    ----------
    X: np.ndarray
        Матрица признаков размеров n × d.

    y: np.ndarray
        Регрессионные значения, вектор размера n.

    ref_coef: float
        Коэффициент регуляризации λ > 0

    w0_plus: np.ndarray
        Начальная точка w_0^+, вектор размера d.

    w0_minus: np.ndarray
        Начальная точка w_0^-, вектор размера d.

    tol: float, опционально
        Точность оптимизации по функции: φ(\hat{w}) − φ* ≤ ε.

    tol_inner: float, опционально
        Точность оптимизации функции φ_t на внутренних итерациях метода
        Ньютона: ||nabla φ_t(w_k^+, w_k^-)||_inf < ε

    max_iter: int, опционально
        Максимальное число (внешних) итераций метода.

    max_iter_inner: int, опционально
        Максимальное число внутренних итераций метода Ньютона.

    t0: float, опционально
        Начальное значение параметра центрирования t.

    gamma: float, опционально
        Коэффициент увеличения параметра t на внешних итерациях:
        t_{k+1} = γ * t_k

    c1: float, опционально
        Значение константы c1 в условии Армихо.

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, пройденное
        время, значение функции, зазор двойственности и пр.) или нет.

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет.

    Возврат:
    --------
    w_hat: np.ndarray
        Найденная точка \hat{w}, вектор размера d.

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью: φ(\hat{w}) − φ∗ ≤ ε;
            1: достигнуто максимальное число итераций.

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям (новая запись добавляется на
        каждой внутренней итерации метода). Словарь со следующими полями:
            elaps_t: np.ndarray
                Время, пройденное с начала оптимизации.
            phi: np.ndarray
                Текущее значение функции φ(w_k).
            dual_gap: np.ndarray
                Текущий зазор двойственности η(w_k, µ(w_k)), заданный в (4)

    """


def subgrad(X, y, reg_coef, w0, tol=1e-2, max_iter=1000, alpha=1,
            disp=False, trace=False):
    """
    Субградиентный метод для решения задачи (1)

    Параметры:
    ----------
    X: np.ndarray
        Матрица признаков размеров n × d.

    y: np.ndarray
        Регрессионные значения, вектор размера n.

    ref_coef: float
        Коэффициент регуляризации λ > 0

    w0: np.ndarray
        Начальная точка w0, вектор размера d.

    tol: float, опционально
        Точность оптимизации по функции: φ(\hat{w}) − φ* ≤ ε.

    max_iter: int, опционально
        Максимальное число (внешних) итераций метода.

    alpha: float, опционально
        Константа α для выбора длины шага: α_k = α/sqrt(k + 1).

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, пройденное
        время, значение функции, зазор двойственности и пр.) или нет.

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет.

    Возврат:
    --------
    w_hat: np.ndarray
        Найденная точка \hat{w}, вектор размера d.

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью: φ(\hat{w}) − φ∗ ≤ ε;
            1: достигнуто максимальное число итераций.

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям (новая запись добавляется на
        каждой внутренней итерации метода). Словарь со следующими полями:
            elaps_t: np.ndarray
                Время, пройденное с начала оптимизации.
            phi: np.ndarray
                Текущее значение функции φ(w_k).
            dual_gap: np.ndarray
                Текущий зазор двойственности η(w_k, µ(w_k)), заданный в (4)

    """


def prox_grad(X, y, reg_coef, w0, tol=1e-5, max_iter=1000, L0=1,
              disp=False, trace=False):
    """
    Проксимальный метод для решения задачи (1)

    Параметры:
    ----------
    X: np.ndarray
        Матрица признаков размеров n × d.

    y: np.ndarray
        Регрессионные значения, вектор размера n.

    ref_coef: float
        Коэффициент регуляризации λ > 0

    w0: np.ndarray
        Начальная точка w0, вектор размера d.

    tol: float, опционально
        Точность оптимизации по функции: φ(\hat{w}) − φ* ≤ ε.

    max_iter: int, опционально
        Максимальное число (внешних) итераций метода.

    L0: float, опционально
        Константа L0 в схеме Нестерова для подбора длины шага.

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, пройденное
        время, значение функции, зазор двойственности и пр.) или нет.

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет.

    Возврат:
    --------
    w_hat: np.ndarray
        Найденная точка \hat{w}, вектор размера d.

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью: φ(\hat{w}) − φ∗ ≤ ε;
            1: достигнуто максимальное число итераций.

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям (новая запись добавляется на
        каждой внутренней итерации метода). Словарь со следующими полями:
            elaps_t: np.ndarray
                Время, пройденное с начала оптимизации.
            phi: np.ndarray
                Текущее значение функции φ(w_k).
            dual_gap: np.ndarray
                Текущий зазор двойственности η(w_k, µ(w_k)), заданный в (4)
            ls_iters:
                Cуммарное (куммулятивное) число итераций одномерного поиска с
                самого начала работы метода.

    """

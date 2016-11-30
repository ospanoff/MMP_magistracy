import numpy as np
import time
from functools import wraps


def phi_main(w, X, y, reg_coef):
    n, d = X.shape
    phi_ = 0.5 * ((X.dot(w) - y) ** 2).sum() / n + reg_coef * np.abs(w).sum()

    return phi_


def duality_gap(w, X, y, reg_coef):
    n = y.size
    r = X.dot(w) - y
    mu = min(1, (reg_coef * n) / np.linalg.norm(X.T.dot(r), np.inf))
    mu *= (1 / n) * r
    eta = r.dot(r) / (2 * n) + reg_coef * np.linalg.norm(w, 1)
    eta += (n / 2) * mu.dot(mu) + y.dot(mu)
    return eta


def degenerate_case(method):
    @wraps(method)
    def wrapper(X, y, reg_coef, *args, **kwargs):
        disp = kwargs['disp'] if 'disp' in kwargs else False
        trace = kwargs['trace'] if 'trace' in kwargs else False

        n, d = X.shape
        status = 1
        hist = {'elaps_t': [], 'phi': [], 'dual_gap': []}
        if reg_coef >= np.linalg.norm(X.T.dot(y), ord=np.inf):
            w_hat = np.zeros(d)
            hist['elaps_t'] = np.array([0])
            hist['phi'] = np.array([phi_main(w_hat, X, y, reg_coef)])
            hist['dual_gap'] = np.array([duality_gap(w_hat, X, y, reg_coef)])
            status = 0
            if disp:
                print("lambda >= ||X.T * y||_inf")

            if trace:
                return (w_hat, status, hist)
            else:
                return (w_hat, status)

        else:
            return method(X, y, reg_coef, *args, **kwargs)

    return wrapper


@degenerate_case
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
    n, d = X.shape

    def phi(w_plus, w_minus, tau):  # phi_t
        phi_ = np.sum((X.dot(w_plus - w_minus) - y) ** 2)
        phi_ = tau / (2 * n) * phi_
        phi_ = phi_ + tau * reg_coef * np.sum(w_plus + w_minus)
        phi_ = phi_ - np.sum(np.log(w_plus) + np.log(w_minus))

        return phi_

    def nab_phi(w_plus, w_minus, tau):  # \nabla phi_t
        tmp = tau / n * X.T.dot(X.dot((w_plus - w_minus)) - y)

        nab_phi_p = tmp
        nab_phi_p += tau * reg_coef - 1 / w_plus

        nab_phi_n = -tmp
        nab_phi_n += tau * reg_coef - 1 / w_minus

        return nab_phi_p, nab_phi_n

    w_plus = w0_plus.astype(np.float)
    w_minus = w0_minus.astype(np.float)
    tau = t0

    status = 1
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    hist = {'elaps_t': [], 'phi': [], 'dual_gap': []}

    start_ = time.time()

    for k_out in range(max_iter):
        np_p, np_n = nab_phi(w_plus, w_minus, tau)
        phi_old = phi(w_plus, w_minus, tau)
        for k_inn in range(max_iter_inner):
            A = -tau / n * X.T.dot(X * ((w_plus / w_minus) ** 2 + 1))
            A = A - np.diag(1 / (w_minus ** 2))

            b = (1 / w_minus - 2 * tau * reg_coef) * (w_plus ** 2)
            b = 2 * w_plus - w_minus + b
            b = X.dot(b) - y
            b = - tau / n * X.T.dot(b) + tau * reg_coef - 1 / w_minus

            p_minus = np.linalg.solve(A, b)

            p_plus = (1 / w_minus - 2 * tau * reg_coef) * (w_plus ** 2)
            p_plus = w_plus + p_plus
            p_plus -= (w_plus ** 2) / (w_minus ** 2) * p_minus

            w = np.concatenate((w_plus, w_minus))
            p = np.concatenate((p_plus, p_minus))
            ind = p < 0
            if ind.sum() == 0:
                alpha_max = 10
            else:
                alpha_max = np.min(-w[ind] / p[ind])

            alpha = np.minimum(1, 0.95 * alpha_max)

            # Armijo
            while phi(w_plus + alpha * p_plus,
                      w_minus + alpha * p_minus, tau) > phi_old + \
                    alpha * c1 * (np_p.dot(p_plus) + np_n.dot(p_minus)):
                alpha /= 2

            w_plus += alpha * p_plus
            w_minus += alpha * p_minus

            phi_old = phi(w_plus, w_minus, tau)

            np_p, np_n = nab_phi(w_plus, w_minus, tau)
            np_p_norm = np.linalg.norm(np_p, ord=np.inf)
            np_n_norm = np.linalg.norm(np_n, ord=np.inf)
            if min(np_p_norm, np_n_norm) < tol_inner:
                break

        tau *= gamma

        w_hat = w_plus - w_minus
        dg = duality_gap(w_hat, X, y, reg_coef)
        el_t = time.time() - start_
        phi_m = phi_main(w_hat, X, y, reg_coef)
        if trace:
            hist['dual_gap'] += [dg]
            hist['elaps_t'] += [el_t]
            hist['phi'] += [phi_m]

        if disp:
            tpl = "iter. #{iter}: elaps_t={time: .{tol}f},\t" +\
                  "phi={phi: .{tol}f},\tdual_gap={dg: .{tol}f},\t" +\
                  "inner_it#={in_it}"
            print(tpl.format(iter=k_out, time=el_t, phi=phi_m,
                             dg=dg, in_it=k_inn, tol=disp_dig))

        if dg < tol:
            status = 0
            break

    if trace:
        hist['dual_gap'] = np.array(hist['dual_gap'])
        hist['elaps_t'] = np.array(hist['elaps_t'])
        hist['phi'] = np.array(hist['phi'])

        return (w_hat, status, hist)

    else:
        return (w_hat, status)


@degenerate_case
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
    n, d = X.shape
    w0 = w0.astype(np.float)
    w_hat = w0.copy()

    status = 1
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    hist = {'elaps_t': [], 'phi': [], 'dual_gap': []}

    start_ = time.time()

    subg = np.empty(w0.shape)
    min_phi = phi_main(w0, X, y, reg_coef)
    for k in range(max_iter):
        subg[w0 >= 0] = 1
        subg[w0 < 0] = -1
        g = 1 / n * X.T.dot(X.dot(w0) - y) + reg_coef * subg

        w0 -= alpha / np.sqrt(k + 1) * g / np.linalg.norm(g, ord=2)
        phi = phi_main(w0, X, y, reg_coef)
        if phi < min_phi:
            w_hat = w0.copy()
            min_phi = phi

        dg = duality_gap(w0, X, y, reg_coef)
        el_t = time.time() - start_

        if trace:
            hist['dual_gap'] += [dg]
            hist['elaps_t'] += [el_t]
            hist['phi'] += [phi]

        if disp:
            tpl = "iter. #{iter}: elaps_t={time: .{tol}f},\t" +\
                  "phi={phi: .{tol}f},\tdual_gap={dg: .{tol}f},\t"
            print(tpl.format(iter=k, time=el_t, phi=phi,
                             dg=dg, tol=disp_dig))

        if dg < tol:
            status = 0
            break

    if trace:
        hist['dual_gap'] = np.array(hist['dual_gap'])
        hist['elaps_t'] = np.array(hist['elaps_t'])
        hist['phi'] = np.array(hist['phi'])

        return (w_hat, status, hist)

    else:
        return (w_hat, status)


@degenerate_case
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
    L = L0
    n, d = X.shape
    w0 = w0.astype(np.float)

    status = 1
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    ls_iters = 0
    hist = {'elaps_t': [], 'phi': [], 'dual_gap': [], 'ls_iters': []}

    start_ = time.time()

    phi = phi_main(w0, X, y, reg_coef)
    for k in range(max_iter):
        nab_phi = 1 / n * X.T.dot(X.dot(w0) - y)
        while 7:
            ls_iters += 1
            delta = nab_phi - L * w0

            w = w0.copy()

            ind = delta > reg_coef
            w[ind] = (-delta[ind] + reg_coef) / L

            ind = delta < -reg_coef
            w[ind] = (-delta[ind] - reg_coef) / L

            w[np.abs(delta) <= reg_coef] = 0

            m = nab_phi.dot(w - w0) + L / 2 * ((w - w0) ** 2).sum()
            m += phi + reg_coef * np.linalg.norm(w, ord=1)

            if phi_main(w, X, y, reg_coef) > m:
                L /= 2
            else:
                break

        w0 = w.copy()
        phi = phi_main(w0, X, y, reg_coef)
        L = np.maximum(L0, L / 2)

        dg = duality_gap(w0, X, y, reg_coef)
        el_t = time.time() - start_

        if trace:
            hist['dual_gap'] += [dg]
            hist['elaps_t'] += [el_t]
            hist['phi'] += [phi]
            hist['ls_iters'] += [ls_iters]

        if disp:
            tpl = "iter. #{iter}: elaps_t={time: .{tol}f},\t" +\
                  "phi={phi: .{tol}f},\tdual_gap={dg: .{tol}f},\t" +\
                  "ls_iters#={in_it}"
            print(tpl.format(iter=k, time=el_t, phi=phi,
                             dg=dg, in_it=ls_iters, tol=disp_dig))

        if dg < tol:
            status = 0
            break

    if trace:
        hist['dual_gap'] = np.array(hist['dual_gap'])
        hist['elaps_t'] = np.array(hist['elaps_t'])
        hist['phi'] = np.array(hist['phi'])
        hist['ls_iters'] = np.array(hist['ls_iters'])

        return (w0, status, hist)

    else:
        return (w0, status)

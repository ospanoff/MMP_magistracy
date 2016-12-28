import numpy as np
from logutils import FuncSumWrapper, Logger


def sgd(fsum, x0, n_iters=1000, step_size=0.1, trace=False):
    """
    Метод SGD

    Параметры:
    ----------
    fsum: FuncSum
        Оптимизируемая функция, объект класса FuncSum.

    x0: np.ndarray
        Начальная точка x_0 \in R^d.

    n_iters: int, опционально
        Число итераций K метода.

    step_size: float, опционально
        Длина шага h.

    trace: bool, опционально
        Отслеживать прогресс метода для возврата истории или нет.

    Возврат:
    --------
    x_out: np.ndarray
        Приближенное решение (результат работы метода).

    hist: dict, возвращается только если trace=True
        История процесса оптимизации. Словарь со следующими полями:
            epoch: np.ndarray
                Номер эпохи: {сумм. число вызовов оракула} / {число функций n}.

            f: np.ndarray
                Значение полной функции.

            norm_g: np.ndarray
                (Бесконечная) норма градиента полной функции.
    """
    fsum = FuncSumWrapper(fsum)
    logger = Logger(fsum)
    x_sum = x0
    for k in range(n_iters):
        i = np.random.randint(fsum.n_funcs)
        f, g = fsum.call_ith(i, x0)
        x0 = x0 - step_size * g
        x_sum = x_sum + x0
        if trace:
            logger.record_point(x_sum / (k + 2))

    x_out = x_sum / (n_iters + 1)

    return (x_out, logger.get_hist()) if trace else x_out


def svrg(fsum, x0, n_stages=10, n_inner_iters=None,
         tol=1e-4, trace=False, L0=1, save_grad=False):
    """
    Метод SVRG с адаптивным подбором длины шага

    Параметры:
    ----------
    fsum: FuncSum
        Оптимизируемая функция, объект класса FuncSum.

    x0: np.ndarray
        Начальная точка x_0 \in R^d.

    n_stages: int, опционально
        Число стадий S.

    n_inner_iters: int или None, опционально
        Число внутренних итераций m. Если None, то взять равным 2n.

    tol: float, опционально
        Точность для критерия остановки: если ||g^~||_\inf < tol, то выход.

    trace: bool, опционально
        Отслеживать прогресс метода для возврата истории или нет.

    L0: float, опционально
        Начальная оценка константы Липшица.

    Возврат:
    --------
    x_out: np.ndarray
        Приближенное решение (результат работы метода).

    hist: dict, возвращается только если trace=True
        История процесса оптимизации. Словарь со следующими полями:
            epoch: np.ndarray
                Номер эпохи: {сумм. число вызовов оракула} / {число функций n}.

            f: np.ndarray
                Значение полной функции.

            norm_g: np.ndarray
                (Бесконечная) норма градиента полной функции.
    """
    fsum = FuncSumWrapper(fsum)
    logger = Logger(fsum)
    m = n_inner_iters if n_inner_iters else 2 * fsum.n_funcs
    x_s = x0
    L = L0
    for s in range(n_stages):
        if save_grad:
            g_s_all = [fsum.call_ith(i, x_s)[1] for i in range(fsum.n_funcs)]
            g_s = np.mean(g_s_all, axis=0)
        else:
            g_s = 0  # doesn't matter if np.zeros
            for i in range(fsum.n_funcs):
                g_s = g_s + fsum.call_ith(i, x_s)[1]
            g_s = g_s / fsum.n_funcs

        if np.linalg.norm(g_s, ord=np.inf) < tol:
            break

        x_k = x_s
        x_sum = x_k
        for k in range(m):
            i = np.random.randint(fsum.n_funcs)
            f, g = fsum.call_ith(i, x_k)
            if save_grad:
                g_k = g - g_s_all[i] + g_s
            else:
                g_k = g - fsum.call_ith(i, x_s)[1] + g_s

            while 7:
                x = x_k - (0.1 / L) * g_k
                f_, g_ = fsum.call_ith(i, x)
                if f_ <= f + g.dot(x - x_k) + 0.5 * L * np.sum((x - x_k) ** 2):
                    break

                L *= 2

            x_k = x
            x_sum = x_sum + x_k
            L = np.maximum(L0, L / (2 ** (1 / m)))

            if trace:
                logger.record_point(x_sum / (k + 2))

        x_s = x_sum / (m + 1)

    return (x_s, logger.get_hist()) if trace else x_s

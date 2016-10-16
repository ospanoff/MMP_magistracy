import numpy as np
from special import FuncWrapper
from scipy.optimize.linesearch import line_search_wolfe2
import time


def cg(matvec, b, x0, tol=1e-4, max_iter=None, disp=False, trace=False):
    """
    Метод сопряженных градиентов для решения системы линейных уравнений Ax = b

    Параметры:
    ----------
    matvec: callable matvec(x)
        Функция умножения матрицы системы на произвольный вектор x
        Принимает:
            x: np.ndarray
                Вектор размера n
        Возвращает:
            ax: np.ndarray
                Произведение матрицы системы на вектор x, вектор размера n
    b: np.ndarray
        Правая часть системы, вектор размера n

    x0: np.ndarray
        Начальная точка, вектор размера n

    tol: float, опционально
        Точность по l_∞-норме невязки: norm(A x_k - b, infty) <= tol

    max_iter: int или None, опционально
        Максимальное число итераций метода. Если None, то положить равным n

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, текущая
        норма невязки и пр.) или нет

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет

    Возврат:
    --------
    x_sol: np.ndarray
        Найденное решение системы, вектор размера n

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью tol
            1: достигнуто максимальное число итераций

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям. Словарь со следующими
        полями:
            norm_r: np.ndarray
                Норма невязки ||Ax_k − b||_∞ по итерациям

    """
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    status = 1

    g = matvec(x0) - b
    d = -g
    u = matvec(d)

    hist = {'norm_r': [np.linalg.norm(g, ord=np.inf)]}

    norm = g.dot(g)

    for k in range(b.size if max_iter is None else max_iter):
        alpha = norm / d.dot(u)

        x0 += alpha * d
        g += alpha * u

        norm_new = g.dot(g)

        norm_r = np.linalg.norm(g, ord=np.inf)

        if disp:
            tpl = "iter. #{iter}: ||r||={norm: .{tol}f}\t"
            print(tpl.format(iter=k, norm=norm_r, tol=disp_dig))

        if trace:
            hist['norm_r'] += [norm_r]

        if norm_r < tol:
            status = 0
            break

        beta = norm_new / norm

        d = -g + beta * d
        u = matvec(d)

        norm = norm_new

    if trace:
        hist['norm_r'] = np.array(hist['norm_r'])

        return (
            x0,
            status,
            hist
        )

    else:
        return (
            x0,
            status
        )


def ncg(func, x0, tol=1e-4, max_iter=500, c1=1e-4, c2=0.1, disp=False,
        trace=False):
    """
    Нелинейный метод сопряженных градиентов

    Параметры:
    ----------
    func: callable func(x)
        Оракул минимизируемой функции
        Принимает:
            x: np.ndarray
                Точка вычисления, вектора размера n
        Возвращает:
            f: float
                Значение функции в точке x
            g: np.ndarray
                Градиент функции в точке x, вектора размера n

    x0: np.ndarray
        Начальная точка, вектор размера n

    tol: float, опционально
        Точность по l_∞-норме градиента: norm(nabla f(x_k), infty) < tol

    max_iter: int, опционально
        Максимальное число итераций метода

    c1: float, опционально
        Значение константы c1 в условиях Вульфа

    c2: float, опционально
        Значение константы c2 в условиях Вульфа

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, число
        вызовов оракула, значение функции, норма градиента и пр.) или нет

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет

    Возврат:
    --------
    x_min: np.ndarray
        Найденная точка минимума, вектор размера n

    f_min: float
        Значение функции в точке x_min

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью tol
            1: достигнуто максимальное число итераций или вызовов оракула

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям. Словарь со следующими
        полями:
            f: np.ndarray
                Значение функции
            norm_g: np.ndarray
                l_∞-норма градиента
            n_evals: np.ndarray
                Суммарное число вызовов оракула на текущий момент
            elaps_t: np.ndarray
                Реальное время, пройденное с начала оптимизации

    """
    func_wrapper = FuncWrapper(func)

    disp_dig = int(np.fabs(np.log10(tol))) + 1

    f, g_old = func(x0)
    d = -g_old

    n_evals = 1
    status = 1
    hist = {
        'f': [f],
        'norm_g': [np.linalg.norm(g_old, ord=np.inf)],
        'n_evals': [n_evals],
        'elaps_t': [0]
    }
    start = time.time()

    for k in range(max_iter):
        alpha, fc = line_search_wolfe2(lambda x: func_wrapper(x)[0],
                                       lambda x: func_wrapper(x)[1],
                                       x0, d, g_old, c1=c2, c2=c2)[:2]

        x0 += alpha * d

        f, g = func(x0)
        n_evals += 1 + fc

        norm_g = np.linalg.norm(g, ord=np.inf)

        el_t = time.time() - start

        if disp:
            tpl = "iter. #{iter}: ||g||={norm: .{tol}f},\t" +\
                  "alpha={al: .{tol}f},\tn_evals={n_evals},\t" +\
                  "elaps_t={el_t: .{tol}f}"
            print(tpl.format(iter=k, norm=norm_g, al=alpha, n_evals=n_evals,
                             el_t=el_t, tol=disp_dig))

        if trace:
            hist['f'] += [f]
            hist['norm_g'] += [norm_g]
            hist['n_evals'] += [n_evals]
            hist['elaps_t'] += [el_t]

        if norm_g < tol:
            status = 0
            break

        beta = g.dot(g) / d.dot(g - g_old)

        d = -g + beta * d

        g_old = g

    if trace:
        hist['f'] = np.array(hist['f'])
        hist['norm_g'] = np.array(hist['norm_g'])
        hist['n_evals'] = np.array(hist['n_evals'])
        hist['elaps_t'] = np.array(hist['elaps_t'])

        return (
            x0,
            f,
            status,
            hist
        )

    else:
        return (
            x0,
            f,
            status
        )


def lbfgs_compute_dir(sy_hist, g):
    """
    (L-BFGS) Процедура нахождения направления dk

    Параметры:
    ----------
    sy_hist: collections.deque
        История Hk, структура данных «дек». Каждый элемент sy_hist[-i]
        (т. е. i-й справа) -- это пара (s_k−i, y_k−i) из двух элементов
        типа np.ndarray размера n

    g: np.ndarray
        Градиент grad(f(xk)) в текущей точке xk, n-мерный вектор

    Возврат:
    --------
    d: np.ndarray
        Направление спуска метода L-BFGS, вектор размера n

    """


def lbfgs(func, x0, tol=1e-4, max_iter=500, c1=1e-4, c2=0.9, m=10, disp=False,
          trace=False):
    """
    Метод LBFGS

    Параметры:
    ----------
    func: callable func(x)
        Оракул минимизируемой функции
        Принимает:
            x: np.ndarray
                Точка вычисления, вектора размера n
        Возвращает:
            f: float
                Значение функции в точке x
            g: np.ndarray
                Градиент функции в точке x, вектора размера n

    x0: np.ndarray
        Начальная точка, вектор размера n

    tol: float, опционально
        Точность по l_∞-норме градиента: norm(nabla f(x_k), infty) < tol

    max_iter: int, опционально
        Максимальное число итераций метода

    c1: float, опционально
        Значение константы c1 в условиях Вульфа

    c2: float, опционально
        Значение константы c2 в условиях Вульфа

    m: int, опционально
        параметр, указывающий размер истории, используемой методом
        (т.е. число хранимых пар (s_k, y_k))

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, число
        вызовов оракула, значение функции, норма градиента и пр.) или нет

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет

    Возврат:
    --------
    x_min: np.ndarray
        Найденная точка минимума, вектор размера n

    f_min: float
        Значение функции в точке x_min

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью tol
            1: достигнуто максимальное число итераций или вызовов оракула

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям. Словарь со следующими
        полями:
            f: np.ndarray
                Значение функции
            norm_g: np.ndarray
                l_∞-норма градиента
            n_evals: np.ndarray
                Суммарное число вызовов оракула на текущий момент
            elaps_t: np.ndarray
                Реальное время, пройденное с начала оптимизации

    """


def hfn(func, x0, hess_vec, tol=1e-4, max_iter=500, c1=1e-4, c2=0.9,
        disp=False, trace=False):
    """
    Неточный метод Ньютона

    Параметры:
    ----------
    func: callable func(x)
        Оракул минимизируемой функции
            Принимает:
                x: np.ndarray
                    Точка вычисления, вектор размера n
            Возвращает:
                f: float
                    Значение функции в точке x
                g: np.ndarray
                    Градиент функции в точке x, вектор размера n

    x0: np.ndarray
        Начальная точка, вектор размера n

    hess_vec: callable hess_vec(x, v)
        Функция умножения гессиана (или его аппроксимации) в точке x на
        произвольный вектор v
        Принимает:
            x: np.ndarray
                Точка вычисления, вектор размера n
            v: np.ndarray
                Вектор для умножения размера n
        Возвращает:
            hv: np.ndarray
                Произведение гессиана на вектор v, вектор размера n

    tol: float, опционально
        Точность по l_∞-норме градиента:
        norm(nabla f(x_k), infty) < tol

    max_iter: int, опционально
        Максимальное число итераций метода

    c1: float, опционально
        Значение константы c1 в условиях Вульфа

    c2: float, опционально
        Значение константы c2 в условиях Вульфа

    disp: bool, опционально
        Отображать прогресс метода по итерациям (номер итерации, число
        вызовов оракула, значение функции, норма градиента и пр.) или нет

    trace: bool, опционально
        Сохранять траекторию метода для возврата истории или нет

    Возврат:
    --------
    x_min: np.ndarray
        Найденная точка минимума, вектор размера n

    f_min: float
        Значение функции в точке x_min

    status: int
        Статус выхода, число:
            0: решение найдено с заданной точностью tol
            1: достигнуто максимальное число итераций

    hist: dict, возвращается только если trace=True
        История процесса оптимизации по итерациям. Словарь со следующими
        полями:
            f: np.ndarray
                Значение функции
            norm_g: np.ndarray
                l_∞-норма градиента
            n_evals: np.ndarray
                Суммарное число вызовов оракула (сумма func и hess_vec) на
                текущий момент
            elaps_t: np.ndarray
                Реальное время, пройденное с начала оптимизации

        """
    func_wrapper = FuncWrapper(func)

    disp_dig = int(np.fabs(np.log10(tol))) + 1

    f, g = func(x0)

    n_evals = 1
    status = 1
    norm_g = np.linalg.norm(g, ord=np.inf)
    hist = {
        'f': [f],
        'norm_g': [norm_g],
        'n_evals': [n_evals],
        'elaps_t': [0]
    }
    start = time.time()

    for k in range(max_iter):
        eps = np.minimum(0.5, np.sqrt(norm_g)) * norm_g

        d, _, h = cg(lambda v: hess_vec(x0, v), -g,
                     np.zeros(g.size), eps, trace=True)
        n_evals += h['norm_r'].size

        while d.dot(g) >= 0:
            eps /= 2
            d, _, h = cg(lambda v: hess_vec(x0, v), -g, d, eps, trace=True)[0]
            n_evals += h['norm_r'].size

        alpha, fc = line_search_wolfe2(lambda x: func_wrapper(x)[0],
                                       lambda x: func_wrapper(x)[1],
                                       x0, d, g, c1=c2, c2=c2)[:2]

        x0 += alpha * d

        f, g = func(x0)
        n_evals += 1 + fc

        norm_g = np.linalg.norm(g, ord=np.inf)

        el_t = time.time() - start

        if disp:
            tpl = "iter. #{iter}: ||g||={norm: .{tol}f},\t" +\
                  "alpha={al: .{tol}f},\tn_evals={n_evals},\t" +\
                  "elaps_t={el_t: .{tol}f}"
            print(tpl.format(iter=k, norm=norm_g, al=alpha, n_evals=n_evals,
                             el_t=el_t, tol=disp_dig))

        if trace:
            hist['f'] += [f]
            hist['norm_g'] += [norm_g]
            hist['n_evals'] += [n_evals]
            hist['elaps_t'] += [el_t]

        if norm_g < tol:
            status = 0
            break

    if trace:
        hist['f'] = np.array(hist['f'])
        hist['norm_g'] = np.array(hist['norm_g'])
        hist['n_evals'] = np.array(hist['n_evals'])
        hist['elaps_t'] = np.array(hist['elaps_t'])

        return (
            x0,
            f,
            status,
            hist
        )

    else:
        return (
            x0,
            f,
            status
        )

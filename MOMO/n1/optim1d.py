import numpy as np


def min_golden(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False):
    """
    Метод золотого сечения

    Параметры:
    ----------
        func: callable func(x)
            Оракул минимизируемой функции.
            Принимает:
                x: float
                    Точка вычисления.
            Возвращает:
                f: float
                    Значение функции в точке x.

        a: float
            Левая граница интервала оптимизации.

        b: float
            Правая граница интервала оптимизации.

        tol: float, опционально
            Точность оптимизации по аргументу: abs(x_k - x_opt) <= tol

        max_iter: int, опционально
            Максимальное число итераций метода.

        disp: bool, опционально
            Отображать прогресс метода по итерациям (номер итерации, значение
            функции, длина текущего интервала и пр.) или нет.

        trace: bool, опционально
            Сохранять ли траекторию метода для возврата истории или нет.

    Возврат:
    --------
        x_min: float
            Найденное приближение минимума x_opt.

        f_min: float
            Значение функции в x_min.

        status: int
            Статус выхода, число:
                0: минимум найден с заданной точностью tol;
                1: достигнуто максимальное число итераций.

        hist: dict, возвращается только если trace=True
            История процесса оптимизации по итерациям. Словарь со следующими
            полями:
                x: np.ndarray
                    Точки итерационного процесса.
                f: np.ndarray
                    Значения функции в точках x.

        n_evals: np.ndarray
            Суммарное число вызовов оракула на текущий момент.
    """

    x_l, x_r = a, b
    K = (5 ** 0.5 - 1) / 2
    I = K * (x_r - x_l)
    x_a, x_b = x_r - I, x_l + I
    f_a, f_b = func(x_a), func(x_b)

    status = 1
    n_evals = 2
    digits = int(np.abs(np.log10(tol)))
    hist = dict()
    hist['x'] = []
    hist['f'] = []

    for k in range(max_iter):
        I = K * I
        if f_a <= f_b:
            x_l, x_a, x_b, x_r = x_l, x_b - I, x_a, x_b
            f_a, f_b = func(x_a), f_a
            h = x_a, f_a

        else:
            x_l, x_a, x_b, x_r = x_a, x_b, x_a + I, x_r
            f_a, f_b = f_b, func(x_b)
            h = x_b, f_b

        if trace:
            hist['x'] += [round(h[0], digits)]
            hist['f'] += [round(h[1], digits)]

        if disp:
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\tI={3: .{4}f}"
            print(tpl.format(k + 1, h[0], h[1], I, digits))

        n_evals += 1

        if I < tol:
            status = 0
            f_min = min(f_a, f_b)
            x_min = x_a if f_a < f_b else x_b
            break

    ret = {
        "x_min": round(x_min, digits),
        "f_min": round(f_min, digits),
        "status": status,
        "n_evals": n_evals
    }
    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        ret["hist"] = hist

    return ret

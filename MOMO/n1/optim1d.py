import numpy as np


def p_min(x, x_l, x_r, f, f_l, f_r):
    return x - (
        ((x - x_l) ** 2 * (f - f_r) - (x - x_r) ** 2 * (f - f_l)) /
        ((x - x_l) * (f - f_r) - (x - x_r) * (f - f_l)) / 2
    )


def neq(a, b, c, tol):
        return np.all(np.abs([a - b, a - c, b - c]) > tol)


def min_golden(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False,
               rnd=False):
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

        rnd: bool, опционально
            Округлять ли результат до заданной точности tol

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

    status = 1
    disp_dig = int(np.abs(np.log10(tol)))
    digits = disp_dig if rnd else 20
    hist = {'x': [], 'f': [], 'n_evals': []}

    x_l, x_r = a, b
    K = (5 ** 0.5 - 1) / 2
    J = K * (x_r - x_l)
    x_a, x_b = x_r - J, x_l + J
    f_a, f_b = func(x_a), func(x_b)

    for k in range(max_iter):
        J *= K
        if f_a <= f_b:
            x_a, x_b, x_r = x_b - J, x_a, x_b
            f_a, f_b = func(x_a), f_a
            h = x_a, f_a

        else:
            x_l, x_a, x_b = x_a, x_b, x_a + J
            f_a, f_b = f_b, func(x_b)
            h = x_b, f_b

        if trace:
            hist['x'] += [round(h[0], digits)]
            hist['f'] += [round(h[1], digits)]
            # +2 for first oracul call before the loop
            hist['n_evals'] += [k + 1 + 2]

        if disp:
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\tI={3: .{4}f}"
            print(tpl.format(k + 1, h[0], h[1], J, disp_dig))

        x_min, f_min = h

        if J < tol:
            status = 0
            break

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(x_min, digits + 1),
            round(f_min, digits + 1),
            status,
            hist
        )

    else:
        return (
            round(x_min, digits + 1),
            round(f_min, digits + 1),
            status
        )


def min_parabolic(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False,
                  rnd=False):
    """
    Метод парабол

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

        rnd: bool, опционально
            Округлять ли результат до заданной точности tol

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

    status = 1
    disp_dig = int(np.abs(np.log10(tol)))
    digits = disp_dig if rnd else 20
    hist = {'x': [], 'f': [], 'n_evals': []}

    x_l, x_r = a, b
    x = (a + b) / 2
    f_l, f, f_r = func(x_l), func(x), func(x_r)

    for k in range(max_iter):
        if neq(x, x_l, x_r, tol) and neq(f, f_l, f_r, tol):
            u = p_min(x, x_l, x_r, f, f_l, f_r)

        else:
            status = 0  # if tol if too small

        if status == 0 or np.abs(u - x) < tol:
            status = 0
            break

        f_u = func(u)

        if trace:
            hist['x'] += [round(u, digits)]
            hist['f'] += [round(f_u, digits)]
            hist['n_evals'] += [k + 1 + 3]  # +3 before the loop

        if disp:
            tpl = ("iter. #{0}:\tx={1: .{5}f},\tf={2: .{5}f},\t" +
                   "x_l={3: .{5}f}\tx_r={4: .{5}f}")
            print(tpl.format(k + 1, u, f_u, x_l, x_r, disp_dig))

        if f_u <= f:
            if u <= x:
                x, x_r = u, x
                f, f_r = f_u, f

            else:
                x_l, x = x, u
                f_l, f = f, f_u

        else:
            if u <= x:
                x_l = u

            else:
                x_r = u

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(x, digits + 1),
            round(f, digits + 1),
            status,
            hist
        )

    else:
        return (
            round(x, digits + 1),
            round(f, digits + 1),
            status
        )


def min_brent(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False,
              rnd=False):
    """
    Комбинированный метод Брента

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

        rnd: bool, опционально
            Округлять ли результат до заданной точности tol

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

    status = 1
    disp_dig = int(np.abs(np.log10(tol)))
    digits = disp_dig if rnd else 20
    hist = {'x': [], 'f': [], 'n_evals': []}

    K = (3 - 5 ** 0.5) / 2
    x = w = v = a + K * (b - a)
    f_x = f_w = f_v = func(x)
    d = e = b - a

    for k in range(max_iter):
        g, e = e, d

        rtol = tol * np.abs(x) + tol / 100

        if d < tol:  # np.abs(x - (a + b) / 2) + (b - a) / 2 < 2 * rtol:
            status = 0
            break

        u_admitted = False

        if neq(x, w, v, tol) and neq(f_x, f_w, f_v, tol):
            u = p_min(x, w, v, f_x, f_w, f_v)

            if (a <= u and u <= b) and (np.abs(u - x) < g / 2):
                u_admitted = True
                if (u - a) < 2 * rtol or (b - u) < 2 * rtol:
                    u = x - np.sign(x - (a + b) / 2) * rtol

        if not u_admitted:
            if x < (b + a) / 2:
                e = b - x
                u = x + K * e

            else:
                e = x - a
                u = x - K * e

        d = np.abs(u - x)

        if d < rtol:
            u = x + np.sign(u - x) * rtol

        f_u = func(u)

        if trace:
            hist['x'] += [round(u, digits)]
            hist['f'] += [round(f_u, digits)]
            hist['n_evals'] += [k + 1 + 1]  # +1 before the loop

        if disp:
            method = "parabolic" if u_admitted else "golden"
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\t" + \
                  "dist={3: .{4}f}\t method={5}"
            print(tpl.format(k + 1, u, f_u, d, disp_dig, method))

        if f_u <= f_x:
            if u >= x:
                a = x

            else:
                b = x

            v, w, x = w, x, u
            f_v, f_w, f_x = f_w, f_x, f_u

        else:
            if u >= x:
                b = u

            else:
                a = u

            if f_u <= f_w or w == x:
                v, w = w, u
                f_v, f_w = f_w, f_u

            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(u, digits + 1),
            round(f_u, digits + 1),
            status,
            hist
        )

    else:
        return (
            round(u, digits + 1),
            round(f_u, digits + 1),
            status
        )


def min_secant(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False,
               rnd=False):
    """
    Метод секущих (поиск нуля производной)

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

                f': float
                    Значение производной функции в точке x.

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

        rnd: bool, опционально
            Округлять ли результат до заданной точности tol

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

    status = 1
    disp_dig = int(np.abs(np.log10(tol)))
    digits = disp_dig if rnd else 20
    hist = {'x': [], 'f': [], 'n_evals': []}

    f_a = func(a)
    f_b = func(b)
    x_min = 0

    for k in range(max_iter):
        u = b - f_b[1] * (b - a) / (f_b[1] - f_a[1])
        f = func(u)

        if trace:
            hist['x'] += [round(u, digits)]
            hist['f'] += [round(f[0], digits)]
            # +2 for the first oracul calls before the loop
            hist['n_evals'] += [k + 1 + 2]

        if disp:
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\tf'={3: .{4}f}"
            print(tpl.format(k + 1, u, f[0], f[1], disp_dig))

        if np.abs(x_min - u) < tol:
            status = 0
            x_min = u
            break

        if np.sign(f[1]) == np.sign(f_a[1]):
            a = u
            f_a = f

        else:
            b = u
            f_b = f

        x_min = u

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(x_min, digits + 1),
            round(func(x_min)[0], digits + 1),
            status,
            hist
        )

    else:
        return (
            round(x_min, digits + 1),
            round(func(x_min)[0], digits + 1),
            status
        )


def min_brent_der(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False,
                  rnd=False):
    """
    Комбинированный метод Брента c производной

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

                f': float
                    Значение производной функции в точке x.

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

        rnd: bool, опционально
            Округлять ли результат до заданной точности tol

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

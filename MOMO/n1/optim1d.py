import numpy as np


def p_min(x, x_l, x_r, f, f_l, f_r):
    return x - (
        ((x - x_l) ** 2 * (f - f_r) - (x - x_r) ** 2 * (f - f_l)) /
        ((x - x_l) * (f - f_r) - (x - x_r) * (f - f_l)) / 2
    )


def neq(a, b, c, tol=1e-20):
        return np.all(np.fabs([a - b, a - c, b - c]) >= tol)


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
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    digits = disp_dig if rnd else 20

    x_l, x_r = a, b
    K = (5 ** 0.5 - 1) / 2
    J = K * (x_r - x_l)
    x_a, x_b = x_r - J, x_l + J
    f_a, f_b = func(x_a), func(x_b)

    h = (x_a, f_a) if f_a <= f_b else (x_b, f_b)
    hist = {'x': [h[0]], 'f': [h[1]], 'n_evals': [2]}

    for k in range(max_iter):
        J *= K
        if f_a <= f_b:
            x_a, x_b, x_r = x_b - J, x_a, x_b
            f_a, f_b = func(x_a), f_a

        else:
            x_l, x_a, x_b = x_a, x_b, x_a + J
            f_a, f_b = f_b, func(x_b)

        h = (x_a, f_a) if f_a <= f_b else (x_b, f_b)

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
            round(x_min, digits),
            round(f_min, digits),
            status,
            hist
        )

    else:
        return (
            round(x_min, digits),
            round(f_min, digits),
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
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    digits = disp_dig if rnd else 20

    x_l, x_r = a, b
    x = (a + b) / 2
    f_l, f, f_r = func(x_l), func(x), func(x_r)

    hist = {'x': [x], 'f': [f], 'n_evals': [3]}

    for k in range(max_iter):
        if neq(x, x_l, x_r) and neq(f, f_l, f_r):
            u = p_min(x, x_l, x_r, f, f_l, f_r)
        else:
            status = 0  # if tol if too small

        rtol = tol * np.fabs(x) + tol / 10
        if status == 0 or \
           np.fabs(x - (x_l + x_r) / 2) + (x_r - x_l) / 2 <= 2 * rtol:
            status = 0
            break

        f_u = func(u)

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
                f_l = f_u

            else:
                x_r = u
                f_r = f_u

        if trace:
            hist['x'] += [round(x, digits)]
            hist['f'] += [round(f, digits)]
            hist['n_evals'] += [k + 1 + 3]  # +3 before the loop

        if disp:
            tpl = ("iter. #{0}:\tx={1: .{5}f},\tf={2: .{5}f},\t" +
                   "x_l={3: .{5}f},\tx_r={4: .{5}f}")
            print(tpl.format(k + 1, x, f, x_l, x_r, disp_dig))

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(x, digits),
            round(f, digits),
            status,
            hist
        )

    else:
        return (
            round(x, digits),
            round(f, digits),
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
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    digits = disp_dig if rnd else 20

    K = (3 - 5 ** 0.5) / 2
    x = w = v = a + K * (b - a)
    f_x = f_w = f_v = func(x)
    d = e = b - a

    hist = {'x': [x], 'f': [f_x], 'n_evals': [1]}

    for k in range(max_iter):
        g, e = e, d

        if d < tol:
            status = 0
            break

        u_admitted = False

        if neq(x, w, v) and neq(f_x, f_w, f_v):
            u = p_min(x, w, v, f_x, f_w, f_v)

            if a <= u <= b and np.fabs(u - x) < g / 2:
                u_admitted = True

        if not u_admitted:
            if x < (b + a) / 2:
                e = b - x
                u = x + K * e

            else:
                e = x - a
                u = x - K * e

        d = np.fabs(u - x)

        f_u = func(u)

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
            hist['x'] += [round(x, digits)]
            hist['f'] += [round(f_x, digits)]
            hist['n_evals'] += [k + 1 + 1]  # +1 before the loop

        if disp:
            method = "parabolic" if u_admitted else "golden"
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\t" + \
                  "dist={3: .{4}f},\t method={5}"
            print(tpl.format(k + 1, x, f_x, d, disp_dig, method))

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(u, digits),
            round(f_u, digits),
            status,
            hist
        )

    else:
        return (
            round(u, digits),
            round(f_u, digits),
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
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    digits = disp_dig if rnd else 20
    hist = {'x': [], 'f': [], 'n_evals': []}

    f_a = func(a)
    f_b = func(b)

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

        if np.fabs(f[1]) < tol:
            status = 0
            break

        if (b - u) * f[1] <= 0:  # np.sign(f[1]) == np.sign(f_a[1]):
            a = u
            f_a = f

        else:
            b = u
            f_b = f

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(u, digits),
            round(f[0], digits),
            status,
            hist
        )

    else:
        return (
            round(u, digits),
            round(f[0], digits),
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

    status = 1
    disp_dig = int(np.fabs(np.log10(tol))) + 1
    digits = disp_dig if rnd else 20

    x = w = v = (a + b) / 2
    fdf_x = func(x)
    f_x = f_w = f_v = fdf_x[0]
    df_x = df_w = df_v = fdf_x[1]

    d = e = b - a

    hist = {'x': [x], 'f': [f_x], 'n_evals': [1]}

    for k in range(max_iter):
        g, e = e, d

        if d < tol:
            status = 0
            break

        u1_admitted = False
        u2_admitted = False

        if np.fabs(x - w) > tol and np.fabs(df_x - df_w) > tol:
            u_1 = w - df_w * (w - x) / (df_w - df_x)

            if a <= u_1 <= b and (u_1 - x) * df_x <= 0 and \
               np.fabs(u_1 - x) < g / 2:
                u1_admitted = True

        if np.fabs(x - v) > tol and np.fabs(df_x - df_v) > tol:
            u_2 = v - df_v * (v - x) / (df_v - df_x)

            if a <= u_2 <= b and (u_2 - x) * df_x <= 0 and \
               np.fabs(u_2 - x) < g / 2:
                u2_admitted = True

        if u1_admitted and u2_admitted:
            u = u_1 if np.fabs(u_1 - x) < np.fabs(u_2 - x) else u_2

        elif u1_admitted:
            u = u_1

        elif u2_admitted:
            u = u_2

        else:
            if df_x > 0:
                u = (a + x) / 2
                e = x - a

            else:
                u = (x + b) / 2
                e = b - x

        d = np.fabs(x - u)

        f_u, df_u = func(u)

        if f_u <= f_x:
            if u >= x:
                a = x

            else:
                b = x

            v, w, x = w, x, u
            f_v, f_w, f_x = f_w, f_x, f_u
            df_v, df_w, df_x = df_w, df_x, df_u

        else:
            if u >= x:
                b = u

            else:
                a = u

            if f_u <= f_w or w == x:
                v, w = w, u
                f_v, f_w = f_w, f_u
                df_v, df_w = df_w, df_u

            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u
                df_v = df_u

        if trace:
            hist['x'] += [round(x, digits)]
            hist['f'] += [round(f_x, digits)]
            hist['n_evals'] += [k + 1 + 1]  # +1 before the loop

        if disp:
            method = "parabolic" if u1_admitted or u2_admitted else "bisection"
            tpl = "iter. #{0}:\tx={1: .{4}f},\tf={2: .{4}f},\t" + \
                  "f'={6: .{4}f},\tdist={3: .{4}f},\t method={5}"
            print(tpl.format(k + 1, x, f_x, d, disp_dig, method, df_u))

    if trace:
        hist['x'] = np.array(hist['x'])
        hist['f'] = np.array(hist['f'])
        hist['n_evals'] = np.array(hist['n_evals'])

        return (
            round(u, digits),
            round(f_u, digits),
            status,
            hist
        )

    else:
        return (
            round(u, digits),
            round(f_u, digits),
            status
        )

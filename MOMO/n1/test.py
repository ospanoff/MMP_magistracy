import unittest
import sys
import numpy as np
from optim1d import min_brent, min_golden, min_parabolic,\
    min_secant, min_brent_der


def fun1(x):
        return -5 * (x ** 5) + 4 * (x ** 4) -\
            12 * (x ** 3) + 11 * (x ** 2) - 2 * x + 1


def fun2(x):
    return -np.log(x - 2) ** 2 + np.log(10 - x) ** 2 - x ** 0.2


def fun3(x):
    return -3 * x * np.sin(0.75 * x) + np.exp(-2 * x)


def fun4(x):
    return np.exp(3 * x) + 5 * np.exp(-2 * x)


def fun5(x):
    return 0.2 * x * np.log(x) + (x - 2.3) ** 2


def fun1_der(x):
        return fun1(x), -25 * (x ** 4) + 16 * (x ** 3) -\
            36 * (x ** 2) + 22 * x - 2


def fun2_der(x):
    return fun2(x), -0.2 / (x ** 0.8) - 2 * np.log(10 - x) / (10 - x) -\
        2 * np.log(x - 2) / (x - 2)


def fun3_der(x):
    return fun3(x), -2 * np.exp(-2 * x) - 3 * np.sin(0.75 * x) -\
        2.25 * x * np.cos(0.75 * x)


def fun4_der(x):
    return fun4(x), np.exp(-2 * x) * (3 * np.exp(5 * x) - 10)


def fun5_der(x):
    return fun5(x), 2 * (x + 0.1 * np.log(x) - 2.2)


fun = [fun1, fun2, fun3, fun4, fun5]
fun_der = [fun1_der, fun2_der, fun3_der, fun4_der, fun5_der]

ans = [{'min': 0.8976329719, 'x': 0.1098599151, 'a': -0.5, 'b': 0.5},
       {'min': -5.4059614015, 'x': 9.2062432432, 'a': 6, 'b': 9.9},
       {'min': -7.2743579701, 'x': 2.7064755821, 'a': 0, 'b': 2 * np.pi},
       {'min': 5.1483404214, 'x': 0.2407945677, 'a': 0, 'b': 1},
       {'min': 0.3509778801, 'x': 2.1246397718, 'a': 0.5, 'b': 2.5}]

digits = 5
tol = 1e-6


class TestOptim1d(unittest.TestCase):
    def assertAlmostEqual(self, a, b, digits, msg):
        a = round(a, digits)
        b = round(b, digits)
        return self.assertEqual(a, b, msg)

    def test_golden(self):
        for f, a in zip(fun, ans):
            x_min, f_min, status = min_golden(f, a['a'], a['b'], tol=tol)
            msg = "on func " + f.__name__
            self.assertAlmostEqual(x_min, a['x'], digits, msg)
            self.assertAlmostEqual(f_min, a['min'], digits, msg)

    def test_parabolic(self):
        for f, a in zip(fun, ans):
            x_min, f_min, status = min_parabolic(f, a['a'], a['b'], tol=tol)
            msg = "on func " + f.__name__
            self.assertAlmostEqual(x_min, a['x'], digits, msg)
            self.assertAlmostEqual(f_min, a['min'], digits, msg)

    def test_brent(self):
        for f, a in zip(fun, ans):
            x_min, f_min, status = min_brent(f, a['a'], a['b'], tol=tol)
            msg = "on func " + f.__name__
            self.assertAlmostEqual(x_min, a['x'], digits, msg)
            self.assertAlmostEqual(f_min, a['min'], digits, msg)

    def test_secant(self):
        for f, a in zip(fun_der, ans):
            x_min, f_min, status = min_secant(f, a['a'], a['b'], tol=tol)
            msg = "on func " + f.__name__
            self.assertAlmostEqual(x_min, a['x'], digits, msg)
            self.assertAlmostEqual(f_min, a['min'], digits, msg)

    def test_brent_der(self):
        for f, a in zip(fun_der, ans):
            x_min, f_min, status = min_brent_der(f, a['a'], a['b'], tol=tol)
            msg = "on func " + f.__name__
            self.assertAlmostEqual(x_min, a['x'], digits, msg)
            self.assertAlmostEqual(f_min, a['min'], digits, msg)


if __name__ == '__main__':
    try:
        digits = int(sys.argv[1])
        tol = int(sys.argv[2])
        tol = 1 / 10 ** tol
    except:
        pass

    print("checking up to %s digits" % digits)
    print("computing with %s tol" % tol)

    unittest.main(argv=[""])

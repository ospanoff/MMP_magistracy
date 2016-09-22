import unittest
import numpy as np
from optim1d import min_brent, min_golden, min_parabolic


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


fun = [fun1, fun2, fun3, fun4, fun5]

ans = [{'min': 0.897633, 'x': 0.10986, 'a': -0.5, 'b': 0.5},
       {'min': -5.40596, 'x': 9.20624, 'a': 6, 'b': 9.9},
       {'min': -7.27436, 'x': 2.70648, 'a': 0, 'b': 2 * np.pi},
       {'min': 5.148340, 'x': 0.24079, 'a': 0, 'b': 1},
       {'min': 0.350978, 'x': 2.12464, 'a': 0.5, 'b': 2.5}]

digits = 4
tol = 1e-10


class TestOptim1d(unittest.TestCase):
    def assertAlmostEqual(self, a, b, digits):
        a = round(a, digits)
        b = round(b, digits)
        return self.assertEqual(a, b)

    def test_golden(self):
        for i in range(len(fun)):
            res = min_golden(fun[i], ans[i]['a'], ans[i]['b'], tol=tol)
            self.assertAlmostEqual(res['x_min'], ans[i]['x'], digits)
            self.assertAlmostEqual(res['f_min'], ans[i]['min'], digits)

    def test_parabolic(self):
        for i in range(len(fun)):
            res = min_parabolic(fun[i], ans[i]['a'], ans[i]['b'], tol=tol)
            self.assertAlmostEqual(res['x_min'], ans[i]['x'], digits)
            self.assertAlmostEqual(res['f_min'], ans[i]['min'], digits)

    def test_brent(self):
        for i in range(len(fun)):
            res = min_brent(fun[i], ans[i]['a'], ans[i]['b'], tol=tol)
            self.assertAlmostEqual(res['x_min'], ans[i]['x'], digits)
            self.assertAlmostEqual(res['f_min'], ans[i]['min'], digits)


if __name__ == '__main__':
    unittest.main()

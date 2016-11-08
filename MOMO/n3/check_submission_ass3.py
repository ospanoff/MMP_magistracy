import numpy as np
from scipy.special import expit
from collections import deque
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_array_equal
import unittest

from io import StringIO
import sys

from l1linreg import subgrad, barrier, prox_grad

############################################################################################################
# Check if it's Python 3
if not sys.version_info > (3, 0):
    print('You should use only Python 3!')
    sys.exit()

############################################################################################################
######################################### Auxiliary functions ##############################################
############################################################################################################

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

def duality_gap(w, X, y, reg_coef):
    n = X.shape[0]
    r = X.dot(w) - y
    mu = min(1, (reg_coef*n)/norm(X.T.dot(r), np.inf)) * (1/n)*r
    eta = (1/(2*n)) * r.dot(r) + reg_coef*norm(w, 1) + (n/2)*mu.dot(mu) + y.dot(mu)
    return eta

############################################################################################################
########################################### Data for testing ###############################################
############################################################################################################

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, -1, 1, 0])
reg_coef = 0.01
w0 = np.array([1, 1])
w0_plus = np.array([1, 1])
w0_minus = np.array([2, 2])

############################################################################################################
############################################# TestBarrier ##################################################
############################################################################################################

class TestBarrier(unittest.TestCase):    
    ################################ Tests #######################################
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            w_hat, status = barrier(X, y, reg_coef, w0_plus, w0_minus)

        assert_equal(status, 0)
        self.assertTrue(duality_gap(w_hat, X, y, reg_coef) <= 1e-5)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-7)

    def test_tol_inner(self):
        """Check if argument `tol_inner` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, tol_inner=1e-5)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, max_iter=15)

    def test_max_iter_inner(self):
        """Check if argument `max_iter_inner` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, max_iter_inner=15)

    def test_t0(self):
        """Check if argument `t0` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, t0=100)

    def test_gamma(self):
        """Check if argument `gamma` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, gamma=10)

    def test_c1(self):
        """Check if argument `c1` is supported."""
        barrier(X, y, reg_coef, w0_plus, w0_minus, c1=0.2)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            barrier(X, y, reg_coef, w0_plus, w0_minus, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        w_hat, status, hist = barrier(X, y, reg_coef, w0_plus, w0_minus, trace=True)

        self.assertTrue(isinstance(hist['phi'], np.ndarray))
        self.assertTrue(isinstance(hist['dual_gap'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['dual_gap']), len(hist['phi']))
        assert_equal(len(hist['elaps_t']), len(hist['phi']))

        # Make sure duality gap is a positive number
        self.assertTrue(np.all(hist['dual_gap'] >= 0))

        # Make sure `hist['elaps_t']` is a cumulative sum
        self.assertTrue(np.all(hist['elaps_t'] >= 0))
        self.assertTrue(np.all(hist['elaps_t'][1:] - hist['elaps_t'][:-1] >= 0))


############################################################################################################
############################################# TestSubgrad ##################################################
############################################################################################################

class TestSubgrad(unittest.TestCase):
    ################################ Tests #######################################
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            w_hat, status = subgrad(X, y, reg_coef, w0)

        assert_equal(status, 0)
        self.assertTrue(duality_gap(w_hat, X, y, reg_coef) <= 1e-2)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        subgrad(X, y, reg_coef, w0, tol=1e-7)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        subgrad(X, y, reg_coef, w0, max_iter=15)

    def test_alpha(self):
        """Check if argument `alpha` is supported."""
        subgrad(X, y, reg_coef, w0, alpha=0.1)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            subgrad(X, y, reg_coef, w0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        w_hat, status, hist = subgrad(X, y, reg_coef, w0, trace=True)

        self.assertTrue(isinstance(hist['phi'], np.ndarray))
        self.assertTrue(isinstance(hist['dual_gap'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['dual_gap']), len(hist['phi']))
        assert_equal(len(hist['elaps_t']), len(hist['phi']))

        # Make sure duality gap is a positive number
        self.assertTrue(np.all(hist['dual_gap'] >= 0))

        # Make sure `hist['elaps_t']` is a cumulative sum
        self.assertTrue(np.all(hist['elaps_t'] >= 0))
        self.assertTrue(np.all(hist['elaps_t'][1:] - hist['elaps_t'][:-1] >= 0))

############################################################################################################
############################################ TestProxGrad ##################################################
############################################################################################################

class TestProxGrad(unittest.TestCase):
    ################################ Tests #######################################
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            w_hat, status = prox_grad(X, y, reg_coef, w0)

        assert_equal(status, 0)
        self.assertTrue(duality_gap(w_hat, X, y, reg_coef) <= 1e-5)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        prox_grad(X, y, reg_coef, w0, tol=1e-7)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        prox_grad(X, y, reg_coef, w0, max_iter=15)

    def test_L0(self):
        """Check if argument `L0` is supported."""
        prox_grad(X, y, reg_coef, w0, L0=1e-2)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            prox_grad(X, y, reg_coef, w0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        w_hat, status, hist = prox_grad(X, y, reg_coef, w0, trace=True)

        self.assertTrue(isinstance(hist['phi'], np.ndarray))
        self.assertTrue(isinstance(hist['dual_gap'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))
        self.assertTrue(isinstance(hist['ls_iters'], np.ndarray))

        assert_equal(len(hist['dual_gap']), len(hist['phi']))
        assert_equal(len(hist['elaps_t']), len(hist['phi']))
        assert_equal(len(hist['ls_iters']), len(hist['phi']))

        # Make sure duality gap is a positive number
        self.assertTrue(np.all(hist['dual_gap'] >= 0))

        # Make sure `hist['elaps_t']` is a cumulative sum
        self.assertTrue(np.all(hist['elaps_t'] >= 0))
        self.assertTrue(np.all(hist['elaps_t'][1:] - hist['elaps_t'][:-1] >= 0))

        # Make sure `hist['ls_iters']` is a cumulative sum
        self.assertTrue(np.all(hist['ls_iters'] >= 0))
        self.assertTrue(np.all(hist['ls_iters'][1:] - hist['ls_iters'][:-1] >= 0))

############################################################################################################
################################################## Main ####################################################
############################################################################################################

if __name__ == '__main__':
    unittest.main()

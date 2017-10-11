#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of activation functions used within neural networks


import numpy as np


class BaseActivationFunction(object):
    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within '
                                  'child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within '
                                  'child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within '
                                  'child class!')


class LinearActivationFunction(BaseActivationFunction):
    def val(self, X):
        return X

    def deriv(self, X):
        return np.ones_like(X)

    def second_deriv(self, X):
        return np.zeros_like(X)


class SigmoidActivationFunction(BaseActivationFunction):
    def val(self, X):
        return 1 / (1 + np.exp(-X))

    def deriv(self, X):
        val = self.val(X)
        return val * (1 - val)

    def second_deriv(self, X):
        val = self.val(X)
        return val * (1 - val) * (1 - 2 * val)


class ReluActivationFunction(BaseActivationFunction):
    def val(self, X):
        return np.maximum(0, X)

    def deriv(self, X):
        return (X > 0).astype(X.dtype)

    def second_deriv(self, X):
        return np.zeros_like(X)


class LeakyReluActivationFunction(BaseActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def val(self, X):
        return np.maximum(X, self.alpha * X)

    def deriv(self, X):
        deriv = np.ones_like(X)
        deriv[X < 0] = self.alpha
        return deriv

    def second_deriv(self, X):
        return np.zeros_like(X)

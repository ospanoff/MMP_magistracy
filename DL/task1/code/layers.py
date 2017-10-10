#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of layers used within neural networks

import numpy as np


class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to
        layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer

        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it
        to layer parameters direction vector in convenient shape,
        e.g. matrix shape for fully-connected layer

        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within
        layer parameters.

        :param inputs: input batch, numpy matrix of size
            num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size
            num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within
        layer parameters.

        :param derivs: loss derivatives w.r.t. layer outputs,
            numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs,
            numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters,
            numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within
        layer parameters.

        :param Rp_inputs: Rp input batch, numpy matrix of size
            num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size
            num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.

        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs,
            numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs,
            numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters,
            numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of
            size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within'
                                  'child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of
            BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.afun = afun
        self.shape = shape
        self.theory_shape = (self.shape[1], self.shape[0])
        self.use_bias = use_bias

        self.W = np.random.normal(scale=0.05, size=self.theory_shape)
        if self.use_bias:
            self.bias = np.zeros(self.W.shape[0])
            self.W = np.hstack((self.W, self.bias[:, np.newaxis]))

    @property
    def inputs(self):
        return self.z_prev

    def get_params_number(self):
        return np.prod(self.W.shape)

    def get_weights(self):
        return self.W.ravel()

    def set_weights(self, w):
        self.W = w.reshape(self.W.shape)

    def set_direction(self, p):
        self.P = p.reshape(self.W.shape)

    def forward(self, inputs):
        self.z_prev = inputs
        if self.use_bias:
            self.z_prev = np.vstack((inputs, np.ones(inputs.shape[1])))

        self.u = self.W.dot(self.z_prev)
        self.z = self.afun.val(self.u)
        return self.z

    def backward(self, derivs):
        self.z_derivs = derivs
        self.deriv_u = self.afun.deriv(self.u)
        self.u_derivs = derivs * self.deriv_u
        w_derivs = self.u_derivs.dot(self.z_prev.T).ravel()
        input_derivs = self.W[:, :self.W.shape[1] - self.use_bias]
        input_derivs = input_derivs.T.dot(self.u_derivs)
        return input_derivs, w_derivs

    def Rp_forward(self, Rp_inputs):
        self.Rp_z_prev = Rp_inputs
        if self.use_bias:
            Rp_sh = Rp_inputs.shape[1]
            self.Rp_z_prev = np.vstack((Rp_inputs, np.zeros(Rp_sh)))

        self.Rp_u = self.W.dot(self.Rp_z_prev) + self.P.dot(self.z_prev)
        self.Rp_z_prev = Rp_inputs
        return self.deriv_u * self.Rp_u

    def Rp_backward(self, Rp_derivs):
        Rp_u_derivs = Rp_derivs * self.deriv_u +\
            self.z_derivs * self.afun.second_deriv(self.u) * self.Rp_u

        input_Rp_derivs =\
            self.P[:, :self.P.shape[1] - self.use_bias].T.dot(self.u_derivs) +\
            self.W[:, :self.W.shape[1] - self.use_bias].T.dot(Rp_u_derivs)

        w_Rp_derivs = Rp_u_derivs.dot(self.z_prev.T) +\
            self.u_derivs.dot(self.Rp_z_prev.T)

        return input_Rp_derivs, w_Rp_derivs.T.ravel()

    def get_activations(self):
        return self.z

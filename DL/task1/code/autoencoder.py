#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

import ffnet
import numpy as np


class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and '
                             'outputs is different!')

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of
        data

        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """
        N = inputs.shape[1]
        diff = self.net.compute_outputs(inputs) - inputs
        return (
            0.5 * (diff ** 2).sum() / N,
            self.net.compute_loss_grad(diff / N)
        )

    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector

        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        return self.net.compute_loss_Rp_grad(
            loss_Rp_derivs=self.net.compute_Rp_outputs()
        )

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given
        direction vector

        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        q = self.net.compute_Rp_outputs()
        G = self.net.get_activations(-1).T.dot(q).mean(axis=0)
        return self.net.compute_loss_grad(G)

    def run_sgd(self, inputs, step_size=0.01, momentum=0.9, num_epoch=200,
                minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                display=False):
        """
        Stochastic gradient descent optimization

        :param inputs: training sample, numpy matrix of
            size num_features x num_objects
        :param step_size: step size, number
        :param momentum: momentum coefficient, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of
            size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs,
            the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each
                epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each
                epoch, list
        """
        if display:
            s = "Epoch: Loss \t Norm of loss_grad"
            if test_inputs is not None:
                s += " \t Test loss \t Norm of test loss_grad"
            print(s)

        w = self.net.get_weights()
        v = np.zeros_like(w)

        result = {'train_loss': [], 'train_grad': [],
                  'test_loss': [], 'test_grad': []}

        for k in range(num_epoch):
            epochHist = []
            for indx in get_batch_idx(inputs.shape[1], minibatch_size):
                loss, loss_grad = self.compute_loss(
                    inputs[:, indx]
                )
                v = momentum * v - step_size * (loss_grad + l2_coef * w)
                w += v

                epochHist += [[loss, np.linalg.norm(loss_grad)]]
                self.net.set_weights(w)

            hist = np.mean(epochHist, axis=0).tolist()

            result['train_loss'] += [hist[0]]
            result['train_grad'] += [hist[1]]
            if test_inputs is not None:
                loss, loss_grad = self.compute_loss(test_inputs)
                hist += [loss, np.linalg.norm(loss_grad)]
                result['test_loss'] += [hist[2]]
                result['test_grad'] += [hist[3]]

            if display:
                s = "#{}: {}\t{}"
                if test_inputs is not None:
                    s += "\t{}\t{}"
                print(s.format(k + 1, *hist))

        return result

    def run_rmsprop(self, inputs, step_size=0.01, num_epoch=200,
                    minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                    display=False):
        """
        RMSprop stochastic optimization

        :param inputs: training sample, numpy matrix of
            size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of
            size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs,
            the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each
                epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each
                epoch, list
        """
        if display:
            s = "Epoch: Loss \t Norm of loss_grad"
            if test_inputs is not None:
                s += " \t Test loss \t Norm of test loss_grad"
            print(s)

        w = self.net.get_weights()
        v = np.zeros_like(w.shape)  # (w ** 2).mean()
        gamma = 0.9
        eps = 1e-8

        result = {'train_loss': [], 'train_grad': [],
                  'test_loss': [], 'test_grad': []}

        for k in range(num_epoch):
            epochHist = []
            for indx in get_batch_idx(inputs.shape[1], minibatch_size):
                loss, loss_grad = self.compute_loss(
                    inputs[:, indx]
                )
                loss_grad += l2_coef * w
                v = gamma * v + (1 - gamma) * (loss_grad ** 2)
                w -= step_size * (loss_grad) / np.sqrt(v + eps)

                epochHist += [[loss, np.linalg.norm(loss_grad)]]
                self.net.set_weights(w)

            hist = np.mean(epochHist, axis=0).tolist()

            result['train_loss'] += [hist[0]]
            result['train_grad'] += [hist[1]]
            if test_inputs is not None:
                loss, loss_grad = self.compute_loss(test_inputs)
                hist += [loss, np.linalg.norm(loss_grad)]
                result['test_loss'] += [hist[2]]
                result['test_grad'] += [hist[3]]

            if display:
                s = "#{}: {}\t{}"
                if test_inputs is not None:
                    s += "\t{}\t{}"
                print(s.format(k + 1, *hist))

        return result

    def run_adam(self, inputs, step_size=0.01, num_epoch=200,
                 minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                 display=False):
        """
        ADAM stochastic optimization

        :param inputs: training sample, numpy matrix of
            size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of
            size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs,
            the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each
                epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each
                epoch, list
        """
        if display:
            s = "Epoch: Loss \t Norm of loss_grad"
            if test_inputs is not None:
                s += " \t Test loss \t Norm of test loss_grad"
            print(s)

        w = self.net.get_weights()
        m, v = 0, 0
        beta1, beta2 = 0.9, 0.99
        eps = 1e-8
        t = 0

        result = {'train_loss': [], 'train_grad': [],
                  'test_loss': [], 'test_grad': []}

        for k in range(num_epoch):
            epochHist = []
            for indx in get_batch_idx(inputs.shape[1], minibatch_size):
                t += 1
                loss, loss_grad = self.compute_loss(
                    inputs[:, indx]
                )
                loss_grad += l2_coef * w
                m = beta1 * m + (1 - beta1) * loss_grad
                v = beta2 * v + (1 - beta2) * (loss_grad ** 2)
                m_k = m / (1 - beta1 ** t)
                v_k = v / (1 - beta2 ** t)
                w -= step_size * m_k / np.sqrt(v_k + eps)

                epochHist += [[loss, np.linalg.norm(loss_grad)]]
                self.net.set_weights(w)

            hist = np.mean(epochHist, axis=0).tolist()

            result['train_loss'] += [hist[0]]
            result['train_grad'] += [hist[1]]
            if test_inputs is not None:
                loss, loss_grad = self.compute_loss(test_inputs)
                hist += [loss, np.linalg.norm(loss_grad)]
                result['test_loss'] += [hist[2]]
                result['test_grad'] += [hist[3]]

            if display:
                s = "#{}: {}\t{}"
                if test_inputs is not None:
                    s += "\t{}\t{}"
                print(s.format(k + 1, *hist))

        return result


def get_batch_idx(data_size, batch_size):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    for i in range(0, data_size, batch_size):
        yield indices[i: i + batch_size]

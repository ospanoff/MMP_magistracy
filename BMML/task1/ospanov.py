import numpy as np
from scipy import stats


def expectation(p, x):
    return np.sum(p * x)


def variance(p, x):
    return expectation(p, x ** 2) - expectation(p, x) ** 2


def pa(params, model):
    a = np.arange(params['amin'], params['amax'] + 1)
    p = 1 / (params['amax'] - params['amin'] + 1) * np.ones(a.size)
    return p, a


def pb(params, model):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    p = 1 / (params['bmax'] - params['bmin'] + 1) * np.ones(b.size)
    return p, b


def pc(params, model):
    a_carrier = np.arange(params['amin'], params['amax'] + 1)[:, np.newaxis]
    b_carrier = np.arange(params['bmin'], params['bmax'] + 1)[:, np.newaxis]
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    if model == 1:
        p_A = stats.binom.pmf(a_quant, a_carrier, params['p1']).sum(axis=0)
        p_B = stats.binom.pmf(b_quant, b_carrier, params['p2']).sum(axis=0)

    elif model == 2:
        p_A = stats.poisson.pmf(a_quant, a_carrier * params['p1']).sum(axis=0)
        p_B = stats.poisson.pmf(b_quant, b_carrier * params['p2']).sum(axis=0)

    pC = np.convolve(p_A, p_B)

    return pC / pC.sum(), np.arange(params['amax'] + params['bmax'] + 1)


def pd(params, model):
    d_carrier = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)

    pD_C = stats.binom.pmf(d_carrier[:, np.newaxis], c_carrier,
                           params['p3'], c_carrier)
    pD = np.sum(pD_C * pc(params, model)[0], axis=1)

    return pD / pD.sum(), d_carrier


def pc_a(a, params, model):
    b_carrier = np.arange(params['bmin'], params['bmax'] + 1)[:, np.newaxis]
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    if model == 1:
        p_A = stats.binom.pmf(a_quant, a[:, np.newaxis], params['p1'])
        p_B = stats.binom.pmf(b_quant, b_carrier, params['p2']).sum(axis=0)

    elif model == 2:
        p_A = stats.poisson.pmf(a_quant, a[:, np.newaxis] * params['p1'])
        p_B = stats.poisson.pmf(b_quant, b_carrier * params['p2']).sum(axis=0)

    pC_A = []
    for p_a in p_A:
        tmp = np.convolve(p_a, p_B)
        pC_A += [tmp / tmp.sum()]

    return np.array(pC_A).T, np.arange(params['amax'] + params['bmax'] + 1)


def pc_b(b, params, model):
    a_carrier = np.arange(params['amin'], params['amax'] + 1)[:, np.newaxis]
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    if model == 1:
        p_A = stats.binom.pmf(a_quant, a_carrier, params['p1']).sum(axis=0)
        p_B = stats.binom.pmf(b_quant, b[:, np.newaxis], params['p2'])

    elif model == 2:
        p_A = stats.poisson.pmf(a_quant, a_carrier * params['p1']).sum(axis=0)
        p_B = stats.poisson.pmf(b_quant, b[:, np.newaxis] * params['p2'])

    pC_B = []
    for p_b in p_B:
        tmp = np.convolve(p_A, p_b)
        pC_B += [tmp / tmp.sum()]

    return np.array(pC_B).T, np.arange(params['amax'] + params['bmax'] + 1)


def pc_d(d, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)
    pD_C = stats.binom.pmf(d[:, np.newaxis], c_carrier,
                           params['p3'], c_carrier)
    pC_D = (pD_C * pc(params, model)[0]).T

    return pC_D / pC_D.sum(axis=0), c_carrier


def pc_ab(a, b, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    a = a[:, np.newaxis]
    b = b[:, np.newaxis]

    if model == 1:
        p_A = stats.binom.pmf(a_quant, a, params['p1'])
        p_B = stats.binom.pmf(b_quant, b, params['p2'])
        p = []
        for p_a in p_A:
            p_by_b = []
            for p_b in p_B:
                tmp = np.convolve(
                    p_a, p_b
                )
                p_by_b += [tmp / tmp.sum()]
            p += [p_by_b]

    elif model == 2:
        p = []
        for a_ in a.ravel():
            p += [stats.poisson.pmf(c_carrier,
                                    a_ * params['p1'] + b * params['p2'])]

    return np.transpose(np.array(p), (2, 0, 1)), c_carrier


def pc_abd(a, b, d, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)

    pD_C = stats.binom.pmf(d[:, np.newaxis], c_carrier,
                           params['p3'], c_carrier)
    pC_ABD = []
    for p in pD_C:
        tmp = p.reshape(-1, 1, 1) * pc_ab(a, b, params, model)[0]
        pC_ABD += [tmp / tmp.sum(axis=0)]

    return np.transpose(pC_ABD, (1, 2, 3, 0)), c_carrier

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
        p_A = stats.binom.pmf(a_quant, a, params['p1'])
        p_B = stats.binom.pmf(b_quant, b_carrier, params['p2']).sum(axis=0)

    elif model == 2:
        p_A = stats.poisson.pmf(a_quant, a * params['p1'])
        p_B = stats.poisson.pmf(b_quant, b_carrier * params['p2']).sum(axis=0)

    pC_A = np.convolve(p_A, p_B)

    return pC_A / pC_A.sum(), np.arange(params['amax'] + params['bmax'] + 1)


def pc_b(b, params, model):
    a_carrier = np.arange(params['amin'], params['amax'] + 1)[:, np.newaxis]
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    if model == 1:
        p_A = stats.binom.pmf(a_quant, a_carrier, params['p1']).sum(axis=0)
        p_B = stats.binom.pmf(b_quant, b, params['p2'])

    elif model == 2:
        p_A = stats.poisson.pmf(a_quant, a_carrier * params['p1']).sum(axis=0)
        p_B = stats.poisson.pmf(b_quant, b * params['p2'])

    pC_B = np.convolve(p_A, p_B)

    return pC_B / pC_B.sum(), np.arange(params['amax'] + params['bmax'] + 1)


def pc_d(d, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)
    pD_C = stats.binom.pmf(d, c_carrier, params['p3'], c_carrier)
    pC_D = pD_C * pc(params, model)[0]

    return pC_D / pC_D.sum(), c_carrier


def pc_ab(a, b, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)
    a_quant = np.arange(params['amax'] + 1)
    b_quant = np.arange(params['bmax'] + 1)

    if model == 1:
        p = np.convolve(
            stats.binom.pmf(a_quant, a, params['p1']),
            stats.binom.pmf(b_quant, b, params['p2'])
        )
        p /= p.sum()

    elif model == 2:
        p = stats.poisson.pmf(c_carrier, a * params['p1'] + b * params['p2'])

    return p, c_carrier


def pc_abd(a, b, d, params, model):
    c_carrier = np.arange(params['amax'] + params['bmax'] + 1)

    pD_C = stats.binom.pmf(d, c_carrier, params['p3'], c_carrier)
    pC_ABD = pD_C * pc_ab(a, b, params, model)[0]

    return pC_ABD / pC_ABD.sum(), c_carrier

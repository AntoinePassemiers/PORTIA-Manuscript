import numpy as np


def get_statistics(y_target, y_pred):
    """
    Note: taken from Thomas Cokelaer's implementation:
    https://github.com/dreamtools/dreamtools/blob/250bf595f64a7508d3e98f31e5d28acf65e0f6e6/dreamtools/core/rocs.py#L363
    """

    T = len(y_target)
    P = np.sum(y_target)
    N = T - P
    L = len(y_pred)

    k = 0
    Ak = 0
    TPk = 0
    FPk = 0

    rec = []
    prec = []
    tpr = []
    fpr = []
    while k < L:
        k = k + 1
        if y_target[k-1] == 1:
            TPk = TPk + 1.
            if k == 1:
                delta = 1. / P
            else:
                delta = (1. - FPk * np.log(k / (k - 1.))) / P
            Ak = Ak + delta
        else:
            FPk = FPk + 1.
        rec.append(TPk / float(P))
        prec.append(TPk / float(k))
        tpr.append(rec[k - 1])
        fpr.append(FPk / float(N))

    TPL = TPk

    if L < T:
        rh = (P-TPL) / float(T-L)
    else:
        rh = 0.

    if L > 0:
        recL = rec[L - 1]
    else:
        recL = 0

    while TPk < P:
        k = k + 1
        TPk = TPk + 1.
        rec.append(TPk/float(P))
        if ((rec[k-1]-recL) * P + L * rh) != 0:
            prec.append(rh * P * rec[k - 1] / ((rec[k - 1] - recL) * P + L * rh))
        else:
            prec.append(0)

        tpr.append(rec[k - 1])
        FPk = TPk * (1 - prec[k - 1]) / prec[k - 1]
        fpr.append(FPk/float(N))

    AL = Ak

    if rh != 0 and L != 0:
        AUC = AL + rh * (1. - recL) + rh * (recL - L * rh / P) * np.log((L * rh + P * (1-recL)) / (L * rh))
    elif L == 0:
        AUC = P / float(T)
    else:
        AUC = Ak

    lc = fpr[0] * tpr[0] / 2.
    for n in range(1,int(L + P - TPL - 1 + 1)):
        lc = lc + ( fpr[n] + fpr[n-1]) * (tpr[n] - tpr[n-1]) / 2.

    AUROC = 1. - lc

    auroc = AUROC
    aupr = AUC

    return AUC, AUROC, prec, rec, tpr, fpr

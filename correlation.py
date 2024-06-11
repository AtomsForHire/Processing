import numpy as np
import numpy.ma as ma


def crossCorr(x, y):
    """Takes two arrays and does cross corelation on them

    Parameters
    ----------
    x:  list
        This should be a list of some statistic for each observation at each antenna
    y:  list
        Same thing as x
    """

    assert len(x) == len(y), "Length of input arrays not same for crossCor"

    for i in range(0, len(x)):
        xObs = x[i]
        yObs = y[i]
        test = ma.corrcoef(ma.masked_invalid(xObs), ma.masked_invalid(yObs))
        print(test)

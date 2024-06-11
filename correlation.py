import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


def crossCorr(x, y, obsids, xLab, yLab, corrDir):
    """Takes two arrays and does cross corelation on them

    Parameters
    ----------
    x:  list
        This should be a list of some statistic for each observation at each antenna
    y:  list
        Same thing as x
    """

    assert len(x) == len(y), "Length of input arrays not same for crossCor"

    ant = np.arange(0, 128, 1)
    for i in range(0, len(x)):
        xObs = x[i]
        yObs = y[i]

        plt.scatter(xObs, yObs, c=ant)
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.title(obsids[i])
        plt.colorbar()
        plt.savefig(corrDir + "/" + obsids[i] + ".png")
        plt.clf()

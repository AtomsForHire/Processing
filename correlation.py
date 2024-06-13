import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


def crossCorr(x, y, obsids, xLab, yLab, corrDir, nameExtension=""):
    """Takes two arrays and does cross corelation on them

    Parameters
    ----------
    x:  list
        This should be a list of some statistic for each observation at each antenna
    y:  list
        Same thing as x
    """

    assert len(x) == len(y), "Length of input arrays not same for crossCor"

    cmap = plt.cm.plasma
    cmaplist = [cmap(i) for i in range(cmap.N)]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = np.linspace(0, 127, 128)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ant = np.arange(0, 128, 1)
    for i in range(0, len(x)):
        xObs = x[i]
        yObs = y[i]

        plt.scatter(xObs, yObs, c=ant, cmap=cmap, norm=norm)
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.title(obsids[i])
        plt.colorbar()
        plt.xticks(rotation=45)

        plt.savefig(
            corrDir + "/" + obsids[i] + nameExtension + ".png", bbox_inches="tight"
        )
        plt.clf()

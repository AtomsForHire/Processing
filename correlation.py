import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.stats import linregress
from tqdm import tqdm

import image


def crossCorr(
    x,
    y,
    obsids,
    xLab,
    yLab,
    distribution,
    gridDict,
    uniqueDict,
    corrDir,
    nameExtension="",
):
    """Takes two arrays and does cross corelation on them

    Parameters
    ----------
    x:  list
        This should be a list of some statistic for all observations at each antenna
    y:  list
        Same thing as x
    obsids: list
        List of obsids
    xLab: string
        x label string
    yLab: string
        y label string
    corrDir: string
        directory to save files
    nameExtension: string
        name to add onto end of file
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

    # Store stats for each observation
    pearsonStats = np.zeros(len(obsids))
    gradientStats = np.zeros(len(obsids))
    rValStats = np.zeros(len(obsids))

    # Loop over observations
    for i in tqdm(range(0, len(x))):
        ant = np.arange(0, 128, 1)
        xObs = np.array(x[i])
        yObs = np.array(y[i])
        # yObs = yObs[xObs < 60]
        # ant = ant[xObs < 60]
        # xObs = xObs[xObs < 60]

        assert np.all(
            np.isnan(xObs) == np.isnan(yObs)
        ), "Arrays are not for the same observation!"

        # linearFit = Polynomial.fit(
        #     xObs[~np.isnan(xObs)],
        #     yObs[~np.isnan(yObs)],
        #     deg=1,
        #     domain=[min(xObs[~np.isnan(xObs)]), max(xObs[~np.isnan(xObs)])],
        #     window=[min(xObs[~np.isnan(xObs)]), max(xObs[~np.isnan(xObs)])],
        # )
        slope, intercept, r_value, p_value, std_err = linregress(
            xObs[~np.isnan(xObs)], yObs[~np.isnan(yObs)]
        )
        regression_line = slope * xObs[~np.isnan(xObs)] + intercept

        # Calculate the pearson coefficient
        # xx, yy = linearFit.linspace()
        R = ma.corrcoef(ma.masked_invalid(xObs), ma.masked_invalid(yObs))

        # Store values
        pearsonStats[i] = R[0, 1]
        gradientStats[i] = slope
        rValStats[i] = r_value

        plt.scatter(xObs, yObs, c=ant, cmap=cmap, norm=norm)
        # plt.plot(xx, yy)
        plt.plot(
            xObs[~np.isnan(xObs)],
            regression_line,
            label="m=" + str(round(slope, 4)) + ", r=" + str(round(r_value, 4)),
        )
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.title(obsids[i] + " C=" + str(round(R[0, 1], 2)))
        plt.colorbar()
        plt.legend()
        plt.xticks(rotation=45)

        plt.savefig(
            corrDir + "/" + obsids[i] + nameExtension + ".png", bbox_inches="tight"
        )
        plt.clf()

    # Plot statistics as function of observation, and group by grid number
    if distribution == "grid":
        image.gridPlot(obsids, pearsonStats, gridDict, uniqueDict)
        plt.xticks(rotation=90)
        plt.ylabel("Pearson coefficient")
        plt.savefig(
            corrDir + "/all" + nameExtension + "_pearson.png", bbox_inches="tight"
        )
        plt.clf()
        image.gridPlot(obsids, gradientStats, gridDict, uniqueDict)
        plt.xticks(rotation=90)
        plt.ylabel("Gradient of LOBF")
        plt.savefig(corrDir + "/all" + nameExtension + "_grad.png", bbox_inches="tight")
        plt.clf()
        image.gridPlot(obsids, rValStats, gridDict, uniqueDict)
        plt.xticks(rotation=90)
        plt.ylabel("r-value")
        plt.savefig(corrDir + "/all" + nameExtension + "_rval.png", bbox_inches="tight")
        plt.clf()
    elif distribution == "sorted":
        pass
        # plt.plot(obsids, drVec)

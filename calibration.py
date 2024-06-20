import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from mwa_qa import cal_metrics, read_calfits
from numpy.polynomial import Polynomial
from scipy import signal, stats
from tqdm import tqdm


def calVar(obsids, varDir, solDir):
    """Function for getting variance of gain amplitude calibration solutions through mwa_qa

    Parameters
    ----------
    obsids: list
        List of observation ids
    varDir: string
        Path to save results to
    solDir: string
        Path to directory containing solutions/metafits
    """

    for i in tqdm(range(0, len(obsids))):
        obsid = obsids[i]
        calfitsPath = solDir + "/" + str(obsid) + "_solutions.fits"
        metfitsPath = solDir + "/" + str(obsid) + ".metafits"

        calObj = cal_metrics.CalMetrics(calfitsPath, metfitsPath)
        calObj.run_metrics()
        plt.plot(
            calObj.variance_for_baselines_less_than(1)[0, :, 0], label="XX", marker="."
        )
        plt.plot(
            calObj.variance_for_baselines_less_than(1)[0, :, 3], label="YY", marker="."
        )
        plt.title(obsid + " XX + YY")
        plt.xlabel("Antenna")
        plt.ylabel("Variance")
        plt.legend()
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which="minor", alpha=0.5)
        plt.savefig(varDir + "/" + obsid + "_var.pdf", bbox_inches="tight")
        plt.clf()
        calObj.write_to()


def calRMS(obsids, rmsDir, solDir):
    """Function for getting rms of gain amplitude calibration solutions through mwa_qa

    Parameters
    ----------
    obsids: list
        List of observation ids
    varDir: string
        Path to save results to
    solDir: string
        Path to directory containing solutions/metafits
    """
    for i in tqdm(range(0, len(obsids))):
        metPath = solDir + "/" + obsids[i] + "_solutions_cal_metrics.json"
        with open(metPath) as f:
            calibration = json.load(f)

        plt.plot(calibration["XX"]["RMS"], label="XX", marker=".")
        plt.plot(calibration["YY"]["RMS"], label="YY", marker=".")
        plt.xlabel("Antenna")
        plt.ylabel("RMS")
        plt.title(obsids[i] + " XX + YY")
        plt.legend()
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which="minor", alpha=0.5)
        plt.savefig(rmsDir + "/" + obsids[i] + "_rms.pdf", bbox_inches="tight")
        plt.clf()


def nan_helper(y):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    y: numpy array
        1d numpy array with possible NaNs

    Returns
    -------
    nans: numpy array
        logical indices of NaNs
    index: numpy array
        a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices

    Example
    -------
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    -------
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpChoices(x, y, interp_type):
    """Function for interpolating with different styles

    Parameters
    ----------
    x: list
        x axis
    y: list
        y axis
    interp_type: string
        Choice of interpolation

    Returns
    -------
    y: list
        Interpolated function

    -------
    """
    if interp_type == "linear":
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    if interp_type == "zero":
        y = np.array([0 if math.isnan(x) else x for x in y])

    if interp_type == "cspline":
        nans = np.isnan(y)
        x_interp = x[~nans]
        y_interp = y[~nans]
        cs = sci.interpolate.CubicSpline(x_interp, y_interp)
        y_missing = cs(x[nans])
        y[nans] = y_missing

    return y


def plotAllInterp(
    ants, xxSmoothnessAllInterps, yySmoothnessAllInterps, obs, interps, smoothDir
):
    # Save figure for all interp types
    xMax = np.nanmax(xxSmoothnessAllInterps)
    yMax = np.nanmax(yySmoothnessAllInterps)
    ls = ["solid", "dashed", "dotted"]
    legend_lines = list()
    for n in range(0, len(interps)):
        plt.plot(ants, xxSmoothnessAllInterps[n], color="blue", linestyle=ls[n])
        plt.plot(ants, yySmoothnessAllInterps[n], color="red", linestyle=ls[n])
        ax = plt.gca()
        (temp,) = ax.plot(0, -1, color="grey", linestyle=ls[n], label=interps[n])
        legend_lines.append(temp)

        (l4,) = ax.plot(0, -1, color="blue", label="XX")
        (l5,) = ax.plot(0, -1, color="red", label="YY")
        first_legend = ax.legend(handles=legend_lines, loc="upper right")
        ax.add_artist(first_legend)
        ax.legend(handles=[l4, l5], loc="upper left")
        plt.xlabel("Antenna number")
        plt.ylabel("Smoothness")
        plt.ylim(0, 1.15 * np.max((xMax, yMax)))
        plt.title(obs)
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which="minor", alpha=0.5)
        plt.savefig(smoothDir + "/" + str(obs) + "_all.pdf")
        plt.clf()


def plotSmoothnessAllObs(
    obsids,
    ant,
    smoothness,
    smoothDir,
    distribution,
    pol,
    gridDict,
    uniqueDict,
    yAxis="Smoothness",
    name="",
):
    """Function for plotting the smoothness metric

    Parameters
    ----------
    obsids: list
        List of observation ids
    ant: list
        List of antenna numbers
    smoothness: list[list]
        List of list of all smoothness values across all antennas for all obs
    smoothDir: string
        Path to save results to
    distribution: string
        How the obs are sorted
    pol: string
        String for naming files properly
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur
    rmse: bool
        Save using rmse settings or not, False by default so it doesn't break
        original calls

    Returns
    -------
    None

    -------
    """
    colors = plt.cm.jet(np.linspace(0, 1, len(obsids)))
    if distribution == "grid":
        linestyles = ["solid", "dashed", "dotted", "dashdot"]
        marker = [
            " ",
            ".",
            "s",
            "o",
            "*",
            "x",
        ]
        styles = {}

        for i, key in enumerate(uniqueDict):
            styles[key] = [
                linestyles[i % len(linestyles)],
                marker[int(i // len(linestyles))],
            ]

        obs_legend_list = list()
        for i in range(0, len(obsids)):
            obs = obsids[i]

            (temp,) = plt.plot(
                ant,
                smoothness[i],
                label=obs,
                linestyle=styles[gridDict[obs]][0],
                marker=styles[gridDict[obs]][1],
                color=colors[i],
            )

            obs_legend_list.append(temp)

        # Handle legend for grid points
        grid_legend_list = list()
        for key in styles:
            (temp,) = plt.plot(
                -1,
                color="gray",
                linestyle=styles[key][0],
                marker=styles[key][1],
                label=key,
            )
            grid_legend_list.append(temp)

        ax = plt.gca()
        grid_legend = ax.legend(
            handles=grid_legend_list, bbox_to_anchor=(1.4, 0.5), loc="center left"
        )
        obs_legend = ax.legend(
            handles=obs_legend_list, bbox_to_anchor=(1.04, 0.5), loc="center left"
        )

        ax.add_artist(grid_legend)

    elif distribution == "sorted":
        # plt.gca().set_prop_cycle(plt.cycler('color', c1.colors))
        obs_legend_list = list()
        for i in range(0, len(obsids)):
            obs = obsids[i]
            (temp,) = plt.plot(
                ant, smoothness[i], alpha=0.7, label=obs, color=colors[i]
            )

            obs_legend_list.append(temp)

        ax = plt.gca()
        obs_legend = ax.legend(
            handles=obs_legend_list, bbox_to_anchor=(1.04, 0.5), loc="center left"
        )

        ax.add_artist(obs_legend)

    ax = plt.gca()
    plt.xlabel("Antenna number")
    plt.ylabel(yAxis)

    plt.xticks(np.linspace(0, 127, 128), minor=True)
    plt.grid()
    plt.grid(which="minor", alpha=0.5)
    plt.ylim(0.95 * np.nanmin(smoothness), 1.05 * np.nanmax(smoothness))
    bbox_artists = [obs_legend]
    if distribution == "grid":
        bbox_artists.append(grid_legend)

    plt.savefig(
        smoothDir + "/" + "all_obs_" + pol + name + "_linear.pdf",
        bbox_extra_artists=(bbox_artists),
        bbox_inches="tight",
    )
    plt.clf()


def plotDebug(old, yreal, yimag, y, yf, obs, ant):
    """Function to use when debugging the smoothness parameter

    Parameters
    ----------
    old: array
        Contains the real and imaginary parts of the original calibration solutions
    yreal: array/int
        Interpolated real part of the original calibration sol
    yimag: array
        Same as above but for imaginary
    yf: array
        Fourier transformed yreal + 1j*yimag
    obs: string
        String for observation id
    ant: integer
        antenna number

    Returns
    -------
    None

    -------
    """
    if len(yimag) != 1:
        smooth = np.average(abs(yf[1 : int(3072 / 2)]) / abs(yf[0]))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ax1.plot(old.real, "r.", alpha=0.5, markersize=0.75, label="old")
        ax1.plot(yreal, linewidth=0.5, label="interp")
        ax1.set_title(obs + " amps solutions real antenna " + str(ant))
        ax1.legend()

        ax2.plot(old.imag, "r.", alpha=0.5, markersize=0.75, label="old")
        ax2.plot(yimag, linewidth=0.5, label="interp")
        ax2.set_title(obs + " amps solutions imag")

        ax3.plot(abs(y), linewidth=0.5)
        ax3.set_title(obs + " absolute value")

        ax4.plot(abs(yf))
        ax4.set_title(f"Absolute value of fourier transform {smooth}")
        plt.show()
    else:
        smooth = np.average(abs(yf[1 : int(3072 / 2)]) / abs(yf[0]))
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(old, "r.", alpha=0.5, markersize=0.75, label="old")
        ax1.plot(yreal, linewidth=0.5, label="interp")
        ax1.set_title(obs + " phase solutions smoothness")
        ax1.legend()
        ax2.plot(abs(yf))
        ax2.set_title(f"Absolute value of fourier transform {smooth}")
        plt.show()


def movePhases(phases):
    """Function to make phases not wrap

    Parameters
    ----------
    phases: array
        array of phases to process

    Returns
    -------
    phases: array
        modified phases
    """

    # BUG: NEED TO DEAL WITH FLAGGED FREQUENCY CHANNELS
    prevAngle = phases[0]
    for i in range(1, len(phases)):
        currAngle = phases[i]
        if np.isnan(currAngle):
            continue

        if currAngle - prevAngle > 180.0:
            currAngle -= 360.0
        if currAngle - prevAngle < -180.0:
            currAngle += 360.0

        prevAngle = currAngle
        phases[i] = currAngle

    return phases


def phaseFit(x, y, interp_type, obs, ant, norm, debug, debugTargetObs, debugTargetAnt):
    """Function for calculating RMSE of the phase solutions and cubic and quadratic coeffs

    Parameters
    ----------
    x: array
        frequency array
    y: array
        phase solutions

    Returns
    -------
    rmse: float
        root mean squared error

    """

    y = interpChoices(x, y, interp_type)
    # Do linear fit stats
    # NOTE: For some absolutely non-sensical reason numpy has a domain
    # and window parameter. By default the domain is the domain of the
    # x-values, but window has default value of [-1, 1]. This means
    # the fitting is done in the correct domain, but outputting the
    # coefs are in the window-domain. SO fucking stupid.
    linearFit = Polynomial.fit(
        x, y, deg=1, domain=[x.min(), x.max()], window=[x.min(), x.max()]
    )

    # Calculate the mean absolute deviation (MAD)
    residuals = np.zeros(len(x))
    xx, yy = linearFit.linspace(len(x))
    for i in range(0, len(x)):
        residuals[i] = y[i] - yy[i]

    mad = 1.0 / len(x) * np.sum(np.abs(residuals[:] - np.mean(residuals)))

    rmse = np.sqrt(np.mean((y - yy) ** 2))

    # Do cubic fit stats
    cubicFit = Polynomial.fit(
        x, y, deg=3, domain=[x.min(), x.max()], window=[x.min(), x.max()]
    )
    xx1, yy1 = cubicFit.linspace(len(x))
    # print(linearFit.coef)
    # print(linearFit.convert().coef)
    if debug:
        if obs in debugTargetObs:
            if debugTargetAnt is None:
                fig, (ax1, ax2) = plt.subplots(2)
                ax1.plot(y, "r.", alpha=0.5, markersize=0.75)
                ax1.plot(xx, yy, "b-", alpha=0.5, linewidth=0.5)
                ax1.plot(xx1, yy1, "g-", alpha=0.5, linewidth=0.5)
                ax1.set_title(obs + " " + str(ant))
                ax2.plot(residuals, "b.", markersize=0.9)
                ax2.set_title("Linear fit residuals")
                print(cubicFit)
                print("MAD: ", mad)
                plt.show()
            else:
                if ant in debugTargetAnt:
                    fig, (ax1, ax2) = plt.subplots(2)
                    ax1.plot(y, "r.", alpha=0.5, markersize=0.75)
                    ax1.plot(xx, yy, "b-", alpha=0.5, linewidth=0.5)
                    ax1.plot(xx1, yy1, "g-", alpha=0.5, linewidth=0.5)
                    ax1.set_title(obs + " " + str(ant))
                    ax2.plot(residuals, "b.", markersize=0.9)
                    ax2.set_title("Linear fit residuals")
                    print(linearFit)
                    print(cubicFit)
                    print("MAD: ", mad)
                    plt.show()

    # if norm:
    #     if ant == 127:
    #         rmse = np.nan
    #         mad = np.nan

    return (
        rmse,
        mad,
        linearFit.coef[1],
        (cubicFit.coef[1], cubicFit.coef[2], cubicFit.coef[3]),
    )


def calcSmooth(
    x,
    old,
    yreal,
    yimag,
    interp_type,
    obs,
    ant,
    norm,
    debug,
    debugTargetObs,
    debugTargetAnt,
):
    """Function for calculating the smoothness of calibration solutions

    Parameters
    ----------
    x: array
        contains antenna numbers
    old: array
        contains the original calibration solutions for debugging purposes
    yreal: array
        non-interpolated real part of the calibration solutions
    yimag: array
        non-interpolated imaginary part of the calibratio solutions
    interp_type: string
        interpolation method used
    obs: string
        observation id
    ant: int
        antenna number
    norm: bool
        if using the normalise calibrations then skip antenna 127
    debug: bool
        print and plot debug stuff
    debugTargetObs: list
        list of target observations for debugging
    debugTargetAnt: list
        list of target antennas for debugging

    Returns
    -------
    smooth: float
        a value representing the smoothness of the calibration solutions

    -------
    """

    # If we are looking at normalised calibration solutions then return
    # immediately when we are reference antenna
    # if norm and ant == 127:
    #     return 0

    # Check if we are doing phase calibrations
    if len(yimag) == 1:
        y = interpChoices(x, yreal, interp_type)
    else:
        yreal = interpChoices(x, yreal, interp_type)
        yimag = interpChoices(x, yimag, interp_type)
        y = yreal + 1j * yimag

    yf = np.fft.fft(y)
    smooth = np.average(abs(yf[1 : int(len(x) / 2)]) / abs(yf[0]))

    if interp_type == "linear" and debug:
        if debugTargetObs is None:
            if ant in debugTargetAnt:
                plotDebug(old, y, [0], y, yf, obs, ant)
        elif obs in debugTargetObs:
            if ant in debugTargetAnt:
                # plt.plot(y)
                # xx, yy = linearFit.linspace()
                # print(xx, yy)
                # plt.plot(xx, yy)
                # plt.show()
                plotDebug(old, yreal, yimag, y, yf, obs, ant)

    return smooth


def getEuclidSame(xx, yy):
    copy = xx.copy()
    copy -= xx[0] - yy[0]

    return np.abs(np.mean(copy - yy))


def getEuclid(xx, yy):
    return np.abs(np.mean(xx - yy))


def getKsTest(xx, yy):
    copy = xx.copy()
    copy -= xx[0] - yy[0]
    result = stats.ks_2samp(copy, yy, alternative="two-sided")
    return (result[0], result[1])


def calAmpSmoothness(
    obsids,
    solDir,
    smoothDir,
    distribution,
    gridDict,
    uniqueDict,
    debug,
    debugTargetObs,
    debugTargetAnt,
    normalise,
    useWindow=False,
):
    """Function for calculating smoothness of calibration gain amplitudes for all obs

    Parameters
    ----------
    obsids: list
        List of observation ids
    smoothDir: string
        Path to save results to
    distribution: string
        How the obs are sorted
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur
    normalise: bool
        True or False, enable or disable normalisation

    Returns
    -------
    None

    -------
    """

    ant = np.linspace(0, 127, 128)
    # interps = ['zero', 'linear', 'cspline']
    interps = ["zero", "linear"]
    allObsXXSmoothness = list()
    allObsYYSmoothness = list()
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        filename = solDir + "/" + obs + "_solutions.fits"
        cal = read_calfits.CalFits(filename, norm=normalise)
        nFreq = cal.Nchan
        x = np.linspace(0, nFreq - 1, nFreq)

        if useWindow is True:
            window = signal.windows.blackmanharris(nFreq)
        else:
            window = np.ones(nFreq)

        xxSmoothnessAllInterps = list()
        yySmoothnessAllInterps = list()
        for interp_type in interps:
            xxSmoothness = list()
            yySmoothness = list()
            # Loop over antennas
            for j in range(0, len(cal.gain_array[0, :, 0, 0])):
                # Extract amplitudes for XX pol
                xxOld = cal.gain_array[0, j, :, 0].copy()
                xxReal = cal.gain_array[0, j, :, 0].real.copy() * window
                xxImag = cal.gain_array[0, j, :, 0].imag.copy() * window

                # Skip flagged antennas
                if (
                    (np.nansum(xxImag) == 0.0)
                    and (np.nansum(xxReal) == 0.0)
                    or (normalise and j == 127)
                ):
                    xxSmoothness.append(np.nan)
                    yySmoothness.append(np.nan)
                    continue

                smooth = calcSmooth(
                    x,
                    xxOld,
                    xxReal,
                    xxImag,
                    interp_type,
                    obs,
                    j,
                    normalise,
                    debug,
                    debugTargetObs,
                    debugTargetAnt,
                )

                xxSmoothness.append(smooth)

                # Samething for YY pol
                yyOld = cal.gain_array[0, j, :, 3].copy()
                yyReal = cal.gain_array[0, j, :, 3].real * window
                yyImag = cal.gain_array[0, j, :, 3].imag * window
                smooth1 = calcSmooth(
                    x,
                    yyOld,
                    yyReal,
                    yyImag,
                    interp_type,
                    obs,
                    j,
                    normalise,
                    debug,
                    debugTargetObs,
                    debugTargetAnt,
                )
                yySmoothness.append(smooth1)

            xxSmoothnessAllInterps.append(xxSmoothness)
            yySmoothnessAllInterps.append(yySmoothness)

            if interp_type == "linear":
                allObsXXSmoothness.append(xxSmoothness)
                allObsYYSmoothness.append(yySmoothness)

        # Plot for a single observation, the different smoothness for each interpolation
        plotAllInterp(
            ant, xxSmoothnessAllInterps, yySmoothnessAllInterps, obs, interps, smoothDir
        )

    if useWindow:
        name = "_window"
    else:
        name = ""

    # Save figure for all obsids XX
    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXSmoothness,
        smoothDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
        name=name,
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYSmoothness,
        smoothDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
        name=name,
    )

    return allObsXXSmoothness, allObsYYSmoothness


def calPhaseSmoothness(
    obsids,
    solDir,
    smoothDir,
    phaseStatsDir,
    distribution,
    gridDict,
    uniqueDict,
    debug,
    debugTargetObs,
    debugTargetAnt,
    normalise,
):
    """Function for calculating smoothness of calibration phase amplitudes for all obs

    Parameters
    ----------
    obsids: list
        List of observation ids
    smoothDir: string
        Path to save results to
    distribution: string
        How the obs are sorted
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur
    normalise: bool
        True or False, enable or disable normalisation

    Returns
    -------
    None

    -------
    """

    ant = np.linspace(0, 127, 128)
    interps = ["zero", "linear"]
    allObsXXSmoothness = list()
    allObsYYSmoothness = list()
    allObsXXRMSE = list()
    allObsYYRMSE = list()
    allObsXXMAD = list()
    allObsYYMAD = list()
    allObsXXCubic = list()
    allObsYYCubic = list()
    allObsXXQuad = list()
    allObsYYQuad = list()
    allObsEuclid = list()
    allObsEuclidSame = list()
    allObsPval = list()
    allObsKsTest = list()

    # Loop through observations
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        filename = solDir + "/" + obs + "_solutions.fits"
        cal = read_calfits.CalFits(filename, norm=normalise)

        nFreq = cal.Nchan
        x = np.linspace(0, nFreq - 1, nFreq)

        xxSmoothnessAllInterps = list()
        yySmoothnessAllInterps = list()

        # Loop through interpolation types
        for interp_type in interps:
            xxSmoothness = list()
            yySmoothness = list()

            xxRMSE = list()
            yyRMSE = list()
            xxMAD = list()
            yyMAD = list()
            xxCubic = list()
            yyCubic = list()
            xxQuad = list()
            yyQuad = list()
            euclid = list()
            euclidSame = list()
            pVal = list()
            ksTest = list()

            # Loop over antennas (Except the last one which is the reference antenna)
            for j in range(0, len(cal.phases[0, :, 0, 0])):
                # Extract amplitudes for XX pol
                # 'old' for debugging purposes
                xxOld = cal.phases[0, j, :, 0].copy()
                xx = cal.phases[0, j, :, 0].copy()
                # Normalise all angles to sit between [0, 360]
                xx = movePhases(xx)

                # Skip flagged antennas
                if np.nansum(xx) == 0.0 or (normalise and j == 127):
                    xxSmoothness.append(np.nan)
                    yySmoothness.append(np.nan)
                    xxRMSE.append(np.nan)
                    yyRMSE.append(np.nan)
                    xxMAD.append(np.nan)
                    yyMAD.append(np.nan)
                    xxCubic.append(np.nan)
                    yyCubic.append(np.nan)
                    xxQuad.append(np.nan)
                    yyQuad.append(np.nan)
                    euclid.append(np.nan)
                    euclidSame.append(np.nan)
                    pVal.append(np.nan)
                    ksTest.append(np.nan)
                    continue

                # Interpolate phase solutions
                smooth = calcSmooth(
                    x,
                    xxOld,
                    xx,
                    [0],
                    interp_type,
                    obs,
                    j,
                    normalise,
                    debug,
                    debugTargetObs,
                    debugTargetAnt,
                )

                if interp_type == "linear":
                    rmse, mad, grad, coeffs = phaseFit(
                        x,
                        xx,
                        interp_type,
                        obs,
                        j,
                        normalise,
                        debug,
                        debugTargetObs,
                        debugTargetAnt,
                    )
                    xxRMSE.append(rmse)
                    xxMAD.append(mad)
                    xxCubic.append(coeffs[2])
                    xxQuad.append(coeffs[1])

                xxSmoothness.append(smooth)

                # Samething for YY pol
                yyOld = cal.phases[0, j, :, 3].copy()
                yy = cal.phases[0, j, :, 3]
                yy = movePhases(yy)

                smooth1 = calcSmooth(
                    x,
                    yyOld,
                    yy,
                    [0],
                    interp_type,
                    obs,
                    j,
                    normalise,
                    debug,
                    debugTargetObs,
                    debugTargetAnt,
                )

                if interp_type == "linear":
                    euclidSame.append(getEuclidSame(xx, yy))
                    euclid.append(getEuclid(xx, yy))
                    rmse1, mad1, grad1, coeffs1 = phaseFit(
                        x,
                        yy,
                        interp_type,
                        obs,
                        j,
                        normalise,
                        debug,
                        debugTargetObs,
                        debugTargetAnt,
                    )
                    yyRMSE.append(rmse1)
                    yyMAD.append(mad1)
                    yyCubic.append(coeffs1[2])
                    yyQuad.append(coeffs1[2])
                    tmp = getKsTest(xx, yy)
                    pVal.append(tmp[1])
                    ksTest.append(tmp[0])

                yySmoothness.append(smooth1)

            xxSmoothnessAllInterps.append(xxSmoothness)
            yySmoothnessAllInterps.append(yySmoothness)

            if interp_type == "linear":
                allObsXXSmoothness.append(xxSmoothness)
                allObsYYSmoothness.append(yySmoothness)
                allObsXXRMSE.append(xxRMSE)
                allObsYYRMSE.append(yyRMSE)
                allObsXXMAD.append(xxMAD)
                allObsYYMAD.append(yyMAD)
                allObsXXCubic.append(xxCubic)
                allObsYYCubic.append(yyCubic)
                allObsXXQuad.append(xxQuad)
                allObsYYQuad.append(yyQuad)
                allObsEuclid.append(euclid)
                allObsEuclidSame.append(euclidSame)
                allObsPval.append(pVal)
                allObsKsTest.append(ksTest)

        # Plot for a single observation, the different smoothness for each interpolation
        plotAllInterp(
            ant, xxSmoothnessAllInterps, yySmoothnessAllInterps, obs, interps, smoothDir
        )

    # Save figure for all obsids XX
    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXSmoothness,
        smoothDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYSmoothness,
        smoothDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXRMSE,
        phaseStatsDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
        yAxis="RMSE",
        name="_RMSE",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYRMSE,
        phaseStatsDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
        yAxis="RMSE",
        name="_RMSE",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXMAD,
        phaseStatsDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
        yAxis="Mean absolute deivation",
        name="_MAD",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYMAD,
        phaseStatsDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
        yAxis="Mean absolute deivation",
        name="_MAD",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXCubic,
        phaseStatsDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
        yAxis="Cubic coefficient",
        name="_cubic",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYCubic,
        phaseStatsDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
        yAxis="Cubic coefficient",
        name="_cubic",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsXXQuad,
        phaseStatsDir,
        distribution,
        "xx",
        gridDict,
        uniqueDict,
        yAxis="Quadratic coefficient",
        name="_quad",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsYYQuad,
        phaseStatsDir,
        distribution,
        "yy",
        gridDict,
        uniqueDict,
        yAxis="Quadratic coefficient",
        name="_quad",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsEuclid,
        phaseStatsDir,
        distribution,
        "both",
        gridDict,
        uniqueDict,
        yAxis="mean of euclid distance",
        name="_euclid",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsEuclidSame,
        phaseStatsDir,
        distribution,
        "both",
        gridDict,
        uniqueDict,
        yAxis="mean of euclid distance",
        name="_euclid_same",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsPval,
        phaseStatsDir,
        distribution,
        "both",
        gridDict,
        uniqueDict,
        yAxis="p-value",
        name="_pval",
    )

    plotSmoothnessAllObs(
        obsids,
        ant,
        allObsKsTest,
        phaseStatsDir,
        distribution,
        "both",
        gridDict,
        uniqueDict,
        yAxis="KS metric",
        name="_kstest",
    )

    return (
        allObsXXRMSE,
        allObsYYRMSE,
        allObsXXMAD,
        allObsYYMAD,
        allObsEuclidSame,
        allObsKsTest,
    )

import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from mwa_qa import cal_metrics, read_calfits
from numpy.polynomial import Polynomial
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


def plotSmoothnessAllObs(
    obsids, ant, smoothness, smoothDir, distribution, pol, gridDict, uniqueDict
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
    plt.ylabel("Smoothness")
    plt.xticks(np.linspace(0, 127, 128), minor=True)
    plt.grid()
    plt.grid(which="minor", alpha=0.5)
    plt.ylim(0.95 * np.nanmin(smoothness), 1.05 * np.nanmax(smoothness))
    bbox_artists = [obs_legend]
    if distribution == "grid":
        bbox_artists.append(grid_legend)

    plt.savefig(
        smoothDir + "/" + "all_obs_" + pol + "_linear.pdf",
        bbox_extra_artists=(bbox_artists),
        bbox_inches="tight",
    )
    plt.clf()


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

        xxSmoothnessAll = list()
        yySmoothnessAll = list()
        for interp_type in interps:
            xxSmoothness = list()
            yySmoothness = list()
            # Loop over antennas
            for j in range(0, len(cal.gain_array[0, :, 0, 0])):
                # Extract amplitudes for XX pol
                old = cal.gain_array[0, j, :, 0].copy()
                yreal = cal.gain_array[0, j, :, 0].real.copy()
                yimag = cal.gain_array[0, j, :, 0].imag.copy()

                # Skip flagged antennas
                if (np.nansum(yimag) == 0.0) and (np.nansum(yreal) == 0.0):
                    xxSmoothness.append(np.nan)
                    yySmoothness.append(np.nan)
                    continue

                yreal = interpChoices(x, yreal, interp_type)
                yimag = interpChoices(x, yimag, interp_type)
                y = yreal + 1.0j * yimag
                yf = np.fft.fft(y)
                smooth = np.average(abs(yf[1 : int(nFreq / 2)]) / abs(yf[0]))
                if normalise and j == 127:
                    smooth = 0

                if interp_type == "linear" and debug:
                    if debugTargetObs is None:
                        if j in debugTargetAnt:
                            print(filename)
                            print(cal.annames[j])
                            plotDebug(old, yreal, yimag, y, yf, obs)
                    elif obs in debugTargetObs:
                        if j in debugTargetAnt:
                            # print(filename, cal.annames[j])
                            # plt.plot(abs(cal.gain_array[0, j, :, 0]))
                            # plt.show()
                            print(filename)
                            print(cal.annames[j])
                            plotDebug(old, yreal, yimag, y, yf, obs)

                xxSmoothness.append(smooth)

                # Samething for YY pol
                yreal1 = cal.gain_array[0, j, :, 3].real
                yimag1 = cal.gain_array[0, j, :, 3].imag
                yreal1 = interpChoices(x, yreal1, interp_type)
                yimag1 = interpChoices(x, yimag1, interp_type)
                y1 = yreal1 + 1.0j * yimag1
                yf1 = np.fft.fft(y1)
                smooth1 = np.average(abs(yf1[1 : int(nFreq / 2)]) / abs(yf1[0]))
                if normalise and j == 127:
                    smooth1 = 0
                yySmoothness.append(smooth1)

            xxSmoothnessAll.append(xxSmoothness)
            yySmoothnessAll.append(yySmoothness)

            if interp_type == "linear":
                allObsXXSmoothness.append(xxSmoothness)
                allObsYYSmoothness.append(yySmoothness)

            # Save figure for particular interp_type
            plt.plot(ant, xxSmoothness, label="XX", color="blue")
            plt.plot(ant, yySmoothness, label="YY", color="red")
            plt.xlabel("Antenna number")
            plt.ylabel("Smoothness")
            plt.title(obs + " " + interp_type)
            plt.legend()
            plt.xticks(np.linspace(0, 127, 128), minor=True)
            plt.grid()
            plt.grid(which="minor", alpha=0.5)
            plt.savefig(smoothDir + "/" + str(obs) + "_" + interp_type + ".pdf")
            plt.clf()

        # Save figure for all interp types
        xMax = np.nanmax(xxSmoothnessAll)
        yMax = np.nanmax(yySmoothnessAll)
        ls = ["solid", "dashed", "dotted"]
        interp_label = ["zero", "linear", "cspline"]
        legend_lines = list()
        for n in range(0, len(interps)):
            plt.plot(ant, xxSmoothnessAll[n], color="blue", linestyle=ls[n])
            plt.plot(ant, yySmoothnessAll[n], color="red", linestyle=ls[n])
            ax = plt.gca()
            (temp,) = ax.plot(
                0, -1, color="grey", linestyle=ls[n], label=interp_label[n]
            )
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


def calPhaseSmoothness(
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
    # Loop through observations
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        filename = solDir + "/" + obs + "_solutions.fits"
        cal = read_calfits.CalFits(filename, norm=normalise)

        nFreq = cal.Nchan
        x = np.linspace(0, nFreq - 1, nFreq)

        xxSmoothnessAll = list()
        yySmoothnessAll = list()
        # Loop through interpolation types
        for interp_type in interps:
            xxSmoothness = list()
            yySmoothness = list()

            # Loop over antennas (Except the last one which is the reference antenna)
            for j in range(0, len(cal.phases[0, :, 0, 0])):
                # Extract amplitudes for XX pol
                # old for debugging purposes
                old = cal.phases[0, j, :, 0].copy()
                y = cal.phases[0, j, :, 0].copy()
                # Normalise all angles to sit between [0, 360]
                y = movePhases(y)

                # Skip flagged antennas
                if np.nansum(y) == 0.0:
                    xxSmoothness.append(np.nan)
                    yySmoothness.append(np.nan)
                    continue

                # Interpolate phase solutions
                y = interpChoices(x, y, interp_type)
                linearFit = Polynomial.fit(x, y, deg=1)

                yf = np.fft.fft(y)
                smooth = np.average(abs(yf[1 : int(nFreq / 2)]) / abs(yf[0]))
                if normalise and j == 127:
                    smooth = 0

                if interp_type == "linear" and debug:
                    if debugTargetObs is None:
                        if j in debugTargetAnt:
                            print(filename)
                            print(cal.annames[j])
                            plotDebug(old, y, 0, y, yf, obs)
                    elif obs in debugTargetObs:
                        if j in debugTargetAnt:
                            print(filename)
                            print(cal.annames[j])
                            plt.plot(y)
                            xx, yy = linearFit.linspace()
                            print(xx, yy)
                            plt.plot(xx, yy)
                            plt.show()
                            # plotDebug(old, y, 0, y, yf, obs)

                xxSmoothness.append(smooth)

                # Samething for YY pol
                y1 = cal.phases[0, j, :, 3]
                y1 = movePhases(y1)

                y1 = interpChoices(x, y1, interp_type)
                yf1 = np.fft.fft(y1)
                smooth1 = np.average(abs(yf1[1 : int(nFreq / 2)]) / abs(yf1[0]))
                if normalise and j == 127:
                    smooth1 = 0

                yySmoothness.append(smooth1)

            xxSmoothnessAll.append(xxSmoothness)
            yySmoothnessAll.append(yySmoothness)

            if interp_type == "linear":
                allObsXXSmoothness.append(xxSmoothness)
                allObsYYSmoothness.append(yySmoothness)

            # Save figure for particular interp_type
            plt.plot(ant, xxSmoothness, label="XX", color="blue")
            plt.plot(ant, yySmoothness, label="YY", color="red")
            plt.xlabel("Antenna number")
            plt.ylabel("Smoothness")
            plt.title(obs + " " + interp_type)
            plt.legend()
            plt.xticks(np.linspace(0, 127, 128), minor=True)
            plt.grid()
            plt.grid(which="minor", alpha=0.5)
            plt.savefig(smoothDir + "/" + str(obs) + "_" + interp_type + ".pdf")
            plt.clf()

        # Save figure for all interp types
        xMax = np.nanmax(xxSmoothnessAll)
        yMax = np.nanmax(yySmoothnessAll)
        ls = ["solid", "dashed", "dotted"]
        interp_label = ["zero", "linear", "cspline"]
        legend_lines = list()
        for n in range(0, len(interps)):
            plt.plot(ant, xxSmoothnessAll[n], color="blue", linestyle=ls[n])
            plt.plot(ant, yySmoothnessAll[n], color="red", linestyle=ls[n])
            ax = plt.gca()
            (temp,) = ax.plot(
                0, -1, color="grey", linestyle=ls[n], label=interp_label[n]
            )
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


def plotDebug(old, yreal, yimag, y, yf, obs):
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

    Returns
    -------
    None

    -------
    """
    if type(yimag) is not int:
        smooth = np.average(abs(yf[1 : int(3072 / 2)]) / abs(yf[0]))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ax1.plot(old.real, "r.", alpha=0.5, markersize=0.75, label="old")
        ax1.plot(yreal, linewidth=0.5, label="interp")
        ax1.set_title(obs + " amps solutions real")
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
        ax1.set_title(obs + " phase solutions")
        ax1.legend()
        ax2.plot(abs(yf))
        ax2.set_title(f"Absolute value of fourier transform {smooth}")
        plt.show()

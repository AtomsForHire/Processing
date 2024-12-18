import json
import math

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mwa_qa import cal_metrics, read_calfits
from numba import jit, njit
from numpy.polynomial import Polynomial
from scipy import signal, stats
from tqdm import tqdm


def calVar(obsids, varDir, solDir):
    """Function for getting variance of gain amplitude calibration solutions through mwa_qa

    Parameters
    ----------
    - obsids: `list`
        List of observation ids
    - varDir: `string`
        Path to save results to
    - solDir: `string`
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
    - obsids: `list`
        List of observation ids
    - varDir: `string`
        Path to save results to
    - solDir: `string`
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


@njit
def nan_helper(y):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    - y: `numpy` array
        1d numpy array with possible NaNs

    Returns
    -------
    - nans: `numpy` array
        logical indices of NaNs
    - index: `numpy` array
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


@njit(cache=True)
def interpChoices(x, y, interp_type):
    """Function for interpolating with different styles

    Parameters
    ----------
    - x: `list`
        x axis
    - y: `list`
        y axis
    - interp_type: `string`
        Choice of interpolation

    Returns
    -------
    - y: `list`
        Interpolated function

    -------
    """
    if interp_type == "linear":
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    if interp_type == "zero":
        y = np.array([0 if math.isnan(x) else x for x in y])

    # if interp_type == "cspline":
    #     nans = np.isnan(y)
    #     x_interp = x[~nans]
    #     y_interp = y[~nans]
    #     cs = sci.interpolate.CubicSpline(x_interp, y_interp)
    #     y_missing = cs(x[nans])
    #     y[nans] = y_missing

    return y


def plotDebug(median, old, yreal, y, yf, obs, ant, pol):
    """Function to use when debugging the smoothness parameter

    Parameters
    ----------
    - old: `array`
        Contains the real and imaginary parts of the original calibration solutions
    - yreal: `array`/int
        Interpolated real part of the original calibration sol
    - yf: `array`
        Fourier transformed yreal + 1j*yimag
    - obs: `string`
        String for observation id
    - ant: `integer`
        antenna number
    - pol: `string`
        xx or yy polarization

    Returns
    -------
    None

    -------
    """
    smooth = np.average(abs(yf[1 : int(len(yreal) / 2)]) / abs(yf[0]))
    # smooth = np.average(abs(yf[1 : int(3072 / 2)]) / abs(yf[0]))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(old.real, "r.", alpha=0.5, markersize=1, label="old")
    axs[0, 0].plot(median.real, "b.", alpha=0.5, markersize=1, label="median")
    axs[0, 0].set_title(
        f"{obs} {pol} antenna {str(ant)} smallest median {np.nanmin(np.abs(median.real))}"
    )
    axs[0, 0].legend()

    axs[0, 1].plot(yreal, linewidth=0.5, label="interp")
    axs[0, 1].set_title("Interpolated and normalised")

    axs[1, 0].plot(abs(y), linewidth=0.5)
    axs[1, 0].set_title("absolute value")

    axs[1, 1].plot(abs(yf))
    axs[1, 1].set_title(f"Absolute value of fourier transform {smooth}")
    print(smooth)
    plt.show()


@njit(cache=True)
def movePhases(phases, threshold=180.0, window_size=5):
    """Function to make phases not wrap

    Parameters
    ----------
    - phases: `array`
        array of phases to process

    Returns
    -------
    - phases: `array`
        modified phases
    """

    phases = phases.copy()  # Work on a copy to preserve original

    # Find first valid point to start from
    valid_mask = ~np.isnan(phases)
    if not np.any(valid_mask):
        return phases  # All NaN case

    # First pass: identify and handle outliers
    for i in range(window_size, len(phases)):
        if np.isnan(phases[i]):
            continue

        # Get median of previous valid points within window
        prev_window = phases[max(0, i - window_size) : i]
        prev_median = np.nanmedian(prev_window)

        # Check if current point is an outlier
        diff = phases[i] - prev_median
        if abs(diff) > 50:
            # Mark outlier with NaN so it doesn't affect subsequent unwrapping
            phases[i] = np.nan

    prevAngle = phases[valid_mask][0]  # Start from first valid point

    for i in range(1, len(phases)):
        currAngle = phases[i]
        if np.isnan(currAngle):
            continue

        if currAngle - prevAngle > threshold:
            currAngle -= 360.0
        if currAngle - prevAngle < -threshold:
            currAngle += 360.0

        prevAngle = currAngle
        phases[i] = currAngle

    return phases


def phaseFit(x, y, interp_type, obs, ant, norm, debug, debugTargetObs, debugTargetAnt):
    """Function for calculating RMSE of the phase solutions and cubic and quadratic coeffs

    Parameters
    ----------
    - x: `array`
        frequency array
    - y: `array`
        phase solutions

    Returns
    -------
    - rmse: `float`
        root mean squared error

    """

    # "Real" data

    y = interpChoices(x, y, interp_type)
    # Do linear fit stats
    # NOTE: For some absolutely non-sensical reason numpy has a domain
    # and window parameter. By default the domain is the domain of the
    # x-values, but window has default value of [-1, 1]. This means
    # the fitting is done in the correct domain, but outputting the
    # coefs are in the window-domain. SO fucking stupid.

    # "Predicted" data
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
        if debugTargetObs is None:
            if ant in debugTargetAnt:
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
        elif obs in debugTargetObs:
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

    return (
        rmse,
        mad,
        linearFit.coef[1],
        (cubicFit.coef[1], cubicFit.coef[2], cubicFit.coef[3]),
    )


def calcSmooth(
    median,
    x,
    old,
    y,
    interp_type,
    obs,
    ant,
    norm,
    debug,
    debugTargetObs,
    debugTargetAnt,
    pol,
):
    """Function for calculating the smoothness of calibration solutions

    Parameters
    ----------
    - x: `array`
        contains antenna numbers
    - old: `array`
        contains the original calibration solutions for debugging purposes
    - y: `array`
        non-interpolated real part of the calibration solutions
    - interp_type: `string`
        interpolation method used
    - obs: `string`
        observation id
    - ant: `int`
        antenna number
    - norm: `bool`
        if using the normalise calibrations then skip antenna 127
    - debug: `bool`
        print and plot debug stuff
    - debugTargetObs: `list`
        list of target observations for debugging
    - debugTargetAnt: `list`
        list of target antennas for debugging

    Returns
    -------
    - smooth: `float`
        a value representing the smoothness of the calibration solutions

    -------
    """

    y = interpChoices(x, y, interp_type)
    yf = np.fft.fft(y)
    smooth = np.average(abs(yf[1 : int(len(x) / 2)]) / abs(yf[0]))

    if debug:
        # Plot desired antenna for all obs
        if debugTargetObs is None:
            if ant in debugTargetAnt:
                plotDebug(median, old, y, y, yf, obs, ant, pol)
        # Plot specific obs
        elif obs in debugTargetObs:
            if debugTargetAnt is None:
                plotDebug(median, old, y, y, yf, obs, ant, pol)
            elif ant in debugTargetAnt:
                plotDebug(median, old, y, y, yf, obs, ant, pol)

    return smooth


@njit(cache=True)
def getEuclidSame(xx, yy):
    copy = xx.copy()
    copy -= xx[0] - yy[0]

    return np.abs(np.mean(copy - yy))


def getKsTest(xx, yy):
    copy = xx.copy()
    copy -= xx[0] - yy[0]
    result = stats.ks_2samp(copy, yy, alternative="two-sided")
    return (result[0], result[1])


def getAnderson(xx, yy, ant, obs, debug, debugTargetObs, debugTargetAnt):
    x = np.linspace(0, 3072, len(xx))
    copy = xx.copy()
    copy -= xx[0] - yy[0]
    rng = np.random.default_rng()
    if debug:
        if debugTargetObs is None:
            if ant in debugTargetAnt:
                plt.plot(x, copy, "b.", label="XX")
                plt.plot(x, yy, "r.", label="YY")
                plt.title(f"{obs} antenna {ant}")
                plt.show()
        elif obs in debugTargetObs:
            if debugTargetAnt is None:
                plt.plot(x, copy, "b.", label="XX")
                plt.plot(x, yy, "r.", label="YY")
                plt.title(f"{obs} antenna {ant}")
                plt.show()
            elif ant in debugTargetAnt:
                plt.plot(x, copy, "b.", label="XX")
                plt.plot(x, yy, "r.", label="YY")
                plt.title(f"{obs} antenna {ant}")
                plt.show()
    return stats.anderson_ksamp(
        [copy, yy], method=stats.PermutationMethod(n_resamples=5, random_state=rng)
    )


def saveNormalised(xxNormalised, yyNormalised, obsids, name):
    if name == "amplitude":
        with h5py.File("amp_median_normalised.h5", "w") as f:
            f.create_dataset("xx_median_normalised", data=xxNormalised)
            f.create_dataset("yy_median_normalised", data=yyNormalised)
            f.create_dataset("obsids", data=obsids)
    elif name == "phase":
        with h5py.File("phase_median_normalised.h5", "w") as f:
            f.create_dataset("xx_median_normalised", data=xxNormalised)
            f.create_dataset("yy_median_normalised", data=yyNormalised)
            f.create_dataset("obsids", data=obsids)


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
    - obsids: `list`
        List of observation ids
    - smoothDir: `string`
        Path to save results to
    - distribution: `string`
        How the obs are sorted
    - gridDict: `dictionary`
        Dictionary where keys are obs ids and values are their grid number
    - uniqueDict: `dictionary`
        Dictionary of unique grid numbers and how often they occur
    - normalise: `string`
        "mean" or "median", decides which to use for normalisation

    Returns
    -------
    None

    -------
    """
    # These two are lists of lists
    xxAllCalPerObsList = []
    yyAllCalPerObsList = []

    ###########################################################################
    max_num_tiles = 0
    xxMeanOrMedianSolutionPerObs = []
    yyMeanOrMedianSolutionPerObs = []
    nFreqPerObs = []
    for i in range(0, len(obsids)):
        obs = obsids[i]
        filename = solDir + "/" + obs + "_solutions.fits"

        # These are a list of the calibration solutions for each antenna
        xxObsCals = []
        yyObsCals = []
        cal = read_calfits.CalFits(filename, norm=False)
        nFreqPerObs.append(cal.Nchan)

        if len(cal.gain_array[0, :, 0, 0]) > max_num_tiles:
            max_num_tiles = len(cal.gain_array[0, :, 0, 0])

        # Loop through antennas
        for j in range(0, len(cal.gain_array[0, :, 0, 0])):
            xxObsCals.append(np.abs(cal.gain_array[0, j, :, 0].copy()))
            yyObsCals.append(np.abs(cal.gain_array[0, j, :, 3].copy()))

        xxMeanOrMedianSolutionPerObs.append(np.nanmedian(xxObsCals, axis=0))
        yyMeanOrMedianSolutionPerObs.append(np.nanmedian(yyObsCals, axis=0))
        xxAllCalPerObsList.append(xxObsCals)
        yyAllCalPerObsList.append(yyObsCals)

    ###########################################################################

    eps = 10 ** (-10)
    allObsXXSmoothness = list()
    allObsYYSmoothness = list()
    allObsXXNormalised = list()
    allObsYYNormalised = list()
    allXXAvgSmooth = list()
    allYYAvgSmooth = list()
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        nFreq = nFreqPerObs[i]
        x = np.linspace(0, nFreq - 1, nFreq)

        if useWindow is True:
            window = signal.windows.blackmanharris(nFreq)
        else:
            window = np.ones(nFreq)

        xxSmoothness = list()
        yySmoothness = list()
        obsXXNormalised = []
        obsYYNormalised = []

        # Loop over antennas
        for j in range(0, len(xxAllCalPerObsList[i])):
            # Extract amplitudes for XX pol
            xxOld = xxAllCalPerObsList[i][j].copy()

            xxNormalised = xxAllCalPerObsList[i][j].copy()
            xxMedian = xxMeanOrMedianSolutionPerObs[i]
            xxNormalised[xxMedian > eps] /= xxMedian[xxMedian > eps]
            obsXXNormalised.append(xxNormalised.copy())
            xxNormalised *= window

            # Samething for YY pol
            yyOld = yyAllCalPerObsList[i][j].copy()

            yyNormalised = yyAllCalPerObsList[i][j].copy()
            yyMedian = yyMeanOrMedianSolutionPerObs[i]
            yyNormalised[yyMedian > eps] /= yyMedian[yyMedian > eps]
            obsYYNormalised.append(yyNormalised.copy())
            yyNormalised *= window

            # Skip flagged antennas
            if np.nansum(xxNormalised) == 0.0:
                xxSmoothness.append(np.nan)
                yySmoothness.append(np.nan)
                continue

            smooth = calcSmooth(
                xxMeanOrMedianSolutionPerObs[i],
                x,
                xxOld,
                xxNormalised,
                "linear",
                obs,
                j,
                normalise,
                debug,
                debugTargetObs,
                debugTargetAnt,
                "xx",
            )

            xxSmoothness.append(smooth)

            smooth1 = calcSmooth(
                yyMeanOrMedianSolutionPerObs[i],
                x,
                yyOld,
                yyNormalised,
                "linear",
                obs,
                j,
                normalise,
                debug,
                debugTargetObs,
                debugTargetAnt,
                "yy",
            )
            yySmoothness.append(smooth1)

        # If the observation has less than the maximum number of tiles
        # pad with NaNs at the end of array.
        if len(obsXXNormalised) < max_num_tiles:
            diff = max_num_tiles - len(obsXXNormalised)
            obsXXNormalised.extend([np.full(3072, np.nan)] * diff)
            obsYYNormalised.extend([np.full(3072, np.nan)] * diff)
            xxSmoothness.extend([np.nan] * diff)
            yySmoothness.extend([np.nan] * diff)

        allObsXXNormalised.append(obsXXNormalised)
        allObsYYNormalised.append(obsYYNormalised)
        allObsXXSmoothness.append(xxSmoothness)
        allObsYYSmoothness.append(yySmoothness)
        allXXAvgSmooth.append(np.nanmean(xxSmoothness))
        allYYAvgSmooth.append(np.nanmean(yySmoothness))

    saveNormalised(
        np.array(allObsXXNormalised),
        np.array(allObsYYNormalised),
        np.array(obsids, dtype="S"),
        "amplitude",
    )

    return (
        allObsXXSmoothness,
        allObsYYSmoothness,
        allXXAvgSmooth,
        allYYAvgSmooth,
        allObsXXNormalised,
        allObsYYNormalised,
    )


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
    useWindow=False,
):
    """Function for calculating smoothness of calibration phase amplitudes for all obs

    Parameters
    ----------
    - obsids: `list`
        List of observation ids
    - smoothDir: `string`
        Path to save results to
    - distribution: `string`
        How the obs are sorted
    - gridDict: `dictionary`
        Dictionary where keys are obs ids and values are their grid number
    - uniqueDict: `dictionary`
        Dictionary of unique grid numbers and how often they occur
    - normalise: `string`
        "mean" or "median", decides which to use for normalisation

    Returns
    -------
    None

    -------
    """

    ###########################################################################
    max_num_tiles = 0
    xxAllPhasePerObsList = []
    yyAllPhasePerObsList = []

    nFreqPerObs = []
    # Loop obsid and read in fits file
    for i in range(0, len(obsids)):
        obs = obsids[i]
        filename = solDir + "/" + obs + "_solutions.fits"

        # These are a list of the calibration solutions for each antenna in one obs
        xxObsCals = []
        yyObsCals = []
        cal = read_calfits.CalFits(filename, norm=False)
        nFreqPerObs.append(cal.Nchan)

        if len(cal.gain_array[0, :, 0, 0]) > max_num_tiles:
            max_num_tiles = len(cal.gain_array[0, :, 0, 0])

        for j in range(0, len(cal.gain_array[0, :, 0, 0])):
            xxObsCals.append(cal.phases[0, j, :, 0].copy())
            yyObsCals.append(cal.phases[0, j, :, 3].copy())

        xxAllPhasePerObsList.append(xxObsCals)
        yyAllPhasePerObsList.append(yyObsCals)

    allObsXXRMSE = list()
    allObsYYRMSE = list()
    allObsXXMAD = list()
    allObsYYMAD = list()
    allObsXXCubic = list()
    allObsYYCubic = list()
    allObsXXQuad = list()
    allObsYYQuad = list()
    allObsEuclidSame = list()
    allObsPval = list()
    allObsKsTest = list()
    allObsAndersonTest = list()
    allObsXXNormalised = list()
    allObsYYNormalised = list()

    # Loop through observations
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        nFreq = nFreqPerObs[i]
        x = np.linspace(0, nFreq - 1, nFreq)

        xxRMSE = list()
        yyRMSE = list()
        xxMAD = list()
        yyMAD = list()
        xxCubic = list()
        yyCubic = list()
        xxQuad = list()
        yyQuad = list()
        euclidSame = list()
        pVal = list()
        ksTest = list()
        andersonTest = list()
        obsXXNormalised = []
        obsYYNormalised = []

        # Loop through antennas
        for j in range(0, len(xxAllPhasePerObsList[i])):
            # 'old' for debugging purposes

            xxOld = xxAllPhasePerObsList[i][j].copy()
            xx = xxAllPhasePerObsList[i][j].copy()
            xx = movePhases(xx)

            yyOld = yyAllPhasePerObsList[i][j].copy()
            yy = yyAllPhasePerObsList[i][j].copy()
            yy = movePhases(yy)

            obsYYNormalised.append(yy)
            obsXXNormalised.append(xx)

            # Skip flagged antennas
            if np.nansum(xx) == 0.0:
                xxRMSE.append(np.nan)
                yyRMSE.append(np.nan)
                xxMAD.append(np.nan)
                yyMAD.append(np.nan)
                xxCubic.append(np.nan)
                yyCubic.append(np.nan)
                xxQuad.append(np.nan)
                yyQuad.append(np.nan)
                euclidSame.append(np.nan)
                pVal.append(np.nan)
                ksTest.append(np.nan)
                andersonTest.append(np.nan)
                continue

            rmse, mad, grad, coeffs = phaseFit(
                x,
                xx,
                "linear",
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

            rmse1, mad1, grad1, coeffs1 = phaseFit(
                x,
                yy,
                "linear",
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

            # Similarity stuff here ####

            euclidSame.append(getEuclidSame(xx, yy))

            tmp = getKsTest(xx, yy)
            pVal.append(tmp[1])
            ksTest.append(tmp[0])
            tmp2 = getAnderson(xx, yy, j, obs, debug, debugTargetObs, debugTargetAnt)
            andersonTest.append(tmp2.statistic)

            if debug:
                if debugTargetObs is None:
                    if j in debugTargetAnt:
                        plt.plot(x, xxOld, "b.", label="XX old")
                        plt.plot(x, yyOld, "r.", label="YY old")
                        plt.title("Old solutions")
                        plt.show()

                        plt.plot(x, xx, "b.", label="XX")
                        plt.plot(x, yy, "r.", label="YY")
                        plt.title(
                            f"{obs} antenna {j} KS {tmp[0]} Anderson {tmp2.statistic}"
                        )
                        plt.show()
                elif obs in debugTargetObs:
                    if debugTargetAnt is None:
                        plt.plot(x, xxOld, "b.", label="XX old")
                        plt.plot(x, yyOld, "r.", label="YY old")
                        plt.title("Old solutions")
                        plt.show()

                        plt.plot(x, xx, "b.", label="XX")
                        plt.plot(x, yy, "r.", label="YY")
                        plt.title(
                            f"{obs} antenna {j} KS {tmp[0]} Anderson {tmp2.statistic}"
                        )
                        plt.show()
                    elif j in debugTargetAnt:
                        plt.plot(x, xxOld, "b.", label="XX old")
                        plt.plot(x, yyOld, "r.", label="YY old")
                        plt.title("Old solutions")
                        plt.show()

                        plt.plot(x, xx, "b.", label="XX")
                        plt.plot(x, yy, "r.", label="YY")
                        plt.title(
                            f"{obs} antenna {j} KS {tmp[0]} Anderson {tmp2.statistic}"
                        )
                        plt.show()

        if len(obsXXNormalised) < max_num_tiles:
            diff = max_num_tiles - len(obsXXNormalised)
            obsXXNormalised.extend([np.full(3072, np.nan)] * diff)
            obsYYNormalised.extend([np.full(3072, np.nan)] * diff)
            xxRMSE.extend([np.nan] * diff)
            yyRMSE.extend([np.nan] * diff)
            xxMAD.extend([np.nan] * diff)
            yyMAD.extend([np.nan] * diff)
            xxQuad.extend([np.nan] * diff)
            yyQuad.extend([np.nan] * diff)
            euclidSame.extend([np.nan] * diff)
            pVal.extend([np.nan] * diff)
            ksTest.extend([np.nan] * diff)
            andersonTest.extend([np.nan] * diff)

        allObsXXRMSE.append(xxRMSE)
        allObsYYRMSE.append(yyRMSE)
        allObsXXMAD.append(xxMAD)
        allObsYYMAD.append(yyMAD)
        allObsXXCubic.append(xxCubic)
        allObsYYCubic.append(yyCubic)
        allObsXXQuad.append(xxQuad)
        allObsYYQuad.append(yyQuad)
        allObsEuclidSame.append(euclidSame)
        allObsPval.append(pVal)
        allObsKsTest.append(ksTest)
        allObsAndersonTest.append(andersonTest)
        allObsXXNormalised.append(obsXXNormalised)
        allObsYYNormalised.append(obsYYNormalised)

    saveNormalised(
        np.array(allObsXXNormalised),
        np.array(allObsYYNormalised),
        np.array(obsids, dtype="S"),
        "phase",
    )

    return (
        allObsXXRMSE,
        allObsYYRMSE,
        allObsXXMAD,
        allObsYYMAD,
        allObsEuclidSame,
        allObsKsTest,
        allObsAndersonTest,
    )

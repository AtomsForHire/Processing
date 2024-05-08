import matplotlib.pyplot as plt
import shutil
import math
import numpy as np
import os
import re
import scipy as sci
from mwa_qa import cal_metrics
from mwa_qa import read_calfits
from tqdm import tqdm
from pathlib import Path
import json
import sys


def getDirs():
    """

    """
    file = 'data.in'

    if not Path(file).exists():
        sys.exit('data.in file does not exist')

    with open(file) as f:
        temp = f.read().splitlines()

    statsDir = temp[0]
    rmsDir = temp[1]
    varDir = temp[2]
    smoothDir = temp[3]
    solDir = temp[4]
    distribution = temp[5].split(' ')[0]

    if (distribution == "cyclic"):
        if (len(temp[5].split()) != 2):
            sys.exit('Please enter a period (T) for your observations')
        period = temp[5].split()[1]
        if (len(temp) != 6 + int(period)):
            sys.exit(
                f'You have put down the observations as cyclicly distributed with a period of {period}, but have not included that many pointing center labels')
        pointingCentres = list()

        for i in range(1, int(period) + 1):
            pointingCentres.append(temp[5 + i])

    # Else I assume the observations should be in increasing order, and thus
    # pointing centers do not matter
    elif (distribution == 'sorted'):
        pointingCentres = []
    else:
        sys.exit(
            'Please input either \'sorted\' or \'cyclic\' distribution of observations')

    return statsDir, rmsDir, varDir, smoothDir, solDir, distribution, pointingCentres


def getRMS(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')
        file.close()

    m = re.search("RMS\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def getMax(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')
        file.close()

    m = re.search("Max\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def getMin(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')
        file.close()

    m = re.search("Min\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def getObsVec(directory, distribution):
    point1 = list()
    point2 = list()
    point3 = list()

    temp = list()

    # Grab all obsid and sort
    for file in os.listdir(directory):
        if os.path.isfile(directory + file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            temp.append(obsid)

    temp = sorted(temp)
    result = temp

    if (distribution == 'cyclic'):
        # Sort all obsid into groups with different pointing centres
        for i in range(0, len(temp) - 2, 3):
            # print(i)
            point1.append(temp[i])
            point2.append(temp[i + 1])
            point3.append(temp[i + 2])

        result = point1 + point2 + point3

    return result


def getRMSVec(directory, obsids):
    rmsVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            rms = getRMS(file)

            rmsVec[obsids.index(obsid)] = rms
            # if (point1.count(obsid) == 1):
            #     rms1[point1.index(obsid)] = rms
            # elif (point2.count(obsid) == 1):
            #     rms2[point2.index(obsid)] = rms
            # else:
            #     rms3[point3.index(obsid)] = rms

    return rmsVec


def getMaxVec(directory, obsids):
    maxVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            max = getMax(file)

            maxVec[obsids.index(obsid)] = max
            # if (point1.count(obsid) == 1):
            #     max1[point1.index(obsid)] = max
            # elif (point2.count(obsid) == 1):
            #     max2[point2.index(obsid)] = max
            # else:
            #     max3[point3.index(obsid)] = max

    return maxVec


def getDRVec(directory, obsids):
    drVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            max = getMax(file)
            rms = getRMS(file)
            dr = max/rms

            drVec[obsids.index(obsid)] = dr

    return drVec


def calVar(obsids, varDir, solDir):
    # calfits_path = '../solutions/1195576456_solutions.fits'
    # metafits_path = '../solutions/1195576456.metafits'

    for i in tqdm(range(0, len(obsids))):
        obsid = obsids[i]
        calfitsPath = solDir + str(obsid) + '_solutions.fits'
        metfitsPath = solDir + str(obsid) + '.metafits'

        calObj = cal_metrics.CalMetrics(calfitsPath, metfitsPath)
        calObj.run_metrics()
        plt.plot(calObj.variance_for_baselines_less_than(1)
                 [0, :, 0], label='XX', marker='.')
        plt.plot(calObj.variance_for_baselines_less_than(1)
                 [0, :, 3], label='YY', marker='.')
        plt.title(obsid + ' XX + YY')
        plt.xlabel('Antenna')
        plt.ylabel('Variance')
        plt.legend()
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which='minor', alpha=0.5)
        plt.savefig(varDir + obsid + '_var.pdf', bbox_inches='tight')
        plt.clf()
        calObj.write_to()


def calRMS(obsids, rmsDir, solDir):
    for i in tqdm(range(0, len(obsids))):
        metPath = solDir + \
            obsids[i] + '_solutions_cal_metrics.json'
        with open(metPath) as f:
            calibration = json.load(f)

        plt.plot(calibration["XX"]["RMS"], label='XX', marker='.')
        plt.plot(calibration["YY"]["RMS"], label='YY', marker='.')
        plt.xlabel("Antenna")
        plt.ylabel("RMS")
        plt.title(obsids[i] + ' XX + YY')
        plt.legend()
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which='minor', alpha=0.5)
        plt.savefig(rmsDir + obsids[i] + '_rms.pdf', bbox_inches='tight')
        plt.clf()


def nan_helper(y):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpChoices(x, y, interp_type):
    if (interp_type == "linear"):
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    if (interp_type == "zero"):
        y = np.array([0 if math.isnan(x) else x for x in y])

    if (interp_type == "cspline"):
        nans = np.isnan(y)
        x_interp = x[~nans]
        y_interp = y[~nans]
        cs = sci.interpolate.CubicSpline(x_interp, y_interp)
        y_missing = cs(x[nans])
        y[nans] = y_missing

    return y


def calAmpSmoothness(obsids, solDir, smoothDir, labels):
    x = np.linspace(0, 3073, 3072)
    ant = np.linspace(0, 127, 128)
    # interps = ['zero', 'linear', 'cspline']
    interps = ['zero', 'linear']
    allObsXXSmoothness = list()
    allObsYYSmoothness = list()
    for i in tqdm(range(0, len(obsids))):
        obs = obsids[i]
        filename = solDir + obs + "_solutions.fits"
        cal = read_calfits.CalFits(filename)

        xxSmoothnessAll = list()
        yySmoothnessAll = list()
        for interp_type in interps:
            xxSmoothness = list()
            yySmoothness = list()
            # Loop over antennas
            for j in range(0, len(cal.gain_array[0, :, 0, 0])):
                # Extract amplitudes for XX pol
                # old = cal.gain_array[0, j, :, 0].copy()
                yreal = cal.gain_array[0, j, :, 0].real.copy()
                yimag = cal.gain_array[0, j, :, 0].imag.copy()

                # Skip flagged antennas
                if ((np.nansum(yimag) == 0.0) and (np.nansum(yreal) == 0.0)):
                    xxSmoothness.append(np.nan)
                    yySmoothness.append(np.nan)
                    # print("SKIPPED, ", j)
                    continue

                yreal = interpChoices(x, yreal, interp_type)
                yimag = interpChoices(x, yimag, interp_type)
                y = yreal + 1.0j * yimag
                yf = np.fft.fft(y)
                # plt.plot(yreal, label="real", alpha=0.5)
                # plt.plot(yimag, label="imag", alpha=0.5)
                # if (np.nansum(yreal) == sum(yreal)):
                #     print("TRUE")
                # if (np.sum(yreal) == np.nansum(yreal)):
                #     print("TRUE2")
                # if (sum(yreal) == 0.0):
                #     print("TRUE3")
                # print(j)
                # print(np.all(yreal == 0.0), sum(yreal))
                # print(np.any(yreal == np.nan))
                # print(np.any(yreal == float('nan')))
                # print(yreal[np.where(yreal != 0, True, False)])
                # print(np.all(yimag == 0), sum(yimag))
                # print(np.any(yimag == np.nan))
                # print(yf)
                smooth = np.average(abs(yf[1:int(3072/2)])/abs(yf[0]))

                # if (interp_type == 'linear' and j == 126):
                #     # smooth = np.average(abs(yf[100:-100])/abs(yf[0]))
                #     fig, (ax1, ax2, ax3) = plt.subplots(3)
                #     # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
                #     ax1.plot(old.real, 'r.', alpha=0.5,
                #              markersize=0.75, label='old')
                #     ax1.plot(yreal, linewidth=0.5, label='interp')
                #     ax1.set_title(obs + " amps solutions real")
                #     ax2.plot(old.imag, 'r.', alpha=0.5,
                #              markersize=0.75, label='old')
                #     ax2.plot(yimag, linewidth=0.5, label='interp')
                #     ax2.set_title(obs + " amps solutions imag")
                #     # ax1.set_ylim(0, 7)
                #     ax1.legend()
                #     ax3.plot(abs(yf))
                #     ax3.set_title(
                #         f'Absolute value of fourier transform {smooth}')
                #     # ax4.plot(yf.imag)
                #     # ax4.set_title('Fourier transform real')
                #     plt.show()
                xxSmoothness.append(smooth)

                # Samething for YY pol
                yreal1 = cal.gain_array[0, j, :, 3].real
                yimag1 = cal.gain_array[0, j, :, 3].imag
                yreal1 = interpChoices(x, yreal1, interp_type)
                yimag1 = interpChoices(x, yimag1, interp_type)
                y1 = yreal1 + 1.0j * yimag1
                yf1 = np.fft.fft(y1)
                smooth1 = np.average(abs(yf1[1:int(3072/2)])/abs(yf1[0]))
                yySmoothness.append(smooth1)

            xxSmoothnessAll.append(xxSmoothness)
            yySmoothnessAll.append(yySmoothness)

            if (interp_type == 'linear'):
                allObsXXSmoothness.append(xxSmoothness)
                allObsYYSmoothness.append(yySmoothness)

            # Save figure for particular interp_type
            plt.plot(ant, xxSmoothness, label="XX", color='blue')
            plt.plot(ant, yySmoothness, label="YY", color='red')
            plt.xlabel("Antenna number")
            plt.ylabel("Smoothness")
            plt.title(obs + " " + interp_type)
            plt.legend()
            plt.xticks(np.linspace(0, 127, 128), minor=True)
            plt.grid()
            plt.grid(which='minor', alpha=0.5)
            plt.savefig(smoothDir + str(obs) + "_" + interp_type + ".pdf")
            plt.clf()

        # Save figure for all interp types
        xMax = np.nanmax(xxSmoothnessAll)
        yMax = np.nanmax(yySmoothnessAll)
        ls = ['solid', 'dashed', 'dotted']
        interp_label = ['zero', 'linear', 'cspline']
        legend_lines = list()
        for n in range(0, len(interps)):
            plt.plot(ant, xxSmoothnessAll[n], color='blue', linestyle=ls[n])
            plt.plot(ant, yySmoothnessAll[n], color='red', linestyle=ls[n])
            ax = plt.gca()
            temp, = ax.plot(0, -1, color='grey',
                            linestyle=ls[n], label=interp_label[n])
            legend_lines.append(temp)

        l4, = ax.plot(0, -1, color='blue', label='XX')
        l5, = ax.plot(0, -1, color='red', label='YY')
        first_legend = ax.legend(handles=legend_lines, loc='upper right')
        ax.add_artist(first_legend)
        ax.legend(handles=[l4, l5], loc='upper left')
        plt.xlabel("Antenna number")
        plt.ylabel("Smoothness")
        plt.ylim(0, 1.15 * np.max((xMax, yMax)))
        plt.title(obs)
        plt.xticks(np.linspace(0, 127, 128), minor=True)
        plt.grid()
        plt.grid(which='minor', alpha=0.5)
        plt.savefig(smoothDir + str(obs) + "_all.pdf")
        plt.clf()

    # Save figure for all obsids XX
    linestyles = ['solid', 'dashed', 'dotted']
    j = 0
    for i in range(0, len(obsids)):
        obs = obsids[i]
        if (np.mod(i, 9) == 0):
            style = linestyles[j]
            j += 1

        plt.plot(ant, allObsXXSmoothness[i], label=obs, linestyle=style)

    ax = plt.gca()
    lgd = ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel('Antenna number')
    plt.ylabel('Smoothness')
    plt.xticks(np.linspace(0, 127, 128), minor=True)
    plt.grid()
    plt.grid(which='minor', alpha=0.5)
    plt.savefig(smoothDir + 'all_obs_xx_linear.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

    # Save figure for all obsids YY
    j = 0
    for i in range(0, len(obsids)):
        obs = obsids[i]
        if (np.mod(i, 9) == 0):
            style = linestyles[j]
            j += 1

        plt.plot(ant, allObsYYSmoothness[i], label=obs, linestyle=style)

    ax = plt.gca()
    lgd = ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel('Antenna number')
    plt.ylabel('Smoothness')
    plt.xticks(np.linspace(0, 127, 128), minor=True)
    plt.grid()
    plt.grid(which='minor', alpha=0.5)
    plt.savefig(smoothDir + 'all_obs_yy_linear.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.inf)
    np.set_printoptions(suppress=True, linewidth=np.nan)

    statsDir, rmsDir, varDir, smoothDir, solDir, distribution, pointingCentres = getDirs()
    print(distribution)
    print(pointingCentres)

    Path(rmsDir).mkdir(parents=True, exist_ok=True)
    Path(varDir).mkdir(parents=True, exist_ok=True)
    Path(smoothDir).mkdir(parents=True, exist_ok=True)

    # Group obsid
    obsids = getObsVec(statsDir, distribution)

    if (distribution == 'cyclic'):
        print('CYCLING DISTRIBUTION SELECTED')
        period = len(pointingCentres)
        idx = int(len(obsids)/period)
        print('FIRST POINTING CENTER : ', pointingCentres[0])
        print(obsids[0:idx])
        print('SECOND POINTING CENTER: ', pointingCentres[1])
        print(obsids[idx: 2*idx])
        print('THIRD POINTING CENTER : ', pointingCentres[2])
        print(obsids[2*idx:])

    # Get RMS for obs
    rms = getRMSVec(statsDir, obsids)

    if (distribution == 'cyclic'):
        idx = int(len(obsids)/period)
        for i in range(0, period - 1):
            plt.plot(obsids[i*idx:(i+1)*idx], rms[i*idx:(i+1)*idx],
                     label=pointingCentres[i])

        # Print the last pointing centre
        i += 1
        plt.plot(obsids[i*idx:], rms[i*idx:], label=pointingCentres[i])

        plt.legend()
    elif (distribution == 'sorted'):
        plt.plot(obsids, rms)

    plt.xticks(rotation=90)
    plt.ylabel("RMS (Jy/beam)")
    plt.savefig("obs_rms.pdf", bbox_inches='tight')
    plt.clf()

    # Get max for obs
    max = getMaxVec(statsDir, obsids)

    if (distribution == 'cyclic'):
        idx = int(len(obsids)/period)
        for i in range(0, period - 1):
            plt.plot(obsids[i*idx:(i+1)*idx], max[i*idx:(i+1)*idx],
                     label=pointingCentres[i])

        # Print the last pointing centre
        i += 1
        plt.plot(obsids[i*idx:], max[i*idx:], label=pointingCentres[i])

        plt.legend()
    elif (distribution == 'sorted'):
        plt.plot(obsids, max)
    plt.xticks(rotation=90)
    plt.ylabel("Maximum")
    plt.savefig("obs_max.pdf", bbox_inches='tight')
    plt.clf()

    # Get DR for obs
    dr = getDRVec(statsDir, obsids)

    if (distribution == 'cyclic'):
        idx = int(len(obsids)/period)
        for i in range(0, period - 1):
            plt.plot(obsids[i*idx:(i+1)*idx], dr[i*idx:(i+1)*idx],
                     label=pointingCentres[i])

        # Print the last pointing centre
        i += 1
        plt.plot(obsids[i*idx:], dr[i*idx:], label=pointingCentres[i])

        plt.legend()
    elif (distribution == 'sorted'):
        plt.plot(obsids, dr)

    plt.xticks(rotation=90)
    plt.ylabel("Dynamic Range (max/rms)")
    plt.savefig("obs_dynamic_range.pdf", bbox_inches='tight')
    plt.clf()

    # Attemp Ridhima's QA pipeline
    # print("Calibration variance")
    # calVar(obsids, varDir, solDir)
    #
    # print("Calibration RMS")
    # calRMS(obsids, rmsDir, solDir)

    print("AMP SMOOTHNESS")
    calAmpSmoothness(obsids, solDir, smoothDir, pointingCentres)

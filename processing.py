import collections
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits

import calibration
import image


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def getDirs(filename):
    """Function for reading in config yaml and return settings

    Parameters
    ----------
    filename: string
        Name of the yaml file

    Returns
    -------
    All settings

    -------
    """

    with open(filename) as f:
        temp = yaml.safe_load(f)
        if "stats_dir" in temp.keys():
            statsDir = temp["stats_dir"]
        else:
            sys.exit("PLEASE INCLUDE stats_dir IN CONFIG FILE")

        if "rms_dir" in temp.keys():
            rmsDir = temp["rms_dir"]
        else:
            sys.exit("PLEASE INCLUDE rms_dir IN CONFIG FILE")

        if "var_dir" in temp.keys():
            varDir = temp["var_dir"]
        else:
            sys.exit("PLEASE INCLUDE var_dir IN CONFIG FILE")

        if "smooth_dir" in temp.keys():
            smoothDir = temp["smooth_dir"]
        else:
            sys.exit("PLEASE INCLUDE smooth_dir IN CONFIG FILE")

        if "sol_dir" in temp.keys():
            solDir = temp["sol_dir"]
        else:
            sys.exit("PLEASE INCLUDE sol_dir IN CONFIG FILE")

        if "statistics" in temp.keys():
            stats = temp["statistics"]
        else:
            sys.exit("PLEASE INCLUDE statistics IN CONFIG FILE")

        if "obs_dist" in temp.keys():
            distribution = temp["obs_dist"]
        else:
            sys.exit("PLEASE INCLUDE obs_dist IN CONFIG FILE")

        if "exclude" in temp.keys():
            excludeList = temp["exclude"]
        else:
            sys.exit("PLEASE INCLUDE exclude IN CONFIG FILE")

    return statsDir, rmsDir, varDir, smoothDir, solDir, stats, excludeList, distribution


def getObsVec(directory, distribution):
    """Function for obtaining list of observations

    Parameters
    ----------
    directory: string
        Directory of solutions/metafits files
    distribution: string
        How the obsids should be ordered, at the moment only sorted works

    Returns
    -------
    result: list
        List of observation ids

    -------
    """
    point1 = list()
    point2 = list()
    point3 = list()

    temp = list()

    # Grab all obsid and sort
    for file in os.listdir(directory):
        if os.path.isfile(directory + "/" + file) and file.endswith("_solutions.fits"):
            obsid = file.split("_")[0]
            temp.append(obsid)

    temp = sorted(temp)
    result = temp

    if distribution == "cyclic":
        # Sort all obsid into groups with different pointing centres
        for i in range(0, len(temp) - 2, 3):
            # print(i)
            point1.append(temp[i])
            point2.append(temp[i + 1])
            point3.append(temp[i + 2])

        result = point1 + point2 + point3

    return result


def getGridNum(obsids, solDir):
    """Function for getting observation's grid numbers

    Parameters
    ----------
    obsids: list
        List of observation ids
    solDir: string
        Path to directory containing metafits files

    Returns
    -------
    gridDict: dictionary
        Dictionary where keys are obsids and values are their grid number

    -------
    """
    gridDict = {}
    for obs in obsids:
        with fits.open(solDir + "/" + str(obs) + ".metafits") as hdu:
            hdr = hdu["PRIMARY"].header
            gridDict[obs] = hdr["GRIDNUM"]

    return gridDict


if __name__ == "__main__":
    # np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.inf)
    np.set_printoptions(suppress=True, linewidth=np.nan)
    config = sys.argv[1]
    statsDir, rmsDir, varDir, smoothDir, solDir, stats, excludeList, distribution = (
        getDirs(config)
    )

    Path(rmsDir).mkdir(parents=True, exist_ok=True)
    Path(varDir).mkdir(parents=True, exist_ok=True)
    Path(smoothDir).mkdir(parents=True, exist_ok=True)

    # Group obsid
    obsids = getObsVec(solDir, distribution)
    if excludeList is not None:
        for i in range(0, len(excludeList)):
            obsids.remove(str(excludeList[i]))

    # Regardless of whether grid sorting is selected or not, create dictionary of
    # unique grid nums since they must be passed into the functions anyway
    # the functions themselves will decide if they use grid or other sorting
    gridDict = getGridNum(obsids, solDir)
    uniqueDict = collections.Counter(gridDict.values())
    # Sort by value
    gridDict = {k: v for k, v in sorted(gridDict.items(), key=lambda item: item[1])}

    print(f"{bcolors.OKBLUE}INCLUDED OBS{bcolors.ENDC}: {obsids}")
    print(f"{bcolors.OKBLUE}EXCLUDED OBS{bcolors.ENDC}: {excludeList}")
    print(
        f"{bcolors.OKBLUE}TOTAL NUMBER OF OBSERVATIONS BEING PROCESSED{bcolors.ENDC}: ",
        len(obsids),
    )
    print(f"{bcolors.OKBLUE}OBSERVATIONS AND THEIR GRIDNUM{bcolors.ENDC}: {gridDict}")
    print(
        f"{bcolors.OKBLUE}UNIQUE GRIDNUMS AND FREQUENCY {bcolors.ENDC}: {len(uniqueDict)} {uniqueDict}"
    )

    if stats == "image" or stats == "both":
        # Get RMS for obs
        rms = image.getRMSVec(statsDir, obsids, distribution, gridDict, uniqueDict)

        # Get max for obs
        max = image.getMaxVec(statsDir, obsids, distribution, gridDict, uniqueDict)

        # Get DR for obs
        dr = image.getDRVec(statsDir, obsids, distribution, gridDict, uniqueDict)

    if stats == "calibration" or stats == "both":
        # Attemp Ridhima's QA pipeline
        # print("Calibration variance")
        # calibration.calVar(obsids, varDir, solDir)
        #
        # print("Calibration RMS")
        # calibration.calRMS(obsids, rmsDir, solDir)

        print("AMP SMOOTHNESS")
        calibration.calAmpSmoothness(
            obsids, solDir, smoothDir, distribution, gridDict, uniqueDict
        )

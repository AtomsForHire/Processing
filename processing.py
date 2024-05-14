import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import os
import sys
import image
import calibration
import collections
from astropy.io import fits


def getDirs(filename):

    with open(filename) as f:
        temp = yaml.safe_load(f)
        if ("stats_dir" in temp.keys()):
            statsDir = temp["stats_dir"]
        else:
            sys.exit("PLEASE INCLUDE stats_dir IN CONFIG FILE")

        if ("rms_dir" in temp.keys()):
            rmsDir = temp["rms_dir"]
        else:
            sys.exit("PLEASE INCLUDE rms_dir IN CONFIG FILE")

        if ("var_dir" in temp.keys()):
            varDir = temp["var_dir"]
        else:
            sys.exit("PLEASE INCLUDE var_dir IN CONFIG FILE")

        if ("smooth_dir" in temp.keys()):
            smoothDir = temp["smooth_dir"]
        else:
            sys.exit("PLEASE INCLUDE smooth_dir IN CONFIG FILE")

        if ("sol_dir" in temp.keys()):
            solDir = temp["sol_dir"]
        else:
            sys.exit("PLEASE INCLUDE sol_dir IN CONFIG FILE")

        if ("statistics" in temp.keys()):
            stats = temp["statistics"]
        else:
            sys.exit("PLEASE INCLUDE statistics IN CONFIG FILE")

        if ("obs_dist" in temp.keys()):
            distribution = temp["obs_dist"]
        else:
            sys.exit("PLEASE INCLUDE obs_dist IN CONFIG FILE")

        if ("exclude" in temp.keys()):
            excludeList = temp["exclude"]
        else:
            sys.exit("PLEASE INCLUDE exclude IN CONFIG FILE")

    return statsDir, rmsDir, varDir, smoothDir, solDir, stats, excludeList, distribution


def getObsVec(directory, distribution):
    point1 = list()
    point2 = list()
    point3 = list()

    temp = list()

    # Grab all obsid and sort
    for file in os.listdir(directory):
        if os.path.isfile(directory + "/" + file) and file.endswith("_solutions.fits"):
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


def getGridNum(obsids, solDir):
    gridDict = {}
    for obs in obsids:
        with fits.open(solDir + '/' + str(obs) + '.metafits') as hdu:
            hdr = hdu['PRIMARY'].header
            gridDict[obs] = hdr['GRIDNUM']
            print(obs, hdr['GRIDNUM'])

    return gridDict


if __name__ == '__main__':
    # np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.inf)
    np.set_printoptions(suppress=True, linewidth=np.nan)
    config = sys.argv[1]
    statsDir, rmsDir, varDir, smoothDir, solDir, stats, excludeList, distribution = getDirs(
        config)

    Path(rmsDir).mkdir(parents=True, exist_ok=True)
    Path(varDir).mkdir(parents=True, exist_ok=True)
    Path(smoothDir).mkdir(parents=True, exist_ok=True)

    # Group obsid
    obsids = getObsVec(solDir, distribution)
    if (excludeList is not None):
        for i in range(0, len(excludeList)):
            obsids.remove(str(excludeList[i]))

    # If grid_num distribution is selected do some processing there
    gridDict = getGridNum(obsids, solDir)
    uniqueDict = collections.Counter(gridDict.values())

    print('INCLUDED OBS: ', obsids)
    print('EXCLUDED OBS: ', excludeList)
    print('TOTAL NUMBER OF OBSERVATIONS BEING PROCESSED: ', len(obsids))
    print('OBSERVATIONS AND THEIR GRIDNUM: ', gridDict)
    sys.exit()

    if (stats == 'image' or stats == 'both'):
        # Get RMS for obs
        rms = image.getRMSVec(statsDir, obsids, distribution)

        # Get max for obs
        max = image.getMaxVec(statsDir, obsids, distribution)

        # Get DR for obs
        dr = image.getDRVec(statsDir, obsids, distribution)

    if (stats == 'calibration' or stats == 'both'):
        # Attemp Ridhima's QA pipeline
        print("Calibration variance")
        calibration.calVar(obsids, varDir, solDir)

        print("Calibration RMS")
        calibration.calRMS(obsids, rmsDir, solDir)

        print("AMP SMOOTHNESS")
        calibration.calAmpSmoothness(obsids, solDir, smoothDir, distribution)

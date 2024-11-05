import collections
import os
import sys
from pathlib import Path

import calibration
import correlation
import h5py
import image
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits


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

        if "smooth_dir_amp" in temp.keys():
            smoothDirAmps = temp["smooth_dir_amp"]
        else:
            sys.exit("PLEASE INCLUDE smooth_dir_amp IN CONFIG FILE")

        if "smooth_dir_phase" in temp.keys():
            smoothDirPhase = temp["smooth_dir_phase"]
        else:
            sys.exit("PLEASE INCLUDE smooth_dir_phase IN CONFIG FILE")

        if "phase_stats_dir" in temp.keys():
            phaseStatsDir = temp["phase_stats_dir"]
        else:
            sys.exit("PLEASE INCLUDE phase_stats_dir IN CONFIG FILE")

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

        if "debug" in temp.keys():
            debug = temp["debug"]
        else:
            sys.exit("PLEASE INCLUDE debug IN CONFIG FILE")

        if "debug_obs" in temp.keys():
            debugObsList = temp["debug_obs"]
        else:
            sys.exit("PLEASE INCLUDE debug_obs IN CONFIG FILE")

        if "debug_ant" in temp.keys():
            debugAntList = temp["debug_ant"]
        else:
            sys.exit("PLEASE INCLUDE debug_ant IN CONFIG FILE")

        if "norm" in temp.keys():
            norm = temp["norm"]
        else:
            sys.exit("PLEASE INCLUDE norm IN CONFIG FILE")

        if "window" in temp.keys():
            window = temp["window"]
        else:
            sys.exit("PLEASE INCLUDE window IN CONFIG FILE")

        if "corr_dir" in temp.keys():
            corrDir = temp["corr_dir"]
        else:
            sys.exit("PLEASE INCLUDE corr_dir IN CONFIG FILE")

        if "img_dir" in temp.keys():
            imgDir = temp["img_dir"]
        else:
            sys.exit("PLEASE INCLUDE img_dir IN CONFIG FILE")

    return (
        statsDir,
        rmsDir,
        varDir,
        smoothDirAmps,
        smoothDirPhase,
        phaseStatsDir,
        solDir,
        imgDir,
        corrDir,
        stats,
        excludeList,
        distribution,
        window,
        debug,
        debugObsList,
        debugAntList,
        norm,
    )


def getObsVec(directory, distribution):
    """Function for obtaining list of observations

    Parameters
    ----------
    - directory: `string`
        Directory of solutions/metafits files
    - distribution: `string`
        How the obsids should be ordered, at the moment only sorted works

    Returns
    -------
    - result: `list`
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
    - obsids: `list`
        List of observation ids
    - solDir: `string`
        Path to directory containing metafits files

    Returns
    -------
    - gridDict: `dictionary`
        Dictionary where keys are obsids and values are their grid number

    -------
    """
    gridDict = {}
    for obs in obsids:
        with fits.open(solDir + "/" + str(obs) + ".metafits") as hdu:
            hdr = hdu["PRIMARY"].header
            gridDict[obs] = hdr["GRIDNUM"]

    return gridDict


def saveText(gridDict, stats, obsids, name):
    with open("text/" + name + ".txt", "w") as f:
        for i in range(0, len(obsids)):
            obs = obsids[i]
            stat = stats[i]
            gridNum = gridDict[obs]
            f.write(f"{obs} {gridNum} {stat}\n")


def saveHDF5(
    gridNums,
    obsids,
    rms,
    dr,
    xxGainSmoothness,
    yyGainSmoothness,
    xxPhaseRMSE,
    yyPhaseRMSE,
    xxPhaseMAD,
    yyPhaseMAD,
    phaseEuclidSame,
    phaseKs,
    phaseAnderson,
):
    with h5py.File("output.h5", "w") as f:
        # save obsids
        f.create_dataset("obsids", data=obsids)

        f.create_dataset("grid_nums", data=gridNums)

        # save image stats
        img = f.create_group("image_group")
        img.create_dataset("rms", data=rms)
        img.create_dataset("dr", data=dr)

        # Save gain stats
        gain = f.create_group("gain_group")
        gain.create_dataset("xxGainSmoothness", data=xxGainSmoothness)
        gain.create_dataset("yyGainSmoothness", data=yyGainSmoothness)

        # Save phase stats
        phase = f.create_group("phase_group")
        phase.create_dataset("xxPhaseRMSE", data=xxPhaseRMSE)
        phase.create_dataset("yyPhaseRMSE", data=yyPhaseRMSE)
        phase.create_dataset("xxPhaseMAD", data=xxPhaseMAD)
        phase.create_dataset("yyPhaseMAD", data=yyPhaseMAD)
        phase.create_dataset("phaseEuclidSame", data=phaseEuclidSame)
        phase.create_dataset("phaseKs", data=phaseKs)
        phase.create_dataset("phaseAnderson", data=phaseAnderson)


def main():
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.inf)
    # np.set_printoptions(suppress=True, linewidth=np.nan)
    config = sys.argv[1]
    (
        statsDir,
        rmsDir,
        varDir,
        smoothDirAmps,
        smoothDirPhase,
        phaseStatsDir,
        solDir,
        imgDir,
        corrDir,
        stats,
        excludeList,
        distribution,
        window,
        debug,
        debugObsList,
        debugAntList,
        norm,
    ) = getDirs(config)

    Path(rmsDir).mkdir(parents=True, exist_ok=True)
    Path(varDir).mkdir(parents=True, exist_ok=True)
    Path(smoothDirAmps).mkdir(parents=True, exist_ok=True)
    Path(smoothDirPhase).mkdir(parents=True, exist_ok=True)
    Path(phaseStatsDir).mkdir(parents=True, exist_ok=True)
    Path(corrDir).mkdir(parents=True, exist_ok=True)
    Path("text").mkdir(parents=True, exist_ok=True)

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

    # Get RMS for obs
    rms = image.getRMSVec(statsDir, obsids, distribution, gridDict, uniqueDict, imgDir)
    saveText(gridDict, rms, obsids, "rms")

    # Get max for obs
    max = image.getMaxVec(statsDir, obsids, distribution, gridDict, uniqueDict, imgDir)

    # Get DR for obs
    dr = image.getDRVec(statsDir, obsids, distribution, gridDict, uniqueDict, imgDir)
    saveText(gridDict, dr, obsids, "dr")
    gridNums = list()
    for i in range(0, len(obsids)):
        gridNums.append(gridDict[obsids[i]])

    # Ridhima's QA pipeline
    # print("Calibration variance")
    # calibration.calVar(obsids, varDir, solDir)
    #
    # print("Calibration RMS")
    # calibration.calRMS(obsids, rmsDir, solDir)

    print("AMP SMOOTHNESS")
    (
        xxGainSmoothness,
        yyGainSmoothness,
        xxAvgSmoothness,
        yyAvgSmoothness,
        allObsXXNormalised,
        allObsYYNormalised,
    ) = calibration.calAmpSmoothness(
        obsids,
        solDir,
        smoothDirAmps,
        distribution,
        gridDict,
        uniqueDict,
        debug,
        debugObsList,
        debugAntList,
        norm,
        window,
    )

    print("PHASE SMOOTHNESS")
    (
        xxPhaseRMSE,
        yyPhaseRMSE,
        xxPhaseMAD,
        yyPhaseMAD,
        phaseEuclidSame,
        phaseKs,
        phaseAnderson,
    ) = calibration.calPhaseSmoothness(
        obsids,
        solDir,
        smoothDirPhase,
        phaseStatsDir,
        distribution,
        gridDict,
        uniqueDict,
        debug,
        debugObsList,
        debugAntList,
        norm,
        useWindow=False,
    )

    saveHDF5(
        gridNums,
        obsids,
        rms,
        dr,
        xxGainSmoothness,
        yyGainSmoothness,
        xxPhaseRMSE,
        yyPhaseRMSE,
        xxPhaseMAD,
        yyPhaseMAD,
        phaseEuclidSame,
        phaseKs,
        phaseAnderson,
    )

    exit()

    print("CORRELATION")
    print("XX RMSE VS XX SMOOTH")
    print(len(xxPhaseRMSE), len(xxGainSmoothness))
    correlation.crossCorrAcrossAnt(
        xxGainSmoothness,
        xxPhaseRMSE,
        obsids,
        "xx smoothness",
        "xx phase RMSE",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "_xxRMSE_xxSmooth",
    )

    print("XX SMOOTH VS XX SMMOTH")
    correlation.crossCorrAcrossAnt(
        xxGainSmoothness,
        yyGainSmoothness,
        obsids,
        "xx smoothness",
        "yy smoothness",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "_yySmooth_xxSmooth",
    )

    print("YY RMSE VS XX RMSE")
    correlation.crossCorrAcrossAnt(
        xxPhaseRMSE,
        yyPhaseRMSE,
        obsids,
        "xx phase RMSE",
        "yy phase RMSE",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "_yyRMSE_xxRMSE",
    )

    print("XX MAD VS XX RMSE")
    correlation.crossCorrAcrossAnt(
        xxPhaseRMSE,
        xxPhaseMAD,
        obsids,
        "xx phase RMSE",
        "xx phase MAD",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "_xxMAD_xxRMSE",
    )

    print("EUCLID VS KS")
    correlation.crossCorrAcrossAnt(
        phaseEuclidSame,
        phaseKs,
        obsids,
        "phase euclidian distance",
        "phase KS-metric",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "_euclid_KS",
    )

    print("IMAGE RMS VS XX GAIN SMOOTHNESS")
    correlation.crossCorrAcrossObs(
        rms,
        xxAvgSmoothness,
        obsids,
        "Image RMS",
        "XX Average Smoothness",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "xxSmooth_rms",
    )

    print("IMAGE RMS VS YY GAIN SMOOTHNESS")
    correlation.crossCorrAcrossObs(
        rms,
        yyAvgSmoothness,
        obsids,
        "Image RMS",
        "YY Average Smoothness",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "yySmooth_rms",
    )

    print("IMAGE DR VS GAIN SMOOTHNESS")
    correlation.crossCorrAcrossObs(
        dr,
        xxAvgSmoothness,
        obsids,
        "Image Dynamic Range",
        "XX Average Smoothness",
        distribution,
        gridDict,
        uniqueDict,
        corrDir,
        "xxSmooth_dr",
    )

    # print("IMAGE RMS VS XX PHASE RMSE")
    # correlation.crossCorrAcrossObs(
    #     rms,
    #     xx,
    #     obsids,
    #     "Image Dynamic Range",
    #     "XX Average Smoothness",
    #     distribution,
    #     gridDict,
    #     uniqueDict,
    #     corrDir,
    #     "xxSmooth_dr",
    # )


if __name__ == "__main__":
    main()

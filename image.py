import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def getRMS(filename):
    """Function to get RMS from carta stats file

    Parameters
    ----------
    filename: string
        name of tsv file

    Returns
    -------
    rms: float
        rms of observation

    -------
    """
    with open(filename, "r") as file:
        data = file.read().replace("\n", " ")
        file.close()

    m = re.search("RMS\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def getMax(filename):
    """Function to get max from carta stats file

    Parameters
    ----------
    filename: string
        name of tsv file

    Returns
    -------
    max: float
        max of observation

    -------
    """
    with open(filename, "r") as file:
        data = file.read().replace("\n", " ")
        file.close()

    m = re.search("Max\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def getMin(filename):
    """Function to get min from carta stats file

    Parameters
    ----------
    filename: string
        name of tsv file

    Returns
    -------
    min: float
        min of observation

    -------
    """
    with open(filename, "r") as file:
        data = file.read().replace("\n", " ")
        file.close()

    m = re.search("Min\s+(.*?)\s+Jy/beam", data)

    return float(m.group(1))


def gridPlot(obsids, statVec, gridDict, uniqueDict):
    """Function for plotting image statistics group by grid number

    Parameters
    ----------
    obsids: list
        List of observations
    statVec: array
        Array containg whatever statistic for each observation (e.g. rms, max, etc.)
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur

    Returns
    -------
    None

    -------
    """

    # Create many combination of line and marker styles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    marker = [
        "d",
        ".",
        "s",
        "o",
        "*",
        "x",
    ]
    styles = {}

    # For some particular grid number, it will have it's own unique combination
    for i, key in enumerate(uniqueDict):
        styles[key] = [
            linestyles[i % len(linestyles)],
            marker[int(i // len(linestyles))],
        ]

    obs_legend_list = list()
    # Plot line for each unique grid value
    for key in uniqueDict:
        obsWithSameKey = [k for k, v in gridDict.items() if v == key]
        # Grab index of obs in obsids
        tempIdx = []
        for j, findThisObs in enumerate(obsWithSameKey):
            for i, obs in enumerate(obsids):
                if obs == findThisObs:
                    tempIdx.append(i)

        (temp,) = plt.plot(
            obsWithSameKey,
            statVec[tempIdx[:]],
            linestyle=styles[key][0],
            marker=styles[key][1],
            label=key,
        )
        obs_legend_list.append(temp)

    ax = plt.gca()
    obs_legend = ax.legend(
        handles=obs_legend_list, bbox_to_anchor=(1.04, 0.5), loc="center left"
    )

    return obs_legend


def getRMSVec(directory, obsids, distribution, gridDict, uniqueDict):
    """Function for getting list of rms values per observation

    Parameters
    ----------
    directory: string
        Path to where the carta statistics files are
    obsids: list
        List of observation ids
    distribution: string
        How the obsids are sorted
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur

    Returns
    -------
    None

    -------
    """
    rmsVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split("_")[0]

            if obsid not in obsids:
                continue

            rms = getRMS(file)

            rmsVec[obsids.index(obsid)] = rms

    if distribution == "grid":
        obs_legend = gridPlot(obsids, rmsVec, gridDict, uniqueDict)
    elif distribution == "sorted":
        plt.plot(obsids, rmsVec)

    plt.xticks(rotation=90)
    plt.ylabel("RMS (Jy/beam)")
    if distribution == "grid":
        plt.savefig(
            "obs_rms.pdf",
            bbox_extra_artists=([obs_legend]),
            bbox_inches="tight",
        )
    elif distribution == "sorted":
        plt.savefig("obs_rms.pdf", bbox_inches="tight")
    plt.clf()


def getMaxVec(directory, obsids, distribution, gridDict, uniqueDict):
    """Function for getting list of max values per observation

    Parameters
    ----------
    directory: string
        Path to where the carta statistics files are
    obsids: list
        List of observation ids
    distribution: string
        How the obsids are sorted
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur

    Returns
    -------
    None

    -------
    """
    maxVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split("_")[0]
            if obsid not in obsids:
                continue
            max = getMax(file)

            maxVec[obsids.index(obsid)] = max

    if distribution == "grid":
        gridPlot(obsids, maxVec, gridDict, uniqueDict)
    elif distribution == "sorted":
        plt.plot(obsids, maxVec)
    plt.xticks(rotation=90)
    plt.ylabel("Maximum (Jy/beam)")
    plt.savefig("obs_max.pdf", bbox_inches="tight")
    plt.clf()


def getDRVec(directory, obsids, distribution, gridDict, uniqueDict):
    """Function for getting list of dynamic range values per observation

    Parameters
    ----------
    directory: string
        Path to where the carta statistics files are
    obsids: list
        List of observation ids
    distribution: string
        How the obsids are sorted
    gridDict: dictionary
        Dictionary where keys are obs ids and values are their grid number
    uniqueDict: dictionary
        Dictionary of unique grid numbers and how often they occur

    Returns
    -------
    None

    -------
    """
    drVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split("_")[0]
            if obsid not in obsids:
                continue
            max = getMax(file)
            rms = getRMS(file)
            dr = max / rms

            drVec[obsids.index(obsid)] = dr

    if distribution == "grid":
        gridPlot(obsids, drVec, gridDict, uniqueDict)
    elif distribution == "sorted":
        plt.plot(obsids, drVec)

    plt.xticks(rotation=90)
    plt.ylabel("Dynamic Range (max/rms)")
    plt.savefig("obs_dynamic_range.pdf", bbox_inches="tight")
    plt.clf()

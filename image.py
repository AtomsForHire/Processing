import os
import matplotlib.pyplot as plt
import numpy as np
import re


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


def getRMSVec(directory, obsids, distribution):
    rmsVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]

            if (obsid not in obsids):
                continue

            rms = getRMS(file)

            rmsVec[obsids.index(obsid)] = rms

    if (distribution == 'cyclic'):
        pass
    elif (distribution == 'sorted'):
        plt.plot(obsids, rmsVec)

    plt.xticks(rotation=90)
    plt.ylabel("RMS (Jy/beam)")
    plt.savefig("obs_rms.pdf", bbox_inches='tight')
    plt.clf()


def getMaxVec(directory, obsids, distribution):
    maxVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            if (obsid not in obsids):
                continue
            max = getMax(file)

            maxVec[obsids.index(obsid)] = max

    if (distribution == 'cyclic'):
        pass
    elif (distribution == 'sorted'):
        plt.plot(obsids, maxVec)
    plt.xticks(rotation=90)
    plt.ylabel("Maximum (Jy/beam)")
    plt.savefig("obs_max.pdf", bbox_inches='tight')
    plt.clf()


def getDRVec(directory, obsids, distribution):
    drVec = np.zeros(len(obsids))

    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith(".tsv"):
            obsid = file.split('_')[0]
            if (obsid not in obsids):
                continue
            max = getMax(file)
            rms = getRMS(file)
            dr = max/rms

            drVec[obsids.index(obsid)] = dr

    if (distribution == 'cyclic'):
        pass
    elif (distribution == 'sorted'):
        plt.plot(obsids, drVec)

    plt.xticks(rotation=90)
    plt.ylabel("Dynamic Range (max/rms)")
    plt.savefig("obs_dynamic_range.pdf", bbox_inches='tight')
    plt.clf()

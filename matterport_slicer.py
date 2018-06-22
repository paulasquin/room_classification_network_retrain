# Script to detect, open and convert .ply files to image files, slices at regular altitude from the ply mesh.
# Written by Paul Asquin for Awabot Intelligence, June 2018
from __future__ import print_function
from plyfile import PlyData  # , PlyElement
import numpy as np
from PIL import Image
import io
import sys
from tools import *

lesAltitudes = [0.2, 0.4, 0.5, 0.7, 1]  # Altitudes around which the section will be taken
sectionWidth = 0.04  # A section is 4 cm high
imageFolder = "JPG"
plyFolder = "HOUSE_SEGMENTATION"
imageSize = 500
if imageSize != 500:
    imageFolder += "_" + str(imageSize)
lesLabel = [
    ["a", "bathroom"],
    ["b", "bedroom"],
    ["k", "kitchen"],
    ["l", "living"]]


def getSlicesLoop(lesPoints, sectionsDownUp):
    """ Open lesPoints, read the data and return slices points with extrema """
    # Initialize min-max values
    minX, minY, minZ, maxX, maxY, maxZ = 1, 1, 1, 0, 0, 0

    # Initialisation of "sliced" section points
    sliced = [np.array((0, 0, 0), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])] * len(sectionsDownUp)
    # Initialisation first point write
    init = [True] * len(sectionsDownUp)

    for i in range(len(lesPoints)):
        (x, y, z) = lesPoints[i]

        # Searching for map extrema
        if i == 0:
            # Initialisation of first values
            minX, maxX, minY, maxY, minZ, maxZ = x, x, y, y, z, z
        else:
            minX = min(minX, x)
            minY = min(minY, y)
            minZ = min(minZ, z)
            maxX = max(maxX, x)
            maxY = max(maxY, y)
            maxZ = max(maxZ, z)

        for j in range(len(sectionsDownUp)):
            downLimit = sectionsDownUp[j][0]
            upLimit = sectionsDownUp[j][1]
            if downLimit <= z <= upLimit:
                if init[j]:
                    init[j] = False
                    sliced[j] = np.array((x, y, z), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                else:
                    sliced[j] = np.append(sliced[j], np.array((x, y, z), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]))

    return sliced, [minX, maxX, minY, maxY, minZ, maxZ]


def getMetas(pathToHouse):
    global lesLabel
    print("Opening " + pathToHouse.split("/")[-1])
    lesMeta = []
    lesCorrespLabel = []
    with open(pathToHouse, 'r') as f:
        for line in f:
            ligneProcess = line.replace(" \n", "").split("  ")

            for label in lesLabel:
                if ligneProcess[0] == 'R' and ligneProcess[3] == label[0]:
                    print("\tFound a " + label[1] + " !")
                    lesMeta.append(ligneProcess)
                    lesCorrespLabel.append(label[1])
    return lesMeta, lesCorrespLabel


def extractRoomLoop(lesMeta, pathToPly):
    lesBoundsBox = []
    lesRoomPly = [np.array((0, 0, 0), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])] * len(lesMeta)
    for meta in lesMeta:
        # Get bounds values
        val_p = meta[4].split(" ")
        val_lo = meta[5].split(" ")
        val_hi = meta[6].split(" ")

        # Write value to BoundsBox object
        lesBoundsBox.append(
            BoundsBox(
                Point(float(val_p[0]), float(val_p[1]), float(val_p[2])),
                Point(float(val_lo[0]), float(val_lo[1]), float(val_lo[2])),
                Point(float(val_hi[0]), float(val_hi[1]), float(val_hi[2]))
            )
        )

    print("Opening " + pathToPly + " this may take a while...", end="")
    sys.stdout.flush()
    plydata = PlyData.read(pathToPly)
    print(" - Done !")
    lenVertex = plydata['vertex'].count

    # Initialize first point write in lesRoomPly
    init = [True] * len(lesBoundsBox)
    for i in range(lenVertex):
        if i % 10000 == 0:
            print(str(i) + "/" + str(lenVertex) + " vertex")
        (x, y, z, _, _, _, _, _, _, _, _) = plydata['vertex'][i]
        for j, bound in enumerate(lesBoundsBox):
            if bound.lo <= Point(float(x), float(y), float(z)) <= bound.hi:
                if init[j]:
                    init[j] = False
                    lesRoomPly[j] = np.array(
                        (x, y, z),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                    )
                else:
                    lesRoomPly[j] = np.append(
                        lesRoomPly[j],
                        np.array(
                            (x, y, z),
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                    )
    return lesRoomPly


def extractRoom(lesMeta, pathToPly):
    """ Extract points from PLY files with use of metadatas.
    Vectorized implementation """
    lesBoundsBox = []
    lesRoomPly = []
    for meta in lesMeta:
        # Get bounds values
        val_p = meta[4].split(" ")
        val_lo = meta[5].split(" ")
        val_hi = meta[6].split(" ")

        # Write value to BoundsBox object
        lesBoundsBox.append(
            BoundsBox(
                Point(float(val_p[0]), float(val_p[1]), float(val_p[2])),
                Point(float(val_lo[0]), float(val_lo[1]), float(val_lo[2])),
                Point(float(val_hi[0]), float(val_hi[1]), float(val_hi[2]))
            )
        )

    print("Opening " + pathToPly + " this may take a while...", end="")
    sys.stdout.flush()
    plydata = PlyData.read(pathToPly)
    print(" - Done !")
    print("Extracting rooms...")
    for k, boundsBox in enumerate(lesBoundsBox):
        print("\tRoom " + str(k + 1) + "/" + str(len(lesBoundsBox)), end="")
        sys.stdout.flush()
        lessLoX = plydata['vertex'][np.where(boundsBox.lo.x <= plydata['vertex']['x'])]
        lessHiX = lessLoX[np.where(lessLoX['x'] <= boundsBox.hi.x)]
        lessLoY = lessHiX[np.where(boundsBox.lo.y <= lessHiX['y'])]
        lessHiY = lessLoY[np.where(lessLoY['y'] <= boundsBox.hi.y)]
        lessLoZ = lessHiY[np.where(boundsBox.lo.z <= lessHiY['z'])]
        lessHiZ = lessLoZ[np.where(lessLoZ['z'] <= boundsBox.hi.z)]
        print(" - " + str(len(lessHiZ)) + " points")
        lesRoomPly.append(lessHiZ)
    return lesRoomPly


def generateImage(pathToPly, pathToHouse, lesAltitudes, sectionWidth):
    """ Treat information to use of given .ply file to generate images at given altitudes with given sectionWidth """
    global imageFolder
    try:
        with open(imageFolder + "/house_out.log", 'r') as f:
            if pathToHouse in f.read():
                print(" - already treated")
                return 0
            else:
                print("")
    except FileNotFoundError:
        print("No house_out.log file yet")
    lesMeta, lesCorrespLabel = getMetas(pathToHouse=pathToHouse)
    lesRoomPly = extractRoom(lesMeta=lesMeta, pathToPly=pathToPly)
    i = 0
    while i < len(lesRoomPly):
        try:
            lenRoom = len(lesRoomPly[i])
        except TypeError:
            lenRoom = 0
        if lenRoom == 0:
            print("Room " + str(i + 1) + "/" + str(len(lesRoomPly)) + " - " + lesCorrespLabel[i] + " is empty")
            lesRoomPly.pop(i)
            lesCorrespLabel.pop(i)
        else:
            i += 1
    sectionsDownUp = computeSections(lesAltitudes=lesAltitudes, sectionWidth=sectionWidth)
    for k, roomPly in enumerate(lesRoomPly):
        slices, extrema = getSlices(roomPly, sectionsDownUp=sectionsDownUp)
        for i in range(len(lesAltitudes)):
            # Try to export and detect error
            if exportSlice(
                    slice=slices[i],
                    extrema=extrema,
                    label=lesCorrespLabel[k],
                    path=filepathImage(
                        pathToPly=pathToPly,
                        label=lesCorrespLabel[k],
                        prefix=str(k) + "-" + lesCorrespLabel[k],
                        suffix=str(lesAltitudes[i]),
                        extension="jpg",
                        imgFolder=imageFolder),
                    width=imageSize,
                    height=imageSize) != 0:
                print("Error with export of " + str(lesCorrespLabel[k]) + " altitude " + str(lesAltitudes[i]))
                # log error
                with open(imageFolder + "/house_error.log", 'a') as f:
                    try:
                        if pathToHouse not in f.read():
                            f.write(pathToHouse + "\n")
                    except io.UnsupportedOperation:
                        f.write(pathToHouse + "\n")
                return -1  # Send back we have an error
    # Write to out.log
    with open(imageFolder + "/house_out.log", 'a') as f:
        f.write(pathToHouse + "\n")


def main():
    global lesAltitudes
    global sectionWidth

    # Create image directory if it doesn't exist yet
    createFolder(imageFolder)

    # Load files and sort the names
    lesPlyPath = locate_files(
        extension=".ply",
        dbName="plyfiles",
        path=relative_to_absolute_path(plyFolder))
    lesPlyPath.sort()
    lesHousePath = locate_files(
        extension=".house",
        dbName="housefiles",
        path=relative_to_absolute_path(plyFolder))
    lesHousePath.sort()

    for i in range(len(lesPlyPath)):
        print("\nHouse " + str(i + 1) + "/" + str(len(lesPlyPath)) + " : " + lesHousePath[i], end="")
        sys.stdout.flush()
        generateImage(
            pathToPly=lesPlyPath[i],
            pathToHouse=lesHousePath[i],
            lesAltitudes=lesAltitudes,
            sectionWidth=sectionWidth)
    return 0


if __name__ == "__main__":
    main()

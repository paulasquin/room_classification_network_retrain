# Script to detect, open and convert .ply files to image files with slices at given altitude from the ply mesh.
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018

from plyfile import PlyData
import sys
import numpy as np
import io
from tools import *

lesAltitudes = [0.4, 0.5, 0.7]  # Altitudes around which the section will be taken
sectionWidth = 0.04  # A section is 4 cm high
imageFolder = "JPG"
plyFolder = "Scannet_PLY"


def getSlicesLoop(filepath, sectionsDownUp):
    """ Open .ply file, read the data and return slices points """
    print("Opening " + filepath)
    plydata = PlyData.read(filepath)
    lenVertex = plydata['vertex'].count

    minX = 1
    minY = 1
    minZ = 1
    maxX = 0
    maxY = 0
    maxZ = 0

    # Initialisation of "sliced" section points
    sliced = [np.array((0, 0, 0), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])] * len(sectionsDownUp)
    # Initialisation first point write
    init = [True] * len(sectionsDownUp)

    for i in range(lenVertex):
        (x, y, z, r, g, b, a) = plydata['vertex'][i]

        # Searching for map extrema
        if i == 0:
            # Initialisation of first values
            minX = x
            maxX = x
            minY = y
            maxY = y
            minZ = z
            maxZ = z
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

def getLabels():
    """ Get label names by looking à .txt files in the main folder """
    parentPath = os.getcwd()
    # Go to subfolder labels
    os.chdir("labels")
    # Get file names
    brut = os.listdir()
    # Go back to parent folder
    os.chdir(parentPath)
    lesLabels = []
    for b in brut:
        if '.txt' in b:
            lesLabels.append(b.replace(".txt", ""))
    return lesLabels


def filepathImage(filepath, label, suffix="", folderName="", extension="png"):
    """ Generate the path name for the image, default folder name is label.title()
    A suffix can be indicated to indicate, for example, the altitude level """
    global imageFolder
    if folderName == "":
        folderName = label.title()
    if suffix != "":
        suffix = "-" + suffix
    sceneId = filepath.split("/")[-2]
    path = os.getcwd() + "/" + imageFolder + "/" + folderName + "/" + sceneId + suffix + "." + extension

    return path

def extractPoints(pathToPly):
    print("Opening " + pathToPly + " this may take a while...", end="")
    sys.stdout.flush()
    plydata = PlyData.read(pathToPly)
    print(" - Done !")
    return plydata['vertex']

def generateImage(filepath, lesAltitudes, sectionWidth, label):
    """ Treat information to use of given .ply file to generate image at given altitudes with given sectionWidth
    A label have to be indicated to precise if the file is a Kitchen, a bathroom, etc """
    global imageFolder

    try:
        with open(imageFolder + "/out.log", 'r') as f:
            if filepath in f.read():
                print(filepath + " already treated")
                return 0

    except FileNotFoundError:
        print("No out.log file yet")

    sectionsDownUp = computeSections(lesAltitudes=lesAltitudes, sectionWidth=sectionWidth)
    lesPoints = extractPoints(pathToPly=filepath)
    slices, extrema = getSlices(lesPoints, sectionsDownUp=sectionsDownUp)
    for i in range(len(lesAltitudes)):
        # Try to export and detect error
        if exportSlice(slice=slices[i], extrema=extrema, label=label,
                       path=filepathImage(filepath=filepath, label=label, suffix=str(lesAltitudes[i]),
                                          extension="jpg"), width=100, height=100) != 0:
            print("Error with export of " + str(label) + " altitude " + str(lesAltitudes[i]))
            # log error
            with open(imageFolder + "/error.log", 'a') as f:
                try:
                    if filepath not in f.read():
                        f.write(filepath + "\n")
                except io.UnsupportedOperation:
                    f.write(filepath + "\n")
            return 1

    # Write to out.log
    with open(imageFolder + "/out.log", 'a') as f:
        f.write(filepath + "\n")


def main():
    # Create image directory if it doesn't exist yet
    createFolder(imageFolder)
    for label in getLabels():
        paths = locate_files(extension="_vh_clean_2.ply", dbName=label + "_plyfiles",
                             path=relative_to_absolute_path(plyFolder + "/" + label.title()))
        print("\n--- " + label + " : " + str(len(paths)) + " ---")
        for i in range(len(paths)):
            print("\n" + str(i + 1) + "/" + str(len(paths)) + " " + str(label))
            generateImage(filepath=paths[i], lesAltitudes=lesAltitudes, sectionWidth=sectionWidth, label=label)
    return 0


if __name__ == "__main__":
    main()

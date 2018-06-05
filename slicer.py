# Script to detect, open and convert .ply files to image files, slices at regular altitude from the ply mesh.
# Written by Paul Asquin for Awabot Intelligence, May 2018

from plyfile import PlyData, PlyElement
import os
import subprocess
import numpy as np
from PIL import Image
import io

lesAltitudes = [0.2, 0.4, 0.5, 0.7, 1]  # Altitudes around which the section will be taken
sectionWidth = 0.04  # A section is 4 cm high
imageFolder = "JPG"


def computeSections(lesAltitudes, sectionWidth):
    """ From given altitudes and section width, generate the lower and upper z values of a section.
     The values are taken from alt - width/2 to alt + width/2 """

    sectionsDownUp = []
    for alt in lesAltitudes:
        sectionsDownUp.append([alt - sectionWidth / 2, alt + sectionWidth / 2])
    return sectionsDownUp


def relative_to_absolute_path(path=""):
    """ Send back absolute path if relative was given """
    # Check if not already absolute path
    if path == "":
        path = os.getcwd()
    elif path[0] != "/":
        path = os.getcwd() + "/" + path
    return path


def locate_files(extension, path=os.getcwd(), dbName="locate"):
    """ Locate files using .db database. May need sudo to write the database"""
    print("\n\nCreating the database \"" + dbName + ".db\" for the \"local\" command")
    print("Searching " + extension + " in " + path)
    try:
        cmd = "updatedb -l 0 -o " + dbName + ".db -U " + path + "/"
        print(cmd)
        subprocess.call(["echo 'You may need sudo access' &" + cmd], shell=True)
    except:
        print(
            "Might be an error in permission to write the locate database. Try to launch the script with \"sudo python\"")

    cmd = "locate -d " + dbName + ".db " + relative_to_absolute_path(path) + "/*" + extension
    print(cmd)
    p = subprocess.Popen([cmd],
                         stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    paths = out.decode('utf-8')
    paths = paths.split('\n')
    paths.pop()  # We delete the last, empty, element
    return paths


def getSlices(filepath, sectionsDownUp):
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


def createFolder(label):
    global imageFolder
    subPath = imageFolder + "/" + str(label.title())
    print("Creating folder " + str(subPath))
    os.mkdir(subPath)
    return 0


def exportSlice(slice, extrema, path, label, width=500, height=500):
    global imageFolder
    """ Write raw slice points to a image files. Resolution can be change, default is 500*500 """
    resize = slice.copy()
    minX, maxX, minY, maxY, minZ, maxZ = extrema
    # Put values to a [0, width-1]*[0, height-1] domain
    # Minus width to have a "bird like" view (mirror transformation)
    try:
        resize['x'] = (width - 1) - (slice['x'] - minX) * (width - 1) / (maxX - minX)
        resize['y'] = (slice['y'] - minY) * (height - 1) / (maxY - minY)
    except IndexError:
        print("Problem with resize for " + str(path)
              + "\nminX : " + str(minX) + " maxX : " + str(maxX) + " minY : " + str(minY) + " maxY : " + str(maxY)
              + "\nnp.minX : " + str(np.min(resize['x'])) + " np.maxX : " + str(np.max(resize['x'])) + "np.minY : " +
              str(np.min(resize['y'])) + " np.maxY : " + str(np.max(resize['y'])))
        return -1
        # Create a image from given coordinates :

    # Image initialization to white
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Affect coordinates of points to black
    try:
        image[(resize['y'].astype(int), resize['x'].astype(int))] = [0, 0, 0]
    except IndexError:
        print("Problem with resize for " + str(path)
              + "\nminX : " + str(minX) + " maxX : " + str(maxX) + " minY : " + str(minY) + " maxY : " + str(maxY)
              + "\nnp.minX : " + str(np.min(slice['x'])) + " np.maxX : " + str(np.max(slice['x'])) + "np.minY : " +
              str(np.min(slice['y'])) + " np.maxY : " + str(np.max(slice['y'])))
        return -1
    img = Image.fromarray(image, 'RGB')
    try:
        if imageFolder == "JPG":
            img.save(path, "JPEG", quality=80, optimize=True, progressive=True)
        else:
            img.save(path)
    except FileNotFoundError:
        createFolder(label)
    print(path + " exported")
    return 0


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


def generateImage(filepath, lesAltitudes, sectionWidth, label):
    """ Treat information to use of given .ply file to generate image at given altitudes with given sectionWidth
    A label have to be indicated to precise if the file is a Kitchen, a bathroom, etc """
    global imageFolder

    try:
        with open("out.log", 'r') as f:
            if filepath in f.read():
                print(filepath + " already treated")
                return 0

    except FileNotFoundError:
        print("No out.log file yet")

    sectionsDownUp = computeSections(lesAltitudes=lesAltitudes, sectionWidth=sectionWidth)
    slices, extrema = getSlices(filepath=filepath, sectionsDownUp=sectionsDownUp)
    for i in range(len(lesAltitudes)):
        # Try to export and detect error
        if exportSlice(slice=slices[i], extrema=extrema, label=label,
                       path=filepathImage(filepath=filepath, label=label, suffix=str(lesAltitudes[i]),
                                          extension=imageFolder.lower())) != 0:
            print("Error with export of " + str(label) + " altitude " + str(lesAltitudes[i]))
            # log error
            with open("error.log", 'a') as f:
                try:
                    if filepath not in f.read():
                        f.write(filepath + "\n")
                except io.UnsupportedOperation:
                    f.write(filepath + "\n")
            return 1

    # Write to out.log
    with open("out.log", 'a') as f:
        f.write(filepath + "\n")


def main():

    # Create image directory if it doesn't exist yet
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    for label in getLabels():
        paths = locate_files(extension="_vh_clean_2.ply", dbName=label + "_plyfiles",
                             path=relative_to_absolute_path(label.title()))
        print("\n--- " + label + " : " + str(len(paths)) + " ---")
        for i in range(len(paths)):
            print("\n" + str(i + 1) + "/" + str(len(paths)) + " " + str(label))
            generateImage(filepath=paths[i], lesAltitudes=lesAltitudes, sectionWidth=sectionWidth, label=label)
    return 0


if __name__ == "__main__":
    main()

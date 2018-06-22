# Script created by Paul Asquin for Awabot Intelligence, May 2018
# Automating scene downloading of scene id files

from tools import *

labelFolder = "labels/"
plyFolder = "PLY/"


def getFiles(extension=".txt"):
    """ Get files name with a given extension (default : .txt)
    in this directory """
    global labelFolder
    parentPath = os.getcwd()

    # Go to labels subfolder
    os.chdir(labelFolder)

    # Get file names
    brut = os.listdir()
    os.chdir(parentPath)

    ret = []
    for b in brut:
        if '.txt' in b:
            ret.append(b)
    return ret


def downloadScene(sceneId, folder):
    os.system("python scannet_download.py -o " + folder + "/ --id " + sceneId + " --type _vh_clean_2.ply")


def getTxtFilePath(file):
    """ Return txt file path considering labelFolder """
    global labelFolder
    path = file
    if labelFolder != "":
        path = labelFolder + "/" + path
    return path


def main():
    global plyFolder
    createFolder(plyFolder)
    files = getFiles()
    folders = []
    for file in files:
        print("--- " + str(file) + " ---")
        folders.append(file.replace(".txt", "").title())
        createFolder(folders[-1])
        with open(getTxtFilePath(file), 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                if "id" not in line:
                    print("Downloading " + str(line))
                    try:
                        downloadScene(sceneId=line, folder=plyFolder + folders[-1])
                    except KeyboardInterrupt:
                        return 1


if __name__ == '__main__':
    main()

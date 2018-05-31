# Script created by Paul Asquin for Awabot Intelligence, May 2018
# Automating scene downloading of scene id files

import os
import subprocess


def getFiles(extension=".txt"):
    """ Get files name with a given extension (default : .txt)
    in this directory """
    brut = os.listdir()
    ret = []
    for b in brut:
        if '.txt' in b:
            ret.append(b)
    return ret


def createFolder(folderName):
    """" Create a folder if it not already exists"""
    # If folder doesn't exist, we create it
    if os.path.isdir(folderName) == False:
        print("Creating " + str(folderName))
        os.mkdir(folderName)


def downloadScene(sceneId, folder):
    os.system("python download-scannet.py -o " + folder + "/ --id " + sceneId)
    out, err = p.communicate()
    return out.decode('utf-8')


def main():
    files = getFiles()
    folders = []
    for file in files:
        print("--- " + str(file) + " ---")
        folders.append(file.replace(".txt", "").title())
        createFolder(folders[-1])
        with open(file, 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                if "id" not in line:
                    print("Downloading " + str(line))
                    downloadScene(sceneId=line, folder=folders[-1])


if __name__ == '__main__':
    main()

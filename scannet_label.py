# Call the model create by dataset_inception_retrain.py for later use
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018

import subprocess
import argparse
import os

# Problem loading .pb graph, maybe not right type (changes in binary files)

tensorFolder = "tensorflow"


def getGraphPath():
    global tensorFolder
    files = os.listdir(tensorFolder)
    lesPathGraph = []
    for file in files:
        if ".pb" in file:
            lesPathGraph.append(file)
    if len(lesPathGraph) == 0:
        return ""
    return tensorFolder + "/" + lesPathGraph[-1]


def getLabelPath():
    return tensorFolder + "/scannet_labels.txt"


def runLabel(args):
    dir = os.getcwd() + "/"
    cmd = "python3 label_image.py" + \
          " --graph=" + dir + getGraphPath() + \
          " --labels=" + dir + getLabelPath() + \
          " --input_layer=Placeholder" + \
          " --output_layer=final_result" + \
          " --image=" + dir + args.path
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))


def main(args):
    runLabel(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label an image using last graph')
    parser.add_argument('path', metavar='path', type=str, help='Path to the image to labeled')
    args = parser.parse_args()
    main(args)

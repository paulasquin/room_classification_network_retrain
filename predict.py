#!/usr/bin/python3
# Use tensorflow_retrain.py in order to attempt to calssify a given map
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018


from tools import *
import os
import subprocess

TENSOR_FOLDER = "tensorflow"

saved_model.pb


def runPredict(graph_path, labels_path, image_path):
    """ Call tensorflow_predict.py script with a large panel of arguments """
    exportNumber = getExportNumber(TENSOR_FOLDER)
    exportPath = str(TENSOR_FOLDER) + "/export_" + str(exportNumber)
    os.mkdir(exportPath)

    print("exportPath : " + exportPath)
    createFolder(TENSOR_FOLDER)

    cmd = "python3 tensorflow_predict.py" \
          " --graph=" + graph_path + \
          " --labels " + labels_path + \
          " --input_layer=Placeholder" + \
          " --output_layer=final_result" + \
          " --image " + image_path
          
    print(cmd)
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))
    return 0

def main():
    les_pb_path = locate_files(extension=".pb", path=os.getcwd() + "/" + TENSOR_FOLDER, dbName="pb")
    print("Choose a model : ")
    for i, pb_path in enumerate(les_pb_path):
        print("\n\n" + str(i) + " : " + str(pb_path))
        cmd_txt_path = str('/'.join(pb_path.split("/")[:-2]) + "/cmd.txt")
        print(cmd_txt_path)
        try:
            with open(cmd_txt_path, 'r') as f:
                for line in f:
                    print("\t" + str(line.replace("\n", "")))
            print("")
        except FileNotFoundError:
            print("// No cmd.txt \n")
    model_num = int(input(">> "))

    try:
        pb_path = les_pb_path[model_num]
        model_dir_path = '/'.join(pb_path.split("/")[:-1]) + "/"
    except IndexError or TypeError:
        print("Wrong input")
        return -1
    runPredict(graph_path, labels_path, image_path)

if __name__ == "__main__":
    main()

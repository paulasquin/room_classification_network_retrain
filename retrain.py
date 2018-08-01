# use modified version of retrain.py to retrain Inception for Room Classification application using 2D slices generated
# from datasets like Scannet and Matterport
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018

from tools import *
import os
import subprocess

TENSOR_FOLDER = "tensorflow"
createFolder(TENSOR_FOLDER)
DATASET_PATH = "../Datasets/JPG"
TRAIN_STEP = 8000


def runRetrain():
    """ Call retrain.py script with a large panel of arguments """
    exportNumber = getExportNumber(TENSOR_FOLDER)
    exportPath = str(TENSOR_FOLDER) + "/export_" + str(exportNumber)
    os.mkdir(exportPath)

    print("exportPath : " + exportPath)
    createFolder(TENSOR_FOLDER)

    cmd = "python3 retrain.py" \
          " --image_dir " + DATASET_PATH + \
          " --saved_model_dir " + exportPath + "/model/" + \
          " --validation_batch_size -1" + \
          " --print_misclassified_test_images True" + \
          " --how_many_training_steps " + TRAIN_STEP + \
          " --path_mislabeled_names " + exportPath + \
          " --bottleneck_dir bottleneck/" + DATASET_PATH + \
          " --summaries_dir /tmp/retrain_logs/" + \
          " --train_maximum True"  + \
          " --validation_percentage 5" + \
          " --testing_percentage 5" #+ \
          #" --tfhub_module 'https://tfhub.dev/google/imagenet/pnasnet_large/classification/2'" # https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1'" #

    print(cmd)
    with open(exportPath + "/cmd.txt", 'w') as f:
        f.write(cmd + "\n")
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))
    return 0

def main():
    try:
        runRetrain()
    except KeyboardInterrupt:
        print("Wait for clean exit of the retrain script")
        pass
    return 0

if __name__ == "__main__":
    main()

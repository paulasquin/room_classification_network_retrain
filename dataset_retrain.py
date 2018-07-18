# use modified version of retrain.py to retrain Inception for Room Classification application using 2D slices generated
# from datasets like Scannet and Matterport
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018

from tools import *
import os
import subprocess

TENSOR_FOLDER = "tensorflow"
IMAGE_DIR = "JPG_Scannet_Matterport_Tri_Aug"
TENSORBOARD_PATH = "~/.local/lib/python3.5/site-packages/tensorboard/main.py"



def runRetrain():
    """ Call retrain.py script with a large panel of arguments """
    exportNumber = getExportNumber(TENSOR_FOLDER)
    exportPath = str(TENSOR_FOLDER) + "/export_" + str(exportNumber)
    os.mkdir(exportPath)

    print("exportPath : " + exportPath)
    createFolder(TENSOR_FOLDER)

    cmd = "python3 retrain.py" \
          " --image_dir " + IMAGE_DIR + \
          " --saved_model_dir " + exportPath + "/model/" + \
          " --validation_batch_size -1" + \
          " --print_misclassified_test_images True" + \
          " --how_many_training_steps 8000" + \
          " --path_mislabeled_names " + exportPath + \
          " --bottleneck_dir /media/nas/Tensorflow/bottleneck/" + IMAGE_DIR + \
          " --summaries_dir /tmp/retrain_logs/" + \
          " --train_maximum True"  + \
          " --validation_percentage 5" + \
          " --testing_percentage 5" + \
          " --tfhub_module 'https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1'" #https://tfhub.dev/google/imagenet/pnasnet_large/classification/2
    # " --learning_rate 0.05" #+ \
    # /tmp/retrain_logs/

    # " --flip_left_right" + \
    # " --output_graph " + tensorFolder + "/scannet_inception.db" + \
    # " --output_labels " + tensorFolder + "/scannet_labels.txt"
    print(cmd)
    with open(exportPath + "/cmd.txt", 'w') as f:
        f.write(cmd + "\n")
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))
    return 0


def runTensorboard():
    """ Run tensorboard for data monitoring """
    #subprocess.call(["killall tensorboard"], shell=True)
    subprocess.call(["sudo tensorboard --logdir /tmp/retrain_logs/"], shell=True)
    return 0

def main():
    try:
        runTensorboard()
        runRetrain()
    except KeyboardInterrupt:
        print("Wait for clean exit of the retrain script")
        pass
    # Wait for user to make the program stop
    input("Press a key to stop the program and stop tensorboard")
    return 0

if __name__ == "__main__":
    main()

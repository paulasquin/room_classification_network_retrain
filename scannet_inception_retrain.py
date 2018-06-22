import os
import threading
import subprocess
import time

def getExportNumber(tensorFolder):
    """ Get the number of the export folder looking at already existing folders
    Handle the presence of '_precisions' at the end of the folder name """

    lesDir = os.listdir(tensorFolder)
    lesExport = []
    lesNum = []
    num = 0
    for dir in lesDir:
        if "export_" in dir:
            lesExport.append(dir)
    for i in range(len(lesExport)):
        # Get number of export and add 1 to it
        # If we have an extension in the name
        if lesExport[i][7:].find("_") != -1:
            lesNum.append(int(lesExport[i][7:7 + lesExport[i][7:].find("_")]))
        # If there is not extension
        else:
            lesNum.append(int(lesExport[i][7:]))

    if len(lesNum) != 0:
        num = max(lesNum) + 1

    return num


def runRetrain():
    tensorFolder = "tensorflow"
    image_dir = "JPG_Scannet_Matterport"
    exportNumber = getExportNumber(tensorFolder)
    exportPath = str(tensorFolder) + "/export_" + str(exportNumber)
    os.mkdir(exportPath)

    print("exportPath : " + exportPath)
    if not os.path.isdir(tensorFolder):
        os.mkdir(tensorFolder)


    cmd = "python3 retrain.py" \
          " --image_dir " + image_dir + \
          " --saved_model_dir " + exportPath + "/model/" + \
          " --validation_batch_size -1" + \
          " --how_many_training_steps 4000" + \
          " --suffix -0.5" \
          " --learning_rate 0.01" + \
          " --validation_percentage 15" + \
          " --testing_percentage 0" + \
          " --path_mislabeled_names " + exportPath + \
          " --bottleneck_dir /media/nas/Tensorflow/bottleneck/" + image_dir + \
          " --summaries_dir /tmp/retrain_logs/" #\
          #" --tfhub_module 'https://tfhub.dev/google/imagenet/pnasnet_large/classification/1'"
        # " --flip_left_right" + \
    # " --output_graph " + tensorFolder + "/scannet_inception.db" + \
    # " --output_labels " + tensorFolder + "/scannet_labels.txt"
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))
    with open(exportPath + "/cmd.txt", 'w') as f:
        f.write(cmd+"\n")
    return 0


def runTensorboard():
    time.sleep(2)
    cmd = "tensorboard --logdir /tmp/retrain_logs/"
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p1.communicate()
    print(out.decode('utf-8'))


def main():
    try:
        # Kill precedent residual tensorboard server
        os.system("killall tensorboard")
        # Launch retrain thread
        threadR = threading.Thread(target=runRetrain)
        # threadR.daemon = True
        threadR.start()
        # Launch tensorboard thread
        threadT = threading.Thread(target=runTensorboard)
        # threadT.daemon = True
        threadT.start()
        # Wait for user to Make the program stop
        input("Press a key to stop the program and stop tensorboard")

        return 0


    except KeyboardInterrupt:
        threadR._stop()
        threadT._stop()
        return 0


if __name__ == "__main__":
    main()

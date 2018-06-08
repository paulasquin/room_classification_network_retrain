import os
import threading
import subprocess

echo = ""


def getExportNumber(tensorFolder):
    lesDir = os.listdir(tensorFolder)
    lesExport = []
    for dir in lesDir:
        if "export_" in dir:
            lesExport.append(dir)
    if len(lesExport) != 0:
        lesExport.sort()
        # Get number of export and add 1 to it
        num = 1 + int(lesExport[-1][7: 7 + lesExport[-1][7:].find("_")])

    return num


def runRetrain():
    tensorFolder = "tensorflow"
    image_dir = "JPG"
    exportNumber = getExportNumber(tensorFolder)
    exportPath = str(tensorFolder) + "/export_" + str(exportNumber)
    if not os.path.isdir(tensorFolder):
        os.mkdir(tensorFolder)
    cmd = "python3 retrain.py" \
          " --image_dir " + str(image_dir) + \
          " --output_graph " + str(tensorFolder) + "/scannet_inception.db" + \
          " --output_labels " + str(tensorFolder) + "/scannet_labels.txt" + \
          " --saved_model_dir " + exportPath + "/" + \
          " --print_misclassified_test_images" + \
          " --validation_batch_size=-1" #+ \
        # " --suffix -0.5" + \
    global echo
    echo = cmd
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))
    with open(tensorFolder + "/retrain_cmd.txt", 'a') as f:
        f.write(str(exportNumber) + " : " + str(cmd))
    with open(exportPath + "/cmd.txt", 'w') as f:
        f.write(cmd)
    return 0


def runTensorboard():
    cmd = "tensorboard --logdir /tmp/retrain_logs"
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
        # Wait for user to Make the programm stop
        input("Press a key to stop the program and stop tensorboard")

        print(echo)
        return 0


    except KeyboardInterrupt:
        threadR._stop()
        threadT._stop()
        return 0


if __name__ == "__main__":
    main()

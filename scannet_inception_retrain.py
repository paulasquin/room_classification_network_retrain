import os
import threading
import subprocess

echo = ""

def runRetrain():
    tensorFolder = "tensorflow"
    image_dir = "JPG"

    if not os.path.isdir(tensorFolder):
        os.mkdir(tensorFolder)
    cmd = "python3 retrain.py" \
          " --image_dir " + str(image_dir) + \
          " --output_graph " + str(tensorFolder) + "/scannet_inception.db" + \
          " --output_labels " + str(tensorFolder) + "/scannet_labels.txt" + \
          " --saved_model_dir " + str(tensorFolder) +"/export/" \
          " --validation_batch_size=-1" + \
          " --print_misclassified_test_images"
    global echo
    echo = cmd
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))


def runTensorboard():
    cmd = "tensorboard --logdir /tmp/retrain_logs"
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p1.communicate()
    print(out.decode('utf-8'))


def main():
    try:
        threadR = threading.Thread(target=runRetrain)
        threadR.daemon = True
        threadR.start()

        threadT = threading.Thread(target=runTensorboard)
        threadT.daemon = True
        threadT.start()
        input("")
        print(echo)


    except KeyboardInterrupt:
        threadR._stop()
        threadT._stop()


if __name__ == "__main__":
    main()

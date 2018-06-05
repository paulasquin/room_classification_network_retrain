import os
import threading
import subprocess


def runRetrain():
    tensorFolder = "tensorflow"
    image_dir = "JPG"

    if not os.path.isdir(tensorFolder):
        os.mkdir(tensorFolder)
    cmd = "python3 retrain.py " \
          "--image_dir " + str(image_dir) + \
          " --output_graph " + str(tensorFolder) + "/scannet_inception.db " \
                                                   "--output_labels " + str(tensorFolder) + "/scannet_labels.txt "
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out.decode('utf-8'))


def main():
    try:
        thread = threading.Thread(target=runRetrain)
        thread.daemon = True
        thread.start()
        os.system("tensorboard --logdir /tmp/retrain_logs")
        input("Press any key to continue")
    except KeyboardInterrupt:
        thread._stop()


if __name__ == "__main__":
    main()

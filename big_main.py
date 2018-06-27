import subprocess


class Script:
    cmd = ""
    description = ""

    def __init__(self, cmd, description):
        self.cmd = cmd
        self.description = description


LES_SCRIPTS = [
    Script(
        "python3 scannet_download_from_txts.py",
        "Downloading the ScanNet dataset from IDs text files in the 'Scannet_ID' folder"
    ),
    Script(
        "python3 scannet_slicer.py",
        "Slicing the ScanNet dataset"
    ),
    Script(
        "python3 download_mp.py -o HOUSE_SEGMENTATION --type house_segmentations",
        "Downloading the Matterport dataset"
    ),
    Script(
        "python3 matterport_slicer.py",
        "Slicing the Matterport dataset"
    ),
    Script(
        "python3 dataset_inception_retrain.py",
        "Retraining inception using previous transformed datasets. You have to enter Ctrl+C to stop the training"
    )
]


def main():
    print("Big Main program to download, treat and train models of the Map Room Classification project")
    input("Press any key to continue...")

    for script in LES_SCRIPTS:
        print("*" * 10)
        print(script.description)
        print(script.cmd)
        print("*" * 10)
        p = subprocess.Popen([script.cmd], stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        print(out.decode('utf-8'))
    return 0


if __name__ == "__main__":
    main()

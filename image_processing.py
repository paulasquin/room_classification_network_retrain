# Process images to augment the dataset or clean it of blank like image
# Written by Paul Asquin paul.asquin@gmail.com for Awabot Intelligence, 2018

from tools import *
import PIL

lesAugmentation = ['width-flip', 'height-flip', 'cwRotate', 'ccwRotate', 'inverse']
datasetFolder = "JPG_Scannet_Matterport"


def delBlankImage(lesImgPath):
    """ Check if given image paths are empty-like (less than 6ko), if yes, they are removed """
    print("Deleting empty-like image (<6ko)")
    for path in lesImgPath:
        if os.path.getsize(path) < 6000 or "-1.jpg" in path or "-0.2.jpg" in path:
            print("\tDel " + path.split("/")[-1], end="\r")
            os.remove(path)
            with open("empty_image.txt", "a") as f:
                f.write(path + "\n")
    return 0


def getAugmentationPath(imgPath, augmentation):
    return imgPath.replace(".jpg", "-" + augmentation + ".jpg")


def notAlreadyAugmented(imgPath, augmentation):
    return not os.path.isfile(getAugmentationPath(imgPath=imgPath, augmentation=augmentation))


def augmentImage(lesImgPath):
    global lesAugmentation
    print(', '.join(lesAugmentation))
    for imgPath in lesImgPath:
        with PIL.Image.open(imgPath) as img:
            print("Augmenting " + imgPath, end="\r")
            for augmentation in lesAugmentation:
                if augmentation == 'width-flip' and notAlreadyAugmented(imgPath=imgPath, augmentation=augmentation):
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT).save(
                        getAugmentationPath(
                            imgPath=imgPath,
                            augmentation=augmentation)
                    )
                elif augmentation == 'height-flip' and notAlreadyAugmented(imgPath=imgPath, augmentation=augmentation):
                    img.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(
                        getAugmentationPath(
                            imgPath=imgPath,
                            augmentation=augmentation)
                    )
                elif augmentation == 'cwRotate' and notAlreadyAugmented(imgPath=imgPath, augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_270).save(
                        getAugmentationPath(
                            imgPath=imgPath,
                            augmentation=augmentation)
                    )
                elif augmentation == 'ccwRotate' and notAlreadyAugmented(imgPath=imgPath, augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_90).save(
                        getAugmentationPath(
                            imgPath=imgPath,
                            augmentation=augmentation)
                    )
                elif augmentation == 'reverse' and notAlreadyAugmented(imgPath=imgPath, augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_180).save(
                        getAugmentationPath(
                            imgPath=imgPath,
                            augmentation=augmentation)
                    )


def main():
    while True:
        command = input("Enter \n"
                        "\t- 'rm' to del blank-like images from the dataset " + datasetFolder + "\n" + \
                        "\t- 'aug' to augment the dataset with " + ', '.join(lesAugmentation) + "\n" + \
                        "\t- 'e' to end the program\n>> ")
        if command == "rm":
            delBlankImage(locate_files(extension=".jpg", dbName="image", path=datasetFolder))
        elif command == "aug":
            augmentImage(locate_files(extension=".jpg", dbName="image", path=datasetFolder))
        elif command == "e":
            print("Exiting the program")
            return 0
        else:
            print("Command not understood")


if __name__ == "__main__":
    main()

from tools import *


def delBlankImage(lesImgPath):
    """ Check if given image paths are empty-like (less than 6ko), if yes, they are removed """
    print("Deleting empty-like image (<6ko)")
    for path in lesImgPath:
        if os.path.getsize(path) < 6000 or "-1.jpg" in path or "-0.2.jpg" in path:
            print("\tDel " + path.split("/")[-1])
            os.remove(path)
            with open("empty_image.txt", "a") as f:
                f.write(path + "\n")
    return 0

def main():
    delBlankImage(locate_files(extension=".jpg", dbName="image", path="JPG_Scannet_Matterport"))


if __name__ == "__main__":
    main()

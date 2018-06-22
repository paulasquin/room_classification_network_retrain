from tools import *


def delBlankImage(lesImgPath):
    """ Check if given image paths are empty-like (less than 3ko), if yes, they are removed """
    print("Deleting empty-like image (<3ko)")
    for path in lesImgPath:
        if os.path.getsize(path) < 3000:
            print("\tDel " + path.split("/")[-1])
            os.remove(path)
            with open("empty_image.txt", "a") as f:
                f.write(path + "\n")
    return 0

def main():
    delBlankImage(locate_files(extension=".jpg", dbName="image"))


if __name__ == "__main__":
    main()

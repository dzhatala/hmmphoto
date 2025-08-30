#ERROR2 use java instead
import sys # to access the system
fname="../data/bw_03.jpg"

# import cv2

# img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)

# while True:
    # cv2.imshow("Sheep", img)
    # cv2.waitKey(0)
    # sys.exit() # to exit from all the processes

# cv2.destroyAllWindows() # destroy all windows

# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg

# plt.title("Sheep Image")
# plt.xlabel("X pixel scaling")
# plt.ylabel("Y pixels scaling")

# image = mpimg.imread(fname)
# plt.imshow(image)
# plt.show()


from PIL import Image
img = Image.open(fname)
img.show()
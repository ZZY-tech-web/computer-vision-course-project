import cv2
import matplotlib.pyplot as plt

from common import bgr_to_rgb, read_bgr, save_figure, show_image


def main():
    bgr = read_bgr()
    rgb = bgr_to_rgb(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    show_image(axes[0, 0], rgb, "RGB")
    show_image(axes[0, 1], gray, "Grayscale", "gray")
    show_image(axes[0, 2], hsv[:, :, 0], "HSV: H (Hue)", "hsv")
    show_image(axes[0, 3], hsv[:, :, 1], "HSV: S (Saturation)", "gray")

    show_image(axes[1, 0], hsv[:, :, 2], "HSV: V (Brightness)", "gray")
    show_image(axes[1, 1], lab[:, :, 0], "Lab: L (Lightness)", "gray")
    show_image(axes[1, 2], lab[:, :, 1], "Lab: a (Green-Red)", "gray")
    show_image(axes[1, 3], lab[:, :, 2], "Lab: b (Blue-Yellow)", "gray")

    show_image(axes[2, 0], ycrcb[:, :, 0], "YCrCb: Y (Luma)", "gray")
    show_image(axes[2, 1], ycrcb[:, :, 1], "YCrCb: Cr", "gray")
    show_image(axes[2, 2], ycrcb[:, :, 2], "YCrCb: Cb", "gray")
    show_image(axes[2, 3], bgr, "Incorrect BGR display (comparison)")

    fig.suptitle("Task 3: Color Space Conversion Visualization", fontsize=16)
    fig.tight_layout()
    save_figure(fig, "task3_color_spaces.png")


if __name__ == "__main__":
    main()


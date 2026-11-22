import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import RESULT_DIR, bgr_to_rgb, ensure_dirs, read_bgr, save_figure, show_image


def rotate_bound(image, angle):
    """Rotate while enlarging the canvas so that image corners are not cropped."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_w = int(h * sin_value + w * cos_value)
    new_h = int(h * cos_value + w * sin_value)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def main():
    ensure_dirs()
    bgr = read_bgr()
    rgb = bgr_to_rgb(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)

    # Equalize only luminance to avoid changing the image hue dramatically.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    color_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    rotated = bgr_to_rgb(rotate_bound(bgr, 30))
    scaled_down = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    scaled_up = cv2.resize(rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(str(RESULT_DIR / "task1_gray_equalized.png"), gray_equalized)
    cv2.imwrite(
        str(RESULT_DIR / "task1_rotated_30deg.png"),
        cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(RESULT_DIR / "task1_scaled_1_5x.png"),
        cv2.cvtColor(scaled_up, cv2.COLOR_RGB2BGR),
    )

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    show_image(axes[0, 0], rgb, "Original")
    show_image(axes[0, 1], gray, "Grayscale", "gray")
    show_image(axes[0, 2], gray_equalized, "Histogram Equalization", "gray")
    show_image(axes[0, 3], color_equalized, "Color Equalization (Y channel)")
    show_image(axes[1, 0], rotated, "Rotate 30 degrees")
    show_image(axes[1, 1], scaled_down, "Scale 0.5x (INTER_AREA)")
    show_image(axes[1, 2], scaled_up, "Scale 1.5x (INTER_CUBIC)")
    axes[1, 3].axis("off")
    fig.suptitle("Task 1: Histogram Equalization, Rotation and Scaling", fontsize=16)
    fig.tight_layout()
    save_figure(fig, "task1_operations.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(gray.ravel(), bins=256, range=(0, 256), color="dimgray")
    axes[0].set_title("Histogram Before Equalization")
    axes[1].hist(gray_equalized.ravel(), bins=256, range=(0, 256), color="royalblue")
    axes[1].set_title("Histogram After Equalization")
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_xlabel("Gray level")
        ax.set_ylabel("Pixel count")
    fig.tight_layout()
    save_figure(fig, "task1_histogram_comparison.png")

    print(f"Original size: {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"Scaled down: {scaled_down.shape[1]} x {scaled_down.shape[0]}")
    print(f"Scaled up: {scaled_up.shape[1]} x {scaled_up.shape[0]}")


if __name__ == "__main__":
    main()


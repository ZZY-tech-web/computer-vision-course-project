import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import bgr_to_rgb, read_bgr, save_figure, show_image


def main():
    bgr = read_bgr()
    rgb = bgr_to_rgb(bgr)

    mean_blur = cv2.blur(rgb, (7, 7))
    gaussian_blur = cv2.GaussianBlur(rgb, (7, 7), sigmaX=1.5)
    median_blur = cv2.medianBlur(rgb, 7)
    bilateral_blur = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsharp masking: original + alpha * (original - Gaussian-blurred image).
    unsharp = cv2.addWeighted(rgb, 1.8, gaussian_blur, -0.8, 0)

    # A classic 3x3 sharpening convolution kernel.
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kernel_sharpen = cv2.filter2D(rgb, ddepth=-1, kernel=sharpen_kernel)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    show_image(axes[0, 0], rgb, "Original")
    show_image(axes[0, 1], mean_blur, "Mean Filter 7x7")
    show_image(axes[0, 2], gaussian_blur, "Gaussian Filter 7x7")
    show_image(axes[0, 3], median_blur, "Median Filter 7x7")
    show_image(axes[1, 0], bilateral_blur, "Bilateral Filter")
    show_image(axes[1, 1], unsharp, "Unsharp Masking")
    show_image(axes[1, 2], kernel_sharpen, "3x3 Sharpen Kernel")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    axes[1, 3].imshow(np.abs(laplacian), cmap="gray")
    axes[1, 3].set_title("Laplacian Edges")
    axes[1, 3].axis("off")

    fig.suptitle("Task 2: Image Smoothing and Sharpening", fontsize=16)
    fig.tight_layout()
    save_figure(fig, "task2_smoothing_sharpening.png")


if __name__ == "__main__":
    main()


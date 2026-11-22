import cv2
import numpy as np

from common import INPUT_IMAGE, ensure_dirs


def main():
    """Create a deterministic colorful image for all three experiments."""
    ensure_dirs()
    height, width = 480, 720
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

    blue = np.broadcast_to(45 + 150 * (1 - y), (height, width))
    green = np.broadcast_to(55 + 150 * x, (height, width))
    red = 45 + 150 * (0.55 * x[None, :] + 0.45 * y)
    image = np.dstack((blue, green, red)).clip(0, 255).astype(np.uint8)

    cv2.rectangle(image, (55, 70), (280, 250), (45, 185, 245), -1)
    cv2.circle(image, (455, 165), 105, (210, 75, 60), -1)
    pts = np.array([[515, 380], [630, 265], [690, 410]], np.int32)
    cv2.fillPoly(image, [pts], (75, 210, 95))
    cv2.line(image, (45, 365), (405, 300), (250, 245, 235), 12)
    cv2.putText(
        image,
        "Computer Vision",
        (70, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (25, 25, 25),
        3,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(INPUT_IMAGE), image)
    print(f"Saved: {INPUT_IMAGE}")


if __name__ == "__main__":
    main()


from pathlib import Path

import cv2
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_IMAGE = ROOT / "input" / "sample_image.png"
RESULT_DIR = ROOT / "results"


def ensure_dirs():
    (ROOT / "input").mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def read_bgr(path=INPUT_IMAGE):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_figure(fig, name):
    ensure_dirs()
    output = RESULT_DIR / name
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def show_image(ax, image, title, cmap=None):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate test data for onnx2tf calibration.
This script is meant to be run from the terminal for dev
purposes.

The images are downloaded using the script `downloader.py` as
instructed on this page:
https://storage.googleapis.com/openimages/web/download_v7.html.
"""
import argparse
from pathlib import Path

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    def preprocess_img(img: Image.Image) -> np.ndarray:
        """
        Preprocess the image to retain shape for calibration.

        Arguments
        ---------
        img: Image.Image
            The PIL image.

        Returns
        -------
        np.ndarray
            The numpy array representation of the image.
        """
        img = img.convert("RGB")
        img = img.crop((0, 0, 128, 128))
        return np.array(img).astype(np.float32).reshape(1, 128, 128, 3) / 255.0

    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument(
        "--dir", "-d", required=True, help="Directory containing images"
    )

    parser.add_argument(
        "--count",
        "-n",
        required=True,
        type=int,
        help="Number of images to use",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output filename (e.g., result.jpg or out.zip)",
    )

    args = parser.parse_args()

    COUNT = args.count
    images_dir = Path(args.dir)

    n_images = 0
    images = []
    for img_path in images_dir.iterdir():
        if n_images >= COUNT:
            break
        with Image.open(img_path) as img:
            images.append(img.copy())
            n_images += 1

    selected_imgs = list(map(preprocess_img, images))
    selected_imgs = np.concatenate(selected_imgs, axis=0)
    np.save(
        f"resources/{args.output}",
        selected_imgs,
        allow_pickle=False,
    )

    print(f"Saved to resources/{args.output}")

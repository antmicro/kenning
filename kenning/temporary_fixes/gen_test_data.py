# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate test data for onnx2tf calibration.
This script is meant to be run from the terminal for dev
purposes.
"""


if __name__ == "__main__":
    import numpy as np
    import PIL
    from datasets import load_dataset

    def preprocess_img(img: PIL.Image.Image) -> np.ndarray:
        """
        Preprocess the image to retain shape for calibration.

        Arguments
        ---------
        img: PIL.Image.Image
            The PIL image.

        Returns
        -------
        np.ndarray
            The numpy array representation of the image.
        """
        img = img.crop((0, 0, 128, 128))
        return np.array(img).reshape(1, 128, 128, 3)

    COUNT = 20
    ds = load_dataset("timm/mini-imagenet")
    train_ds = ds["train"]
    perms = np.random.permutation(range(len(train_ds)))[0:COUNT].tolist()
    selected_imgs = [preprocess_img(img) for img in train_ds[perms]["image"]]
    selected_imgs = np.concatenate(selected_imgs, axis=0)
    np.save(
        "resources/sample-calibration-20x128x128x3.npy",
        selected_imgs,
        allow_pickle=False,
    )

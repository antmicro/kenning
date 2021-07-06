"""

This program was used to test and debug the process of
downloading and preparation of the data from
OpenImagesV6 dataset for instance segmentation problem

"""

from kenning.datasets.open_images_dataset import OpenImagesDatasetV6
import numpy as np
import cv2

ds = OpenImagesDatasetV6(
                        "./data",
                        1,
                        True,
                        "instance_segmentation",
                        'coco',
                        3,
                        'validation',
                        "NHWC")

result_input = ds.prepare_input_samples(ds.dataX)
result_output = ds.prepare_output_samples(ds.dataY)

iteration = 0
for i, batch_j in zip(result_input, result_output):
    for j in batch_j:
        int_i = np.multiply(i, 255).astype('uint8')
        out_array = cv2.bitwise_and(int_i, int_i, mask=j.mask)
        out_array = cv2.cvtColor(out_array, cv2.COLOR_BGR2RGB)

        out_path = ds.root / 'demo-img'
        out_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path / str(iteration))+".jpg", out_array)
        iteration += 1

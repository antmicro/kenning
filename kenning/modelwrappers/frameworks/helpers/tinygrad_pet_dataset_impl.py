"""
Tinygrad ResNet training script for the Oxford-IIIT Pet Dataset.

Includes training, evaluation, data augmentation, ResNet backbone,
and a classifier head with optional transfer learning.
"""

import os
import random

import numpy as np
from PIL import Image
import torch

import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import CI, fetch, get_child, trange
from tinygrad.nn import BatchNorm2d as BatchNorm
from tinygrad.nn import Conv2d, Linear, optim
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    safe_save,
    torch_load,
)


def make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)

def train(
    model,
    X_train,
    Y_train,
    optim,
    steps,
    BS=128,
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    transform=lambda x: x,
    target_transform=lambda x: x,
    noloss=False,
    allow_jit=True,
):
    """Train a model for a fixed number of steps.

    Args:
        model: Tinygrad model.
        X_train: Training samples.
        Y_train: Training labels.
        optim: Optimizer.
        steps: Number of training steps.
        BS: Batch size.
        lossfn: Loss function.
        transform: Input transform.
        target_transform: Target transform.
        noloss: Disable loss tracking.
        allow_jit: Enable TinyJit compilation.

    Returns
        Lists of losses and accuracies.
    """

    def train_step(x, y):
        out = model.forward(x) if hasattr(model, "forward") else model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        if noloss:
            del loss
        optim.step()
        if noloss:
            return None, None
        cat = out.argmax(axis=-1)
        accuracy = (cat == y).mean()
        return loss.realize(), accuracy.realize()

    if allow_jit:
        train_step = TinyJit(train_step)

    with Tensor.train():
        losses, accuracies = [], []
        for _ in (t := trange(steps, disable=CI)):
            samp = np.random.randint(0, X_train.shape[0], size=(BS,))
            x = Tensor(transform(X_train[samp]), requires_grad=False)
            y = Tensor(target_transform(Y_train[samp]))
            loss, accuracy = train_step(x, y)
            if not noloss:
                loss, accuracy = loss.numpy(), accuracy.numpy()
                losses.append(loss)
                accuracies.append(accuracy)
                t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    return [losses, accuracies]


def evaluate(
    model,
    X_test,
    Y_test,
    num_classes=None,
    BS=128,
    return_predict=False,
    transform=lambda x: x,
    target_transform=lambda y: y,
):
    """Evaluate a trained model on a test dataset."""
    Tensor.training = False

    def numpy_eval(Y_test, num_classes):
        preds_out = np.zeros(list(Y_test.shape) + [num_classes])
        for i in trange(
            (len(Y_test) - 1) // BS + 1,
            disable=CI,
        ):
            x = Tensor(transform(X_test[i * BS : (i + 1) * BS]))
            out = model.forward(x) if hasattr(model, "forward") else model(x)
            preds_out[i * BS : (i + 1) * BS] = out.numpy()
        preds = np.argmax(preds_out, axis=-1)
        Y_test_t = target_transform(Y_test)
        return (Y_test_t == preds).mean(), preds

    if num_classes is None:
        num_classes = Y_test.max().astype(int) + 1

    acc, preds = numpy_eval(Y_test, num_classes)
    print("test set accuracy is %f" % acc)
    return (acc, preds) if return_predict else acc


class InvertedResidual:
    def __init__(self, inp, oup, stride, expand_ratio):
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(BatchNorm2d(hidden_dim))
            layers.append(Tensor.relu6)

        layers.extend(
            [
                Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                BatchNorm2d(hidden_dim),
                Tensor.relu6,
                Conv2d(
                    hidden_dim,
                    oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                BatchNorm2d(oup),
            ]
        )
        self.conv = layers

    def __call__(self, x):
        res = x
        for layer in self.conv:
            x = layer(x) if callable(layer) else x
        return res + x if self.use_res_connect else x


class TinyPetModel:
    def __init__(self, num_classes=37):
        self.stats = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        width_mult = 1.0
        round_nearest = 8

        input_channel = make_divisible(32 * width_mult, round_nearest)
        last_channel = make_divisible(
            1280 * max(1.0, width_mult), round_nearest
        )

        self.features = [
            Conv2d(
                3,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(input_channel),
            Tensor.relu6,
        ]

        for t, c, n, s in self.stats:
            output_channel = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel

        self.features.extend(
            [
                Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
                BatchNorm2d(last_channel),
                Tensor.relu6,
            ]
        )

        self.classifier = Linear(last_channel, num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = x.mean(axis=(2, 3), keepdim=True)
        x = x.reshape(x.shape[0], -1)

        return self.classifier(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def _initialize_weights(self):
        for p in get_parameters(self):
            if len(p.shape) > 1:
                # fan_in calculation
                if len(p.shape) == 2:
                    fan_in = p.shape[1]
                else:
                    fan_in = np.prod(p.shape[1:])

                std = float(np.sqrt(2.0 / fan_in))

                p.assign(Tensor.randn(*p.shape) * std)

    def load_from_local_pth(self, path="mobilenet_v2.pth"):
        print(f"Loading weights from local file: {path}")

        for k, v in torch_load(path).items():
            try:
                obj: Tensor = get_child(self, k)
                obj.assign(v.to(obj.device).reshape(obj.shape))
            except (AttributeError, IndexError):
                continue

    def load_from_pretrained(self):
        url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
        data = fetch(url)
        pt_state = torch.load(data, map_location="cpu")

        if "state_dict" in pt_state:
            pt_state = pt_state["state_dict"]

        tg_params = get_state_dict(self)
        for k, _ in tg_params.items():
            print(k)
        loaded, skipped = 0, 0

        for k, v in pt_state.items():
            if not isinstance(v, torch.Tensor):
                continue

            tg_key = k

            if tg_key.startswith("features."):
                parts = tg_key.split(".")

                if len(parts) > 2 and parts[2] == "conv":
                    feature_idx = int(parts[1])
                    conv_layer_idx = int(parts[3])
                    
                    if len(parts) > 3 and parts[4] == "0":
                        tg_key = tg_key.replace(f"features.{feature_idx}.conv.{conv_layer_idx}.0", 
                                                f"features.{feature_idx}.conv.{conv_layer_idx}")
                    print("fixed!")

            if tg_key.startswith("classifier"):
                continue

            if "num_batches_tracked" in tg_key:
                skipped += 1
                continue

            if "running_mean" in tg_key or "running_var" in tg_key:
                skipped += 1
                continue

            if tg_key not in tg_params:
                skipped += 1
                print(f"Skipped: {tg_key}")
                continue

            tg_tensor = tg_params[tg_key]

            if tg_tensor.shape != tuple(v.shape):
                skipped += 1
                print(f"Shape mismatch: {tg_key}")
                continue

            tg_tensor.assign(Tensor(v.numpy()).to(tg_tensor.device))
            loaded += 1
            print(f"Loaded: {tg_key}")

        print(f"[pretrained] loaded {loaded}, skipped {skipped}")


def fetch_pet_dataset(img_dir, anno_path):
    """Load image paths and labels from the Pet Dataset."""
    images, labels = [], []
    with open(anno_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, class_id = line.split()[:2]
            img_path = os.path.join(img_dir, f"{name}.jpg")
            if os.path.exists(img_path):
                images.append(img_path)
                labels.append(int(class_id) - 1)
    return np.array(images), np.array(labels)

class ComposeTransforms:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x

def random_resized_crop(img, size=224, scale=(0.9, 1.0)):
    w, h = img.size
    s = random.uniform(*scale)
    new_w, new_h = int(w * s), int(h * s)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    return img.resize((size, size))


def random_hflip(img, p=0.5):
    """Randomly horizontally flip an image."""
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def color_jitter(img, b=0.05, c=0.05):
    img = np.asarray(img).astype(np.float32)
    img = img * (1 + random.uniform(-b, b))      # brightness
    mean = img.mean(axis=(0,1), keepdims=True)
    img = (img - mean) * (1 + random.uniform(-c, c)) + mean  # contrast
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    Tensor.default_device = "CUDA"
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    mean = mean.reshape(1, 1, 1, 3)
    std = std.reshape(1, 1, 1, 3)
    transform = ComposeTransforms(
        [
            lambda x: [
                color_jitter(
                    random_hflip(
                        random_resized_crop(
                            Image.open(xx).convert("RGB").resize((256, 256)),
                            size=224
                        )
                    )
                )
                for xx in x
            ],
            lambda x: np.stack([np.asarray(xx) for xx in x], 0),
            lambda x: x / 255.0,
            lambda x: (x - mean) / std,
            lambda x: x.transpose(0, 3, 1, 2).astype(np.float32),
        ]
    )

    test_transform = ComposeTransforms(
        [
            lambda x: [
                Image.open(xx).convert("RGB").resize((224, 224)) for xx in x
            ],
            lambda x: np.stack([np.asarray(xx) for xx in x], 0),
            lambda x: x / 255.0,
            lambda x: (x - mean) / std,
            lambda x: x.transpose(0, 3, 1, 2).astype(np.float32),
        ]
    )

    img_dir = "build/PetDataset/images"
    X_train, Y_train = fetch_pet_dataset(
        img_dir,
        "build/PetDataset/annotations/trainval.txt",
    )
    X_test, Y_test = fetch_pet_dataset(
        img_dir,
        "build/PetDataset/annotations/test.txt",
    )

    classes = 37
    model = TinyPetModel(num_classes=classes)

    model._initialize_weights()
    model.load_from_pretrained()

    lr = 0.001
    optimizer = optim.Adam(get_parameters(model), lr=lr)
    TARGET_ACC = 0.95
    MAX_EPOCHS = 50

    epoch = 0
    train_acc = 0.0

    try:
        while train_acc < TARGET_ACC and epoch < MAX_EPOCHS:
            losses, accs = train(
                model,
                X_train,
                Y_train,
                optimizer,
                steps=100,
                transform=transform
            )

            train_acc = float(np.mean(accs[-10:]))
            print(f"Epoch {epoch}: train acc = {train_acc:.3f}")

            test_acc = evaluate(
                model,
                X_test,
                Y_test,
                num_classes=classes,
                transform=test_transform
            )

            optimizer.lr /= 1.4
            epoch += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    finally:
        state_dict = get_state_dict(model)
        safe_save(state_dict, "model.safetensors")
        print("Saved to model.safetensors")


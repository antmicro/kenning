"""
Tinygrad ResNet training script for the Oxford-IIIT Pet Dataset.

Includes training, evaluation, data augmentation, ResNet backbone,
and a classifier head with optional transfer learning.
"""

import os
import random

import numpy as np
from PIL import Image
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


class ComposeTransforms:
    """Compose multiple transforms into a single callable."""

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x


class BasicBlock:
    """ResNet basic residual block."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64):
        assert (
            groups == 1 and base_width == 64
        ), "BasicBlock only supports groups=1 and base_width=64"

        self.conv1 = Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.downsample = []

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = [
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(self.expansion * planes),
            ]

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out = out + x.sequential(self.downsample)
        return out.relu()


class Bottleneck:
    """ResNet bottleneck block."""

    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        stride_in_1x1=False,
        groups=1,
        base_width=64,
    ):
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = Conv2d(
            in_planes,
            width,
            kernel_size=1,
            stride=stride if stride_in_1x1 else 1,
            bias=False,
        )
        self.bn1 = BatchNorm(width)
        self.conv2 = Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            stride=1 if stride_in_1x1 else stride,
            groups=groups,
            bias=False,
        )
        self.bn2 = BatchNorm(width)
        self.conv3 = Conv2d(
            width,
            self.expansion * planes,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = BatchNorm(self.expansion * planes)

        self.downsample = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = [
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(self.expansion * planes),
            ]

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out)).relu()
        out = self.bn3(self.conv3(out))
        out = out + x.sequential(self.downsample)
        return out.relu()


class ResNet:
    """ResNet backbone implementation."""

    def __init__(
        self,
        num,
        num_classes=None,
        groups=1,
        width_per_group=64,
        stride_in_1x1=False,
    ):
        self.num = num
        self.block = {
            18: BasicBlock,
            34: BasicBlock,
            50: Bottleneck,
            101: Bottleneck,
            152: Bottleneck,
        }[num]

        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[num]

        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = BatchNorm(64)

        self.layer1 = self._make_layer(
            self.block,
            64,
            self.num_blocks[0],
            stride=1,
            stride_in_1x1=stride_in_1x1,
        )
        self.layer2 = self._make_layer(
            self.block,
            128,
            self.num_blocks[1],
            stride=2,
            stride_in_1x1=stride_in_1x1,
        )
        self.layer3 = self._make_layer(
            self.block,
            256,
            self.num_blocks[2],
            stride=2,
            stride_in_1x1=stride_in_1x1,
        )
        self.layer4 = self._make_layer(
            self.block,
            512,
            self.num_blocks[3],
            stride=2,
            stride_in_1x1=stride_in_1x1,
        )

        self.fc = (
            Linear(512 * self.block.expansion, num_classes)
            if num_classes is not None
            else None
        )

    def _make_layer(self, block, planes, num_blocks, stride, stride_in_1x1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if block == Bottleneck:
                layers.append(
                    block(
                        self.in_planes,
                        planes,
                        s,
                        stride_in_1x1,
                        self.groups,
                        self.base_width,
                    )
                )
            else:
                layers.append(
                    block(
                        self.in_planes,
                        planes,
                        s,
                        self.groups,
                        self.base_width,
                    )
                )
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        feature_only = self.fc is None
        features = [] if feature_only else None

        out = self.bn1(self.conv1(x)).relu()
        out = out.pad([1, 1, 1, 1]).max_pool2d((3, 3), 2)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = out.sequential(layer)
            if feature_only:
                features.append(out)

        if feature_only:
            return features

        out = out.mean([2, 3])
        return self.fc(out.cast(dtypes.float32))

    def __call__(self, x):
        return self.forward(x)

    def load_from_pretrained(self):
        """Load pretrained PyTorch ResNet weights."""
        model_urls = {
            (18, 1, 64): (
                "https://download.pytorch.org/models/" "resnet18-5c106cde.pth"
            ),
            (34, 1, 64): (
                "https://download.pytorch.org/models/" "resnet34-333f7ec4.pth"
            ),
            (50, 1, 64): (
                "https://download.pytorch.org/models/" "resnet50-19c8e357.pth"
            ),
            (50, 32, 4): (
                "https://download.pytorch.org/models/"
                "resnext50_32x4d-7cdf4587.pth"
            ),
            (101, 1, 64): (
                "https://download.pytorch.org/models/" "resnet101-5d3b4d8f.pth"
            ),
            (152, 1, 64): (
                "https://download.pytorch.org/models/" "resnet152-b121ed2d.pth"
            ),
        }

        self.url = model_urls[(self.num, self.groups, self.base_width)]
        for k, dat in torch_load(fetch(self.url)).items():
            try:
                obj = get_child(self, k)
            except AttributeError:
                if "fc." in k and self.fc is None:
                    continue
                raise

            if "fc." in k and obj.shape != dat.shape:
                continue

            if "bn" not in k and "downsample" not in k:
                assert obj.shape == dat.shape

            obj.assign(dat.to(obj.device).reshape(obj.shape))


class GlobalAvgPool2d:
    """Global average pooling."""

    def forward(self, x):
        x = x.mean(axis=(2, 3), keepdim=True)
        return x.reshape(x.shape[0], -1)

    def __call__(self, x):
        return self.forward(x)


class ResNetWithClassifier(ResNet):
    """ResNet backbone with a multi-layer classifier head."""

    def __init__(
        self,
        num=50,
        num_classes=37,
        groups=1,
        width_per_group=64,
        stride_in_1x1=False,
    ):
        super().__init__(
            num,
            num_classes=None,
            groups=groups,
            width_per_group=width_per_group,
            stride_in_1x1=stride_in_1x1,
        )

        self.avgpool = GlobalAvgPool2d()
        self.finallayer1 = Linear(512 * self.block.expansion, 1024)
        self.finallayer2 = Linear(1024, 512)
        self.finallayer3 = Linear(512, 128)
        self.finaloutput = Linear(128, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.avgpool(x[-1])
        x = self.finallayer1(x)
        x = x.relu()
        x = self.finallayer2(x)
        x = x.relu()
        x = self.finallayer3(x)
        x = x.relu()
        x = self.finaloutput(x)  # Shape becomes (batch_size, num_classes)

        return x

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


def random_hflip(img, p=0.5):
    """Randomly horizontally flip an image."""
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def color_jitter(img, b=0.05, c=0.05):
    """Apply brightness and contrast jitter."""
    img = np.asarray(img).astype(np.float32)
    img *= 1 + random.uniform(-b, b)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * (1 + random.uniform(-c, c)) + mean
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
                        Image.open(xx).convert("RGB").resize((224, 224)),
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

    model = ResNetWithClassifier(num=50, num_classes=37)
    model.load_from_pretrained()
    from tinygrad import Context, GlobalCounters, TinyJit
    optimizer = optim.Adam(get_parameters(model), lr=0.0001)
    for _ in range(50):
        losses, accs = train(
            model,
            X_train,
            Y_train,
            optimizer,
            steps=100,
            transform=transform,
        )

        train_acc = float(np.mean(accs[-10:]))
        print(f"Epoch {epoch}: train acc = {train_acc:.3f}")

        test_acc = evaluate(
            model,
            X_test,
            Y_test,
            num_classes=classes,
            transform=test_transform,
        )
        print(f"Epoch {epoch}: test acc = {test_acc:.3f}")

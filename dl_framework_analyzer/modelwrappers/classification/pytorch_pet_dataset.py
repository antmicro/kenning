"""
Contains PyTorch model for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset
"""

from pathlib import Path
from torchvision import models, transforms
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm

from dl_framework_analyzer.modelwrappers.frameworks.pytorch import PyTorchWrapper  # noqa: E501


class PyTorchPetDatasetMobileNetV2(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.numclasses = dataset.numclasses
        super().__init__(modelpath, dataset, from_file)

    def prepare_model(self):
        if self.from_file:
            self.load_model(self.modelpath)
        else:
            self.model = models.mobilenet_v2(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(1280, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.numclasses)
            )

    def train_model(
            self,
            batch_size: int,
            learning_rate: int,
            epochs: int,
            logdir: Path):
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(
            0.25
        )

        class PetDatasetPytorch(Dataset):
            def __init__(self, inputs, labels, dataset, transform=None):
                self.inputs = inputs
                self.labels = labels
                self.transform = transform
                self.dataset = dataset

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                X = self.dataset.prepare_input_samples([self.inputs[idx]])[0]
                y = self.dataset.prepare_output_samples([self.labels[idx]])[0]
                X = torch.from_numpy(X).permute(2, 0, 1)
                y = torch.from_numpy(y)
                if self.transform:
                    X = self.transform(X)
                return (X, y)

        mean, std = self.dataset.get_input_mean_std()

        traindat = PetDatasetPytorch(
            Xt, Yt, self.dataset,
            transform=transforms.Compose([
                transforms.ColorJitter(0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean, std)
            ])
        )

        validdat = PetDatasetPytorch(
            Xv, Yv, self.dataset,
            transform=transforms.Compose([
                transforms.ColorJitter(0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean, std)
            ])
        )

        trainloader = torch.utils.data.DataLoader(
            traindat,
            batch_size=batch_size,
            num_workers=0
        )

        validloader = torch.utils.data.DataLoader(
            validdat,
            batch_size=batch_size,
            num_workers=0
        )

        self.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            bar = tqdm(trainloader)
            for i, (images, labels) in enumerate(bar):
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, torch.argmax(labels, axis=1))

                opt.zero_grad()
                loss.backward()
                opt.step()

                bar.set_description(f'train epoch: {epoch:3} loss: {loss.data.cpu().numpy():.4f}')  # noqa: E501

            self.model.eval()
            with torch.no_grad():
                bar = tqdm(validloader)
                total = 0
                losssum = 0
                correct = 0
                losscount = 0
                for (images, labels) in bar:
                    images = images.float().to(self.device)
                    labelsgpu = labels.float().to(self.device)

                    outputs = self.model(images)
                    total += labels.size(0)
                    correct += np.equal(np.argmax(outputs.cpu().numpy(), axis=1), np.argmax(labels, axis=1)).sum()  # noqa: E501
                    loss = criterion(outputs, torch.argmax(labelsgpu, axis=1))
                    losssum += loss.data.cpu().numpy()
                    losscount += 1
                    bar.set_description(f'valid epoch: {epoch:3} loss: {loss.data.cpu().numpy():.4f}')  # noqa: E501

                print(f'corr: {correct} tot: {total}')
                acc = 100 * correct / total
                avgloss = losssum / losscount

                saved = False
                if acc > best_acc:
                    torch.save(self.model, self.modelpath)
                    best_acc = acc

                print(f'ep: {epoch:3} acc: {acc:5.2f}% avgloss: {avgloss}{", model saved" if saved else ""}')  # noqa: E501

        self.model.eval()

    def save_to_onnx(self, modelpath):
        x = torch.randn(self.batch_size, 224, 224, 3, requires_grad=True)
        torch.onnx.export(
            self.model,
            x,
            modelpath,
            export_params=True
        )

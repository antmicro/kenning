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
from torch.utils.tensorboard import SummaryWriter

from dl_framework_analyzer.modelwrappers.frameworks.pytorch import PyTorchWrapper  # noqa: E501


class PyTorchPetDatasetMobileNetV2(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.numclasses = dataset.numclasses
        super().__init__(modelpath, dataset, from_file)

    def get_input_spec(self):
        return {'input': (1, 3, 224, 224)}, 'float32'

    def preprocess_input(self, X):
        return torch.Tensor(
            np.array(X, dtype=np.float32)
        ).to(self.device).permute(0, 3, 1, 2)

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
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
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
            def __init__(self, inputs, labels, dataset, dev, transform=None):
                self.inputs = inputs
                self.labels = labels
                self.transform = transform
                self.dataset = dataset
                self.device = dev

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                X = self.dataset.prepare_input_samples([self.inputs[idx]])[0]
                y = np.array(self.labels[idx])
                X = torch.from_numpy(X).permute(2, 0, 1)
                y = torch.from_numpy(y)
                if self.transform:
                    X = self.transform(X)
                return (X.float().to(self.device), y.to(self.device))

        traindat = PetDatasetPytorch(
            Xt, Yt, self.dataset, self.device,
            transform=transforms.Compose([
                transforms.ColorJitter(0.1, 0.3),
                transforms.RandomHorizontalFlip(),
            ])
        )

        validdat = PetDatasetPytorch(
            Xv, Yv, self.dataset, self.device,
            transform=transforms.Compose([
                transforms.ColorJitter(0.1, 0.3),
                transforms.RandomHorizontalFlip()
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

        writer = SummaryWriter(log_dir=logdir)

        for epoch in range(epochs):
            self.model.train()
            bar = tqdm(trainloader)
            losssum = 0
            losscount = 0
            for i, (images, labels) in enumerate(bar):
                opt.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                opt.step()

                lossval = loss.data.cpu().numpy()
                losssum += lossval
                losscount += 1
                bar.set_description(f'train epoch: {epoch:3} loss: {lossval:.4f}')  # noqa: E501
            writer.add_scalar('Loss/train', losssum / losscount, epoch)

            self.model.eval()
            with torch.no_grad():
                bar = tqdm(validloader)
                total = 0
                losssum = 0
                correct = 0
                losscount = 0
                for (images, labels) in bar:
                    outputs = self.model(images)
                    total += labels.size(0)
                    correct += np.equal(np.argmax(outputs.cpu().numpy(), axis=1), labels.cpu().numpy()).sum()  # noqa: E501
                    loss = criterion(outputs, labels)
                    lossval = loss.data.cpu().numpy()
                    losssum += lossval
                    losscount += 1
                    bar.set_description(f'valid epoch: {epoch:3} loss: {loss.data.cpu().numpy():.4f}')  # noqa: E501
                writer.add_scalar('Loss/valid', losssum / losscount, epoch)
                acc = 100 * correct / total
                writer.add_scalar('Accuracy/valid', acc, epoch)

                if acc > best_acc:
                    torch.save(self.model, self.modelpath)
                    best_acc = acc

        writer.close()
        self.model.eval()

    def save_to_onnx(self, modelpath):
        x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
        torch.onnx.export(
            self.model,
            x,
            modelpath,
            export_params=True,
            opset_version=11
        )

    def convert_input_to_bytes(self, inputdata):
        data = bytes()
        for inp in inputdata.detach().cpu().numpy():
            data += inp.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata):
        result = []
        singleoutputsize = self.numclasses * np.dtype(np.float32).itemsize
        for ind in range(0, len(outputdata), singleoutputsize):
            arr = np.frombuffer(
                outputdata[ind:(ind + singleoutputsize)],
                dtype=np.float32
            )
            result.append(arr)
        return torch.FloatTensor(result)

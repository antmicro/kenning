#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a generic PyTorch Autoencoder classification model wrapper.
"""

from functools import partial
from typing import Callable, TypeVar

import numpy as np
from tqdm import tqdm

from kenning.core import metrics
from kenning.modelwrappers.classification.pytorch_generic import (
    PyTorchGenericClassification,
)
from kenning.utils.logger import LoggerProgressBar

DataLoader = TypeVar("torch.utils.data.DataLoader")
Optimizer = TypeVar("torch.optim.Optimizer")
Tensor = TypeVar("torch.Tensor")


class PyTorchGenericAutoencoderClassification(PyTorchGenericClassification):
    """
    Wrapper for autoencoder based PyTorch classfication models.
    """

    def train_model(self):
        import torch
        import torch.optim as optim

        ae_train_loader, _ = self._prepare_loaders(select_class=0)

        ae_model = self.extra_models[0]

        ae_criterion = torch.nn.MSELoss()
        ae_opt = optim.Adam(ae_model.parameters(), lr=self.learning_rate)

        ae_model.to(self.device)
        ae_model.train()
        self._train_model_ae(
            ae_model,
            ae_train_loader,
            ae_opt,
            ae_criterion,
        )

        train_loader, test_loader = self._prepare_loaders()

        # freeze AE
        self.model.train()
        ae_model.eval()
        for param in ae_model.parameters():
            param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(
            [
                param
                for param in self.model.parameters()
                if param.requires_grad
            ],
            lr=self.learning_rate,
        )

        self.model.to(self.device)
        self._train_model_threshold(
            self.model,
            train_loader,
            opt,
            criterion,
        )

        self.model.eval()
        self._test_model(
            test_loader,
            self._prepare_postprocess(),
            partial(metrics.f1_score, average="binary", pos_label=1),
            40,
        )

        self.save_model(self.model_path)

        self.save_model(
            self.model_path.with_stem(f"{self.model_path.stem}_final"),
        )

    def _train_model_ae(
        self,
        model,
        train_loader: DataLoader,
        opt: Optimizer,
        criterion: Callable[[Tensor], Tensor],
        epoch_start_hook: Callable[[int], None] = None,
    ):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self.logdir)
        for epoch in range(self.num_epochs):
            if epoch_start_hook:
                epoch_start_hook(epoch)

            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0
            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, _ in bar:
                    input = input.to(self.device)
                    opt.zero_grad()

                    outputs = model(input)
                    loss = criterion(outputs, input)

                    loss.backward()
                    opt.step()

                    loss_sum += loss
                    loss_count += 1
                    bar.set_description(
                        f"train epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.3f}"
                    )

            writer.add_scalar(
                "Loss/train", loss_sum.data.cpu().numpy() / loss_count, epoch
            )

    def _train_model_threshold(
        self,
        model,
        train_loader: DataLoader,
        opt: Optimizer,
        criterion: Callable[[Tensor], Tensor],
        epoch_start_hook: Callable[[int], None] = None,
    ):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self.logdir)
        for epoch in range(self.num_epochs):
            if epoch_start_hook:
                epoch_start_hook(epoch)

            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0
            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, label in bar:
                    input = input.to(self.device)
                    label = label.to(self.device)
                    opt.zero_grad()

                    outputs = model(input)
                    loss = criterion(outputs, label)

                    loss.backward()
                    opt.step()

                    loss_sum += loss
                    loss_count += 1
                    bar.set_description(
                        f"train epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.3f}"
                    )

            writer.add_scalar(
                "Loss/train", loss_sum.data.cpu().numpy() / loss_count, epoch
            )

    def _test_model(
        self,
        test_loader,
        postprocess,
        metric_func,
        epoch,
    ):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        predicted = np.array([])
        true = np.array([])
        writer = SummaryWriter(log_dir=self.logdir)
        with torch.no_grad(), LoggerProgressBar() as logger_progress_bar:
            bar = tqdm(test_loader, **logger_progress_bar.kwargs)
            for input, labels in bar:
                input = input.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input)
                print(
                    input[0],
                    self.model.ae(input)[0],
                    self.model.ae.encoder(torch.transpose(input, 1, 2))[0],
                )
                outputs, labels = postprocess(outputs, labels)
                predicted = np.concatenate([predicted, outputs.cpu().numpy()])
                true = np.concatenate((true, labels.cpu().numpy()))
                metric = metric_func(true, predicted)
                bar.set_description(
                    f"valid epoch: {epoch:3d} {str(metric_func)}: "
                    f"{metric:.3f}"
                )

            writer.add_scalar(f"{str(metric_func)}/valid", metric, epoch)

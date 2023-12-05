# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import repdl

from torchvision import datasets
from torchvision.transforms import ToTensor


class torchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)


class repdlModel(torchModel):
    def __init__(self):
        super().__init__()
        self.net = repdl.from_torch_module(self.net)


def init():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())
    global data_loaders
    data_loaders = {
        "train": torch.utils.data.DataLoader(
            train_data, batch_size=100, shuffle=True, num_workers=1
        ),
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=100, shuffle=False, num_workers=1
        ),
    }


def train(model, device):
    model.to(device)
    optimizer = repdl.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 2
    total_step = len(data_loaders["train"])
    print(f"Hash of the initial model: {repdl.utils.get_hash(model)}")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loaders["train"]):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = repdl.nn.functional.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"    Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
                )
    print(f"Hash of the trained model: {repdl.utils.get_hash(model)}")


def test(model, device):
    model.to(device).eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = None
        for images, labels in data_loaders["test"]:
            images = images.to(device)
            labels = labels.to(device)
            test_output = model(images)
            outputs = (
                test_output if outputs is None else torch.cat((outputs, test_output))
            )
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        print(f"Test accuracy of the model on the {total} test images: {correct/total}")
    print(f"Hash of the logits: {repdl.utils.get_hash(outputs)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torchModel()
    model = repdl.from_torch_module(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    init()
    train(model, device)
    test(model, device)

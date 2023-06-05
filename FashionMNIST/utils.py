import argparse

import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor


def timer(fn):
    def timer_wrapper(*args, **kwargs):
        import time

        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        print(f"Time taken for {fn.__name__}: {end - start:.2f}s")

    return timer_wrapper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", "-l", type=float, default=1e-3, help="Sets learning rate."
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=64, help="Sets batch size for training."
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="Sets the epoch numbers for training.",
    )
    parser.add_argument("--use_gpu", type=bool, default=True)

    args, _ = parser.parse_known_args()

    return args


train_data = datasets.FashionMNIST(
    root="./FashionMnist",
    train=True,
    download=False,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="./FashionMnist",
    train=False,
    download=False,
    transform=ToTensor(),
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

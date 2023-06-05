import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import init_args


class LitModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )
        self.lr = lr

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class FashionMnistDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # transforms
        transform = ToTensor()
        # datasets
        self.fashion_mnist_train = datasets.FashionMNIST(
            root="./FashionMnist", train=True, download=True, transform=transform
        )
        self.fashion_mnist_test = datasets.FashionMNIST(
            root="./FashionMnist", train=False, download=True, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(self.fashion_mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fashion_mnist_test, batch_size=self.batch_size)


if __name__ == "__main__":
    args = init_args()

    data_module = FashionMnistDataModule(batch_size=args.batch_size)

    model = LitModel(lr=args.lr)

    trainer = pl.Trainer(
        accelerator="gpu", devices=torch.cuda.device_count(), max_epochs=args.epochs
    )

    import time

    start = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f"Training time: {time.time() - start}s")

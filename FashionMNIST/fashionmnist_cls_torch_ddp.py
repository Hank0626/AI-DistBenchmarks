import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import NeuralNetwork, init_args, test_data, timer, train_data


def train_epoch(dataloader, model, loss_fn, optimizer, device, rank):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[Rank {rank}] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


@timer
def train_fashion_mnist(lr, batch_size, epochs, device, rank, world_size):
    # Initialize the distributed environment.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create data loaders.
    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # Create model.
    model = NeuralNetwork().to(device)
    model = DistributedDataParallel(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer, device, rank)
        loss = validate_epoch(test_dataloader, model, loss_fn, device)
        print(f"Epoch {epoch+1}: Loss = {loss}")


def main_worker(rank, args, world_size):
    device = torch.device("cuda", rank) if args.use_gpu else torch.device("cpu")
    print(f"Using {device} device")

    train_fashion_mnist(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        rank=rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    args = init_args()

    device = "cuda" if args.use_gpu else "cpu"

    print(f"Using {device} device")

    num_gpus = torch.cuda.device_count()
    args.world_size = num_gpus

    mp.spawn(main_worker, nprocs=num_gpus, args=(args, num_gpus), daemon=True)

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn,
               optimizer: Optimizer = None, verbose=True):

    loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for X, y in tqdm(dataloader):

        optimizer.zero_grad()

        probs = model(X)
        batch_loss = loss_fn(probs, y)

        batch_loss.backward()
        optimizer.step()

        labels = torch.argmax(y, dim=-1, keepdim=True)
        pred = torch.argmax(probs, dim=-1, keepdim=True)

        correct += torch.sum(labels == pred).item()
        loss += batch_loss

    loss = loss/num_batches
    accuracy = correct/size

    if verbose:
        print(f'Training metrics: average loss {loss} and accuracy {accuracy*100}% for epoch')

    return {
        'loss': loss,
        'accuracy': accuracy
    }


def eval_loop(dataloader: DataLoader, model: nn.Module, loss_fn, verbose=True):

    loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for X, y in tqdm(dataloader):

        with torch.no_grad():
            probs = model(X)
            batch_loss = loss_fn(probs, y)

        labels = torch.argmax(y, dim=-1, keepdim=True)
        pred = torch.argmax(probs, dim=-1, keepdim=True)

        correct += torch.sum(labels == pred).item()
        loss += batch_loss

    loss = loss/num_batches
    accuracy = correct/size

    if verbose:
        print(f'Evaluation metrics: average loss {loss} and accuracy {accuracy*100}% for epoch')

    return {
        'loss': loss,
        'accuracy': accuracy
    }
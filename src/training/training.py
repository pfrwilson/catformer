import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn,
               optimizer: Optimizer = None, verbose=True, epoch=None):

    loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with tqdm(dataloader) as pbar:

        if epoch is not None:
            pbar.set_description(f'TRAIN: EPOCH {epoch}')

        for X, y in pbar:

            batch_size = len(X)

            optimizer.zero_grad()

            logits = model(X)
            batch_loss = loss_fn(logits, y)

            batch_loss.backward()
            optimizer.step()

            labels = torch.argmax(y, dim=-1, keepdim=True)
            pred = torch.argmax(logits, dim=-1, keepdim=True)

            batch_correct = torch.sum(labels == pred).item()
            correct += batch_correct
            loss += batch_loss

            pbar.set_postfix({
                'loss': batch_loss.item(),
                'accuracy': batch_correct/batch_size
            })

        loss = loss/num_batches
        accuracy = correct/size

        pbar.set_postfix({
            'loss': loss,
            'accuracy': accuracy
        })

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
            logits = model(X)
            batch_loss = loss_fn(logits, y)

        labels = torch.argmax(y, dim=-1, keepdim=True)
        pred = torch.argmax(logits, dim=-1, keepdim=True)

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
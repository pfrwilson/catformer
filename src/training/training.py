import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn,
               optimizer: Optimizer = None, epoch=None):

    loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with tqdm(dataloader) as pbar:

        if epoch is not None:
            pbar.set_description(f'TRAIN: EPOCH {epoch}')
        else:
            pbar.set_description('TRAIN:')

        for X, y in pbar:

            batch_size = len(X)

            optimizer.zero_grad()

            logits = model(X)
            batch_loss = loss_fn(logits, y)

            batch_loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=-1, keepdim=True)

            batch_correct = torch.sum(y == pred).item()
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

    return {
        'loss': loss,
        'accuracy': accuracy
    }


def eval_loop(dataloader: DataLoader, model: nn.Module, loss_fn, verbose=True):

    loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with tqdm(dataloader) as pbar:

        pbar.set_description('EVALUATION:')

        for X, y in pbar:

            batch_size = len(X)

            with torch.no_grad():
                logits = model(X)
                batch_loss = loss_fn(logits, y)

            pred = torch.argmax(logits, dim=-1, keepdim=True)

            batch_correct = torch.sum(y == pred).item()
            correct += batch_correct
            loss += batch_loss

            pbar.set_postfix({
                'loss': batch_loss.item(),
                'accuracy': batch_correct / batch_size
            })

        loss = loss/num_batches
        accuracy = correct/size

        pbar.set_postfix({
            'loss': loss,
            'accuracy': accuracy
        })

    return {
        'loss': loss,
        'accuracy': accuracy
    }


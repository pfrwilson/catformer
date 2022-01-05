from torch.utils.data import DataLoader
import hydra
import torch.utils.data
from omegaconf import DictConfig
from src.training.training import train_loop, eval_loop
from src.datasets.cats_dogs import DogsVsCats
from src.models.factory import ModelFactory
import os
import shutil

CONFIG_PATH = os.path.join(os.getcwd(), 'scripts', 'config.yaml')


@hydra.main(config_path=CONFIG_PATH)
def main(args: DictConfig):

    # SETUP EXPERIMENT DIRECTORY
    expt_dir = args.expt_dir
    shutil.copy(CONFIG_PATH, dst=os.path.join(expt_dir, 'config.yaml'))
    if not os.path.isdir(expt_dir):
        os.mkdir(expt_dir)
    ckpt_dir = os.path.join(expt_dir, 'ckpt')
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not args.training.load_from_latest_ckpt:
        shutil.rmtree(ckpt_dir)
        os.mkdir(ckpt_dir)
    checkpoints = os.listdir(ckpt_dir)


    # DATASET
    dataset = DogsVsCats(args.dataset.root_dir,
                         target_transform=DogsVsCats.get_default_target_transform())

    train_transform = dataset.get_default_transform(
        tuple(args.dataset.target_size),
        use_augmentations=args.dataset.use_augmentations
    )
    test_eval_transform = dataset.get_default_transform(
        tuple(args.dataset.target_size), use_augmentations=False
    )

    train_len = int(0.7*len(dataset))
    cv_len = int(0.1*len(dataset))
    test_len = len(dataset) - train_len - cv_len

    train, cv, test = torch.utils.data.random_split(
        dataset, (train_len, cv_len, test_len)
    )
    train.dataset.transform = train_transform
    cv.dataset.transform = test_eval_transform
    test.dataset.transform = test_eval_transform

    datasets = {
        'train': train,
        'val': cv,
        'test': test
    }

    # MODEL
    model_factory = ModelFactory(args.model)
    model = model_factory.build_model()
    if args.training.load_from_latest_ckpt:
        model.load_state_dict(torch.load(checkpoints[-1]))

    # OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)

    # TRAINING
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    first_epoch = len(checkpoints)
    for epoch in range(first_epoch, args.training.epochs + first_epoch):

        metrics = train_loop(
            DataLoader(datasets['train'], batch_size=args.training.batch_size),
            model,
            torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            epoch=epoch
        )

        val_metrics = eval_loop(
            DataLoader(datasets['val'], batch_size=args.training.batch_size),
            model,
            torch.nn.CrossEntropyLoss(),
        )

        history['loss'].append(metrics['loss'])
        history['accuracy'].append(metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'))


if __name__ == '__main__':
    main()
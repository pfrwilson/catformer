from torch.utils.data import DataLoader
import hydra
import torch.utils.data
from omegaconf import DictConfig
from src.training.training import train_loop, eval_loop
from src.datasets.cats_dogs import DogsVsCats
from src.models.factory import ModelFactory


@hydra.main(config_path='../config.yaml')
def main(args: DictConfig):

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

    # OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)

    # TRAINING
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(args.training.epochs):
        print(f'EPOCH {epoch}/{args.training.epochs - 1}')

        metrics = train_loop(
            DataLoader(datasets['train'], batch_size=args.training.batch_size),
            model,
            torch.nn.CrossEntropyLoss(),
            optimizer=optimizer
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


if __name__ == '__main__':
    main()
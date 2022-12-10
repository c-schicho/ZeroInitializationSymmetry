from torch import Generator

from data import get_dataloader, seed_worker
from train.trainer import Trainer
from train.trainer_config import TrainerConfig


def run_experiment(model, dataset: str, config: TrainerConfig, seed: int):
    print(f"### Running experiment {config.model_name} ###")
    generator = Generator()
    generator.manual_seed(seed)

    flatten = "FNN" in config.model_name

    test_loader = get_dataloader(dataset=dataset, train=False, batch_size=config.batch_size, flatten=flatten,
                                 transform=config.transform)
    train_loader, validation_loader = get_dataloader(dataset=dataset, train=True, batch_size=config.batch_size, flatten=flatten,
                                  num_workers=1, worker_init_fn=seed_worker, generator=generator,
                                  transform=config.transform)

    model.zero_initialization(config.initialization_mode, config.initialization_factor)

    trainer = Trainer(model, optimizer=config.optimizer, model_name=config.model_name)
    trainer.train(train_loader, validation_loader, num_epochs=config.epochs)

    trainer.test(test_loader)
    print("\n")

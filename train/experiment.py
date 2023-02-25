import copy
from typing import Union

from torch import Generator

from data import get_dataloader, seed_worker
from train.trainer import Trainer
from train.trainer_config import TrainerConfig


def run_experiment(
        model,
        dataset: str,
        config: TrainerConfig,
        seed: Union[int, None] = None,
        train_summary: bool = False,
        validation_summary: bool = False,
        validate_after_epoch: bool = False,
        test_after_epoch: bool = False,
        test_final_model: bool = False,
        runs: int = 1
):
    print(f"### Running experiment {config.model_name} ###")

    for run in range(1, runs + 1):
        train_model = copy.deepcopy(model)

        if runs > 1:
            print(f"### {run}. RUN ###")

        generator = Generator()

        if seed is not None:
            generator.manual_seed(seed)

        flatten = "FNN" in config.model_name

        test_loader = get_dataloader(dataset=dataset, train=False, batch_size=config.batch_size, flatten=flatten,
                                     transform=config.transform_test)
        train_loader, validation_loader = get_dataloader(dataset=dataset, train=True, batch_size=config.batch_size,
                                                         flatten=flatten, num_workers=1, worker_init_fn=seed_worker,
                                                         generator=generator, transform=config.transform_train)

        train_model.zero_initialization(config.initialization_mode, config.initialization_factor)

        trainer = Trainer(train_model, optimizer=config.optimizer, model_name=config.model_name)
        trainer.train(train_loader=train_loader, validation_loader=validation_loader, test_loader=test_loader,
                      num_epochs=config.epochs, train_summary=train_summary, validate_after_epoch=validate_after_epoch,
                      test_after_epoch=test_after_epoch, validation_summary=validation_summary)

        if test_final_model:
            trainer.test(test_loader)

        print("\n")

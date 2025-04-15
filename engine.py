from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import datasets
import transformers
import wandb
import pandas as pd
from dataclasses import dataclass
import evaluate

# just for clearing output in jupyter notebook
import IPython


@dataclass
class TraininingConfig:
    max_epochs: int = 50
    lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    weight_decay: float = 0.01
    num_warmup_steps: int = 10
    train_log_interval: int = 10
    val_log_interval: int = 10
    max_patience: int = 5
    output_dir: str = "checkpoints"
    batch_size: int = 512
    num_workers: int = 4

    def get(self, key, default=None):
        return getattr(self, key, default)


class MyTrainer:
    def __init__(
        self,
        config: DictConfig,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        val_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        # labels: list[str] = ["negative", "neutral", "positive"],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[wandb.sdk.wandb_run.Run] = None,
        metrics_to_save_best: Optional[List[str]] = ["val/metrics/accuracy"],
        device: Optional[torch.device] = "cuda",
        collate_fn: Optional[Callable] = None,
    ):

        self.config = config
        self.device = device
        self._logger = logger
        self.setup_output_dir()
        # main components
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # initialize some variables for training loop
        self.cur_step: int = -1
        self.cur_epoch: int = -1
        self.exit_by_patience: bool = False
        self.current_patience: int = -1
        self.max_patience: int = config.get("patience", math.inf)
        self.best_metrics_values: Dict[str, Any] = {
            **{key: -1 for key in metrics_to_save_best}
        }
        self.history_metrics: List[Dict[str, Any]] = []

        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.build_dataloader(train_dataset)
        
        # optimizer and scheduler will be setup before training loop, note here
        # self.setup_optimizers_before_training(config)
        print(self.config)

    def setup_optimizers_before_training(self, config):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            betas=config.get("betas", (0.9, 0.995)),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get("num_warmup_steps", 100),
            num_training_steps=config.get("max_epochs", 10)
            * len(self.train_dataloader),
            num_cycles=config.get("num_cycles", 0.5),
            last_epoch=self.cur_epoch,
        )

    def build_dataloader(self, dataset):
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )

    def setup_output_dir(self):
        self.project_name = self._logger.project
        self.group_name = self._logger.group
        self.experiment_name = self._logger.name
        if self._logger is None:
            return
        output_dir = self.config.get("output_dir", "checkpoints")
        prefix = (
            f"{output_dir}/{self.project_name}/{self.group_name}/{self.experiment_name}"
        )
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        with open(f"{prefix}/config.yaml", "w") as f:
            OmegaConf.save(self.config, f)
        self.checkpoint_prefix = prefix

    def extract_loss(self, output, validation=False) -> torch.Tensor:
        if hasattr(output, "losses"):
            losses = output.losses
        elif isinstance(output, torch.Tensor):
            losses = {"total": output}
        else:
            losses = output
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                loss_reduce = losses[key].detach()
                if validation:
                    mode = "validation"
                else:
                    mode = "train"
                self.log(
                    f"{mode}/losses/{key}",
                    loss_reduce.item(),
                )
                # print(f"{mode}/losses/{key}", loss_reduce.item())
        return total_loss

    def train(self) -> None:
        if not hasattr(self, "optimizer"):
            self.setup_optimizers_before_training(self.config)
            
        model, optimizer, scheduler = (self.model, self.optimizer, self.scheduler)
        # model.to(self.device)
        pbar_train_dataloader = tqdm(
            self.train_dataloader, total=len(self.train_dataloader), desc="Training"
        )
        # Below training loop use model (not self.model) and optimizer (not self.optimizer)
        while True:  # until meet max epoch or max patience
            self.cur_epoch += 1
            # dataloader = self.train_dataloader
            pbar_train_dataloader.reset()
            if self.cur_epoch % 3 == 0:
                IPython.display.clear_output(wait=True)
            print(f"Current Epoch {self.cur_epoch}")
            for data in pbar_train_dataloader:
                self.cur_step += 1
                # CHECK EXIT CONDITION
                if (
                    self.cur_epoch >= self.config.max_epochs
                    or self.exit_by_patience == True
                ):
                    print("Exit requirement reached, exiting")
                    self.save_checkpoint(for_last=True)
                    return self.get_training_results()
                # FORWARD PASS
                model.train()
                data = self.move_to_device(data, self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    if type(data) is dict:
                        output = model(**data)
                    elif type(data) is list:
                        output = model(*data)
                    else:
                        output = model(data)

                    total_loss = self.extract_loss(output)
                # BACKWARD PASS
                total_loss.backward()
                optimizer.step()
                # LEARNING RATE MONITOR
                if self.cur_step % self.config.train_log_interval == 0:
                    for index_group in range(len(optimizer.param_groups)):
                        lr = optimizer.param_groups[index_group]["lr"]
                        self.log(f"train/lr_group_{index_group}", lr)
                    print("Learning rate: ", lr)
                    print("Total loss: ", total_loss.item())
                scheduler.step()
            self.on_validate_start()

    def move_to_device(self, obj, device):
        if isinstance(obj, dict):
            d = {}
            for k, v in obj.items():
                d[k] = self.move_to_device(v, device)
            return d
        if isinstance(obj, list):
            l = []
            for v in obj:
                l.append(self.move_to_device(v, device))
            return l
        if isinstance(obj, str):
            return obj
        return obj.to(device)

    def on_validate_start(self):
        metrics_dict = self.validate()
        self.handle_checkpoint_with_patience(metrics_dict, set_patience=True)
        self.setup_for_patience_callback()
        torch.cuda.empty_cache()

    def validate(self):
        validation_loader = self.val_dataloader
        model = self.model

        model.eval()
        print("Evaluating")
        with torch.no_grad():
            metrics = self.evaluate(validation_loader)
            if metrics is not None:
                history_obj = {}
                for key in metrics:
                    log_key = f"val/metrics/{key}"
                    self.log(
                        log_key,
                        metrics[key],
                    )
                    history_obj[key] = round(metrics[key], 4)
                    print(f"{key}: {metrics[key]}")
                self.history_metrics.append(
                    {**history_obj, "epoch": self.cur_epoch, "step": self.cur_step}
                )
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader):
        results_dict = self.classification_evaluate(
            self.model,
            self.val_dataset,
            self.tokenizer,
            collate_fn=self.collate_fn,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            device=self.device,
        )
        return results_dict

    def setup_for_patience_callback(self):
        if self.current_patience > self.max_patience:
            print("Early stopping")
            self.exit_by_patience = True

    def handle_checkpoint_with_patience(
        self, metrics: Dict[str, Any], set_patience=True
    ):
        reset_patience_flag = False
        if len(self.best_metrics_values) == 0:
            self.best_metrics_values = metrics
            return
        for key in self.best_metrics_values:
            short_key = key.split("/")[-1]
            if metrics.get(short_key, -100) > self.best_metrics_values[key]:
                self.best_metrics_values[key] = metrics[short_key]
                self.save_checkpoint(name=f"best_{short_key}")
                reset_patience_flag = True
        if set_patience:
            if reset_patience_flag:
                self.current_patience = 0
            else:
                self.current_patience += 1

    def save_checkpoint(self, name="last", for_last=False):
        model_no_ddp = self.model
        check_point_file_path = (
            f"{self.checkpoint_prefix}/{name}.pt"
            if not for_last
            else f"{self.checkpoint_prefix}/last.pt"
        )
        model_state_dict = model_no_ddp.state_dict()
        save_obj = {
            "model": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
            "epoch": self.cur_epoch,
            "history": self.history_metrics,
            "best_metrics": self.best_metrics_values,
            "patience": self.current_patience,
        }
        torch.save(
            save_obj,
            check_point_file_path,
        )
        torch.cuda.empty_cache()

    def log(
        self,
        name: str,
        value: Union[torch.Tensor, float, int],
    ):
        if type(self._logger) is not None:
            self._logger.log({name: value, "epoch": self.cur_epoch}, step=self.cur_step)

    def get_training_results(self):
        history_metrics = pd.DataFrame(self.history_metrics).round(4)
        print(history_metrics)
        return {
            "best_metrics": self.best_metrics_values,
            "patience": self.current_patience,
            "history": history_metrics,
        }

    def load_best_checkpoint(self):
        best_checkpoint_path = f"{self.checkpoint_prefix}/best_accuracy.pt"
        checkpoint = torch.load(best_checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.cur_epoch = checkpoint["epoch"]
        self.history_metrics = checkpoint["history"]
        self.best_metrics_values = checkpoint["best_metrics"]
        self.current_patience = checkpoint["patience"]
        return checkpoint

    @staticmethod
    def calculate_accuracy(model, dataloader):
        """Function calculates accuracy of given model on dataloader

        Args:
            model : ...
            dataloader (DataLoader): evaluation dataloader

        Returns:
            float: model's accuracy
        """
        # create metric computer
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        list_metrics = [accuracy_metric, f1_metric, precision_metric, recall_metric]
        # evaluate model
        predictions_list = []
        references_list = []
        device = model.device

        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Evaluate model on val dataset"
        ):
            batch["labels"] = batch["labels"].to(device)
            predictions = model(**batch)["logits"]
            predictions_list.append(torch.argmax(predictions, dim=1))
            references_list.append(batch["labels"])

        results_dict = {}
        for metric in list_metrics:
            if metric.name == "accuracy":
                result_dict = metric.compute(
                    predictions=torch.concat(predictions_list),
                    references=torch.concat(references_list),
                )
            else:
                result_dict = metric.compute(
                    predictions=torch.concat(predictions_list),
                    references=torch.concat(references_list),
                    average="macro",
                )
            results_dict.update(result_dict)

        # rename f1 to macro_f1
        results_dict["macro_f1"] = results_dict.pop("f1")
        results_dict["macro_precision"] = results_dict.pop("precision")
        results_dict["macro_recall"] = results_dict.pop("recall")

        return results_dict

    @staticmethod
    @torch.no_grad()
    def classification_evaluate(
        model: torch.nn.Module,
        dataset: datasets.Dataset,
        tokenizer: None,
        collate_fn: Callable,
        batch_size: int = 2048,
        num_workers: int = 2,
        device: str = "cuda",
    ):

        classifier = model
        test_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        classifier = classifier.to(device)
        results_dict = MyTrainer.calculate_accuracy(classifier, test_dataloader)
        print(f"Evaluate metrics: {results_dict}")
        return results_dict


def train_evaluate_pipeline(
    config: DictConfig,
    model: torch.nn.Module,
    train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
    val_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
    test_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
    collate_fn: Callable,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    device: str = "cuda",
    num_workers: int = 2,
    logger: Optional[wandb.sdk.wandb_run.Run] = None,
):
    
    if logger is not None:
        import datetime
        logger = wandb.init(
            anonymous="allow",
            project="<finetune><clip><har_dataset>",
            group="finetune_har_dataset",
            name=str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-")) + "- finetune_vlsp",
            config=config,
            notes="",
            tags=["finetune", "nlp_project", "vlsp_dataset"],
        )
    IPython.display.clear_output(wait=True)
        
    trainer = MyTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
        device=device,
        collate_fn=collate_fn,
    )
    trainer.train()
    trainer.save_checkpoint(for_last=True)
    training_results = trainer.get_training_results()

    # load best checkpoint
    trainer.load_best_checkpoint()

    test_results = trainer.classification_evaluate(
        trainer.model,
        test_dataset,
        tokenizer,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(f"Test results: {test_results}")
    return {
        "training_results": training_results,
        "test_results": test_results,
    }

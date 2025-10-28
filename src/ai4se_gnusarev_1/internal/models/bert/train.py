import os

import torch
from torch.utils.data import DataLoader
import yaml
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryFBetaScore,
)
from torchmetrics import Metric
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from ai4se_gnusarev_1.internal.models.bert.data import (
    ToxicReviewDataset,
    collate_with_tokenizer
)
from ai4se_gnusarev_1.internal.models.bert.consts import HF_HOME


class BinarySupport(Metric):
    def __init__(self, pos_label: int = 1):
        super().__init__()
        self.add_state(
            name="support", default=torch.tensor(0), dist_reduce_fx="mean"
        )
        self.pos_label = pos_label

    def update(self, _, target):
        self.support += (target == self.pos_label).sum() / len(target)

    def compute(self):
        return self.support.item()


class Trainer:
    def __call__(self, cfg_path: str, save_path: str):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        # init model
        tokenizer = RobertaTokenizer.from_pretrained(
            cfg["model"],
            cache_dir=HF_HOME,
            use_safetensors=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"],
            cache_dir=HF_HOME,
            use_safetensors=True,
            num_labels=2,
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # init metrics
        self.metrics = [
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryFBetaScore(beta=1.0),
            BinarySupport(),
        ]
        # prepare data
        self.train_dataloader = DataLoader(
            dataset=ToxicReviewDataset(cfg["data"]["train"]),
            batch_size=cfg["data"]["batch_size"],
            collate_fn=collate_with_tokenizer(tokenizer),
        )
        test_dataloader = DataLoader(
            dataset=ToxicReviewDataset(cfg["data"]["test"]),
            batch_size=cfg["data"]["batch_size"],
            collate_fn=collate_with_tokenizer(tokenizer),
        )
        # other variables
        epoch_number = 0
        best_vloss = 1_000_000
        num_epochs = cfg["num_epochs"]

        for epoch in tqdm(range(num_epochs)):
            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch()

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            for metric in self.metrics:
                metric.reset()
            with torch.no_grad():
                for i, (vinputs, vlabels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False):
                    voutputs = self.model(**vinputs).logits
                    vloss = self.loss_fn(voutputs, vlabels)
                    self.compute_metrics(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print("EPOCH {}LOSS train {} test {}".format(epoch, avg_loss, avg_vloss))
            for metric in self.metrics:
                metric.compute()
                print("TEST  {}: {}".format(metric.__class__.__name__, metric))

            epoch_number += 1
        # save last model
        model_path = "last_model"
        torch.save(
            self.model.state_dict(), os.path.join(save_path, model_path)
        )

        return best_vloss

    def train_one_epoch(self):
        running_loss = 0.0
        last_loss = 0.0

        for i, (inputs, labels) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(**inputs).logits

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            # compute metrics
            self.compute_metrics(outputs, labels)

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0
                for metric in self.metrics:
                    metric.compute()
                    print(
                        "  batch {} {}: {}".format(
                            i + 1, metric.__class__.__name__, metric
                        )
                    )

        return last_loss

    def compute_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ):
        for metric in self.metrics:
            metric(preds, target)

trainer = Trainer()

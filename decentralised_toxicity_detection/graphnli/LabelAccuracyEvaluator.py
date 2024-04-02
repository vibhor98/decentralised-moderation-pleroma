import torch
from torch.utils.data import DataLoader
import logging
import os
import csv
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error


logger = logging.getLogger(__name__)

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class LabelAccuracyEvaluator():
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "mse", "accuracy", "f1_score", "precision", "recall"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        true_labels = []
        pred_labels = []
        glob_prediction = []
        glob_labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, labels = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            labels = labels.to(model.device)
            label_ids = labels.clone()
            label_ids[label_ids > 0.5] = 1
            label_ids[label_ids <= 0.5] = 0

            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            pred_ids = prediction[:, 1].clone()
            pred_ids[pred_ids > 0.5] = 1
            pred_ids[pred_ids <= 0.5] = 0

            #total += prediction.size(0)
            #correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            true_labels.extend(list(label_ids.cpu()))
            pred_labels.extend(list(pred_ids.cpu()))
            glob_prediction.extend(list(prediction[:, 1].cpu()))
            glob_labels.extend(list(labels.cpu()))
        #accuracy = correct/total
        mse = mean_squared_error(glob_labels, glob_prediction)
        f1 = f1_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)

        # df = pd.DataFrame()
        # df['prediction'] = [x.item() for x in glob_prediction]
        # df['toxicity'] = [x.item() for x in glob_labels]
        # df.to_csv('inference_predictions.csv', index=False)

        # logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, mse, acc, f1, precision, recall])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, mse, acc, f1, precision, recall])
        return acc


# Hate Speech Regression using BERT

import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoConfig, BertTokenizer, BertModel, BertPreTrainedModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_ids, attention_masks, label_ids):
        'Initialization'
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_ids)

  def __getitem__(self, index):
        'Generates one sample of data'
        input = self.input_ids[index]
        mask = self.attention_masks[index]
        label_id = self.label_ids[index]
        return input, mask, label_id


class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(768, 128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128, 128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output


def collate_fn(instances):
    batch = []
    for i in range(len(instances[0])):
        batch.append(torch.stack([torch.tensor(instance[i]) for instance in instances], 0))
    return batch


def dataloader(input_ids, attention_mask, label_ids):
    # tensor_dataset = TensorDataset(input_ids, attention_mask, label_ids)
    data_set = Dataset(input_ids, attention_mask, label_ids)

    dataloader = DataLoader(data_set, batch_size=16, shuffle=True, drop_last=False, collate_fn=collate_fn)
    return dataloader


def bert_tokenization(text):
    input_ids = []
    attention_masks = []
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for i, sent in enumerate(text):
        bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=75,
            pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    # input_ids = np.asarray(input_ids)
    # attention_masks = np.array(attention_masks)
    input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    return input_ids, attention_masks


def save_checkpoint(save_path, model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch
    }, save_path)

    print('Saved model checkpoint to {}.'.format(save_path))


def do_check_and_update_metrics(val_mse, epoch, losses, model, optimizer):
    global best_val_mse
    global best_epoch
    if val_mse <= best_val_mse:
        best_val_mse = val_mse
        best_epoch = epoch
        # save_checkpoint('./checkpoints/' + 'epoch' + str(epoch), model, optimizer)

    new_metrics = {
        'epoch': epoch,
        'train_loss': sum(losses) / len(losses),
        'val_mse': val_mse,
    }
    print('epoch {}: training loss {:.4f}, '
        'validation mse: {:.4f}, (history best: {:.4f} at epoch {}). '
            .format(new_metrics['epoch'], new_metrics['train_loss'],
                new_metrics['val_mse'], best_val_mse, best_epoch))


def train_and_validate(train_dataloader, val_dataloader, test_dataloader):
    EPOCHS = 3
    learning_rate = 1e-5

    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = BertRegresser.from_pretrained('bert-base-uncased', config=config)
    model.cuda()
    model.train()

    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):
        t0 = time.time()
        pred_all = []
        labels_all = []
        loss_all = []
        for inputs, masks, label_ids in train_dataloader:
            # forward
            outputs = model(input_ids=inputs, attention_mask=masks)
            loss = loss_fcn(outputs.float(), label_ids.float())
            pred = outputs

            pred_all.extend(pred.detach().cpu().numpy())
            labels_all.extend(label_ids.detach().cpu().numpy())
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dur = time.time() - t0

        mse = mean_squared_error(pred_all, labels_all)
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | Train MSE {:.4f}"
             .format(epoch, dur, np.mean(loss_all), mse))

        pred_all = []
        labels_all = []
        for inputs, masks, label_ids in val_dataloader:
            outputs = model(input_ids=inputs, attention_mask=masks)
            pred = outputs
            pred_all.extend(pred.detach().cpu().numpy())
            labels_all.extend(label_ids.detach().cpu().numpy())

        do_check_and_update_metrics(mean_squared_error(pred_all, labels_all), epoch, loss_all, model, optimizer)

    model.eval()

    pred_all = []
    labels_all = []
    pred_ids = []
    labels = []
    for inputs, masks, label_ids in test_dataloader:
        outputs = model(input_ids=inputs, attention_mask=masks)
        pred_all.extend(outputs.detach().cpu().numpy())
        labels_all.extend(label_ids.detach().cpu().numpy())

        pred = outputs.clone().detach()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred_ids.extend(pred.cpu().numpy())
        label = label_ids.clone()
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        labels.extend(label.cpu().numpy())

    print('MSE Loss:', mean_squared_error(pred_all, labels_all))
    print("Test Precision:", precision_score(pred_ids, labels, average='macro'))
    print("Test Recall:", recall_score(pred_ids, labels, average='macro'))
    print("Test F1-score:", f1_score(pred_ids, labels, average='macro'))
    print("Test Accuracy:", accuracy_score(pred_ids, labels))


if __name__ == '__main__':
    best_val_mse = 2.0
    best_epoch = -1

    textdata = pd.read_csv('./pleroma_toxic_random_walks.csv')
    print(textdata.shape)

    split = np.random.choice(['train', 'test'], size=len(textdata), p=[0.85, 0.15])
    textdata['split'] = split

    for i in range(1):
        train_data = textdata[textdata['split'] == 'train']
        train_data = train_data["sent1"]
        #train_data = [s.translate(string.punctuation) for s in train_data]
        train_label_ids = textdata[textdata['split'] == 'train']['toxicity']
        #train_label_ids = list(set(train_label_ids))

        train_data, val_data, train_label_ids, val_label_ids = train_test_split(
            train_data, train_label_ids, test_size=0.15, shuffle=True)

        test_data = textdata[textdata['split'] == 'test']
        test_data = test_data["sent1"]
        test_label_ids = textdata[textdata['split'] == 'test']['toxicity']

        train_data = train_data.reset_index(drop=True)
        train_label_ids = train_label_ids.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        val_label_ids = val_label_ids.reset_index(drop=True)
        test_label_ids = test_label_ids.reset_index(drop=True)

        train_inp, train_mask = bert_tokenization(train_data)
        val_inp, val_mask = bert_tokenization(val_data)
        test_inp, test_mask = bert_tokenization(test_data)

        train_dataloader = dataloader(train_inp.to('cuda'), train_mask.to('cuda'), torch.Tensor(train_label_ids).to('cuda'))
        val_dataloader = dataloader(val_inp.to('cuda'), val_mask.to('cuda'), torch.Tensor(val_label_ids).to('cuda'))
        test_dataloader = dataloader(test_inp.to('cuda'), test_mask.to('cuda'), torch.Tensor(test_label_ids).to('cuda'))

        train_and_validate(train_dataloader, val_dataloader, test_dataloader)

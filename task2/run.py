
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import AutoTokenizer, AutoModel
from data_utils import to_cuda, ClsDataset
from torch.utils.data import DataLoader
from config import cls_setting
from tqdm import tqdm
import json

import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset
import json
import torch
import nltk
from nltk.corpus import stopwords
import string

class TextProcessor:
    def __init__(self):
        self.exclude_punctuation = string.punctuation.replace(':', '').replace('°', '')

    def process(self, text):
        text = text.lower()
        text = ''.join(ch for ch in text if ch not in self.exclude_punctuation)
        words = text.split()
        processed_text = ' '.join(words)
        return processed_text


def to_cuda(batch):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].cuda()


class ClsDataset:
    def __init__(self, mode, label2ids, tok, max_length=512):
        self.text_processor = TextProcessor()
        self.max_length = max_length
        self.dataset = self.load_data(mode)
        self.evidences = self.load_evidence()
        self.label2ids = label2ids
        self.tokenizer = tok
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    @staticmethod
    def load_data(mode):
        if mode != "test":
            f = open("data/{}-claims.json".format(mode), "r")
        else:
            f = open("data/retrieval-test-claims.json", "r")
        data = json.load(f)
        f.close()
        return data

    @staticmethod
    def load_evidence():
        f = open("data/evidence.json", "r")
        evidence = json.load(f)
        f.close()
        return evidence

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [self.text_processor.process(data["claim_text"])]
        for evidence_id in data["evidences"]:
            input_text.append(self.text_processor.process(self.evidences[evidence_id]))
        input_text = self.tokenizer.sep_token.join(input_text)
        if self.mode != "test":
            label = self.label2ids[data["claim_label"]]
        else:
            label = None
        return [input_text, label, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        input_texts = []
        labels = []
        datas = []
        claim_ids = []
        for input_text, label, data, claim_id in batch:
            input_texts.append(input_text)
            datas.append(data)
            claim_ids.append(claim_id)
            if self.mode != "test":
                labels.append(label)

        src_text = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_ids"] = src_text.input_ids
        batch_encoding["attention_mask"] = src_text.attention_mask
        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        if self.mode != "test":
            batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding


class CLSModel(nn.Module):
    def __init__(self, pre_encoder):

        super(CLSModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask):
        texts_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)
        return logits


def predict_test(args):
    # load data
    cls_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
    ids2label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    test_set = ClsDataset("test", label2ids, tok, args.max_length)

    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    # build models
    model = CLSModel(args.model_type)


    assert len(args.model_pt) > 0
    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))

    model.cuda()
    model.eval()

    out_data = {}
    for batch in tqdm(dataloader):
        to_cuda(batch)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1).tolist()
        idx = 0
        for data, predict_label in zip(batch["datas"], predict_labels):
            data["claim_label"] = ids2label[predict_label]
            out_data[batch["claim_ids"][idx]] = data
            idx += 1
    fout = open("test-claims-predictions.json", 'w')
    json.dump(out_data, fout)
    fout.close()


def validate(val_dataloader, model):
    model.eval()
    cnt = 0.
    correct_cnt = 0.
    for batch in tqdm(val_dataloader):
        to_cuda(batch)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1)
        result = predict_labels == batch["label"]
        correct_cnt += result.sum().item()
        cnt += predict_labels.size(0)
    acc = correct_cnt / cnt
    print("\n")
    print("evaluation accuracy: %.3f" % acc)
    print("\n")

    model.train()

    return acc
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (self.num_classes - 1)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def run(args):
    cls_setting(args)

    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_type)

    label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}

    train_set = ClsDataset("train", label2ids, tok, args.max_length)
    val_set = ClsDataset("dev", label2ids, tok, args.max_length)

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn)


    model = CLSModel(args.model_type)

    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache",args.model_pt, "best_ckpt.bin")))
    model.cuda()
    model.train()

    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    save_dir = f"./cache/{date}_cls"
    os.makedirs(save_dir, exist_ok=True)

    # ce_fn = nn.CrossEntropyLoss()
    num_classes = len(label2ids)
    ce_fn = CrossEntropyLabelSmooth(num_classes, smoothing=0.05)
    s_optimizer = optim.Adam(model.parameters())

    # keep lr fixed
    for param_group in s_optimizer.param_groups:
        param_group['lr'] = args.max_lr

    # start training
    s_optimizer.zero_grad()
    step_cnt = 0
    all_step_cnt = 0
    avg_loss = 0
    maximum_acc = 0

    for epoch in range(args.epoch):
        epoch_step = 0
        for (i, batch) in enumerate(tqdm(dataloader)):
            to_cuda(batch)
            step_cnt += 1
            # forward pass
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            loss = ce_fn(logits, batch["label"])
            loss = loss / args.accumulate_step
            loss.backward()

            avg_loss += loss.item()
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 4e-9

                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()

            if all_step_cnt % args.report_freq == 0 and step_cnt == 0:
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 4e-9

                # report stats
                print("\n")
                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_step, avg_loss / args.report_freq))
                print(f"learning rate: {lr:.6f}")
                print("\n")

                avg_loss = 0
            del loss, logits

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("\nEvaluate:\n")
                acc = validate(val_dataloader, model)

                if acc > maximum_acc:
                    maximum_acc = acc
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_ckpt.bin"))
                    print("\n")
                    print("best val loss - epoch: %d, epoch_step: %d" % (epoch, epoch_step))
                    print("maximum_acc", acc)
                    print("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-p", "--predict", action="store_true", help="predict test using the best model")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    args = parser.parse_args()

    if args.predict:
        predict_test(args)
    else:
        run(args)

# nohup python -u cls/main.py -p --model_pt cls >test.out 2>&1 &
# nohup python -u cls/main.py >train.out 2>&1 &

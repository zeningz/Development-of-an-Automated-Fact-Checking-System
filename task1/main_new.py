import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import AutoTokenizer, AutoModel
from data_utils import to_cuda, TrainDataset, ValDataset, EvidenceDataset
from torch.utils.data import DataLoader
from config import dpr_setting
from tqdm import tqdm
import json
import wandb
from datetime import datetime
from rank_bm25 import BM25Okapi
from collections import Counter
wandb.init(project="nlp", name="batch")

def generate_triplets(queries, positive_evidences, negative_evidences, num_triplets):
    triplets = []
    for i in range(num_triplets):
        query = random.choice(queries)
        positive_evidence = random.choice(positive_evidences[query])
        negative_evidence = random.choice(negative_evidences[query])
        triplets.append((query, positive_evidence, negative_evidence))
    return triplets

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    distance_positive = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings, p=2)
    distance_negative = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings, p=2)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def generate_train_neg_samples(args, tok, encoder_model, evidence_embeddings, evidence_ids,evidence_set):
    # load data
    test_set = ValDataset("train", tok, args.max_length)
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    # build models

    k_neg_samples = 16
    # Preprocess evidence texts for BM25
    tokenized_evidence_texts = [text.split() for text in evidence_set]
    
    # Create BM25Okapi object
    bm25 = BM25Okapi(tokenized_evidence_texts)
    
    # Build models
    encoder_model.eval()

        
    out_data = {}
    for batch in tqdm(dataloader):
        to_cuda(batch)
        # Generate BM25 negative samples
        query_texts = [data["claim_text"] for data in batch["datas"]]
        bm25_scores = bm25.get_scores(query_texts)
        bm25_topk_indices = np.argsort(bm25_scores)[::-1][:k_neg_samples]
        # Generate negative samples using your original method
        query_embeddings = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state[:, 0, :]
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1).cpu()
        scores = torch.mm(query_embeddings, evidence_embeddings)
        topk_ids = torch.topk(scores, k=k_neg_samples, dim=1).indices.tolist()
        
        for idx, data in enumerate(batch["datas"]):
            bm25_negative_evidences = []
            original_negative_evidences = []
            
            # Generate BM25 negative samples
            for i in bm25_topk_indices:
                if evidence_ids[i] not in batch["evidences"][idx]:
                    bm25_negative_evidences.append(evidence_ids[i])
            
            # Generate negative samples using your original method
            for i in topk_ids[idx]:
                if evidence_ids[i] not in batch["evidences"][idx]:
                    original_negative_evidences.append(evidence_ids[i])
            nagtive_samples = list(set(bm25_negative_evidences + original_negative_evidences))
            data["negative_evidences"] = nagtive_samples
            out_data[batch["claim_ids"][idx]] = data
    
    
    fout = open("data/train-claims-with-negatives_combine.json", "w")
    json.dump(out_data, fout)
    fout.close()


def get_evidence_embeddings(evidence_dataloader,evidence_model):
    # encoder_model.eval()
    # query_model.eval()
    evidence_model.eval()
    # get evidence embedding and normalise
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        to_cuda(batch)
        # evidence_last = encoder_model(input_ids=batch["evidence_input_ids"],
                                    #   attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = torch.nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()
    evidence_model.train()
    return evidence_embeddings, evidence_ids


def validate(val_dataloader, evidence_embeddings, evidence_ids, query_model):
# def validate(val_dataloader, evidence_embeddings, evidence_ids, encoder_model):
    f = []
    for batch in tqdm(val_dataloader):
        to_cuda(batch)
        # query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=args.evidence_num, dim=1).indices.tolist()

        for idx, data in enumerate(batch["datas"]):
            evidence_correct = 0
            pred_evidences = [evidence_ids[i] for i in topk_ids[idx]]
            for evidence_id in batch["evidences"][idx]:
                if evidence_id in pred_evidences:
                    evidence_correct += 1
            if evidence_correct > 0:
                evidence_recall = float(evidence_correct) / len(batch["evidences"][idx])
                evidence_precision = float(evidence_correct) / len(pred_evidences)
                evidence_fscore = (2 * evidence_precision * evidence_recall) / (evidence_precision + evidence_recall)
            else:
                evidence_fscore = 0
            f.append(evidence_fscore)

        # print("----")

    fscore = np.mean(f)
    print("\n")
    print("Evidence Retrieval F-score: %.3f" % fscore)
    print("\n")
    # encoder_model.train()
    query_model.train()
    # evidence_model.train()
    return fscore
def predict_test(args):
    # load data
    dpr_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    test_set = ValDataset("test", tok, args.max_length)
    evidence_set = EvidenceDataset(tok, args.max_length)

    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)
    # build models
    # encoder_model = AutoModel.from_pretrained(args.model_type)
    query_model = AutoModel.from_pretrained(args.model_type)
    evidence_model = AutoModel.from_pretrained(args.model_type)

    assert len(args.model_pt) > 0
    # encoder_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))
    query_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "query_ckpt.bin")))
    evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "evidence_ckpt.bin")))
    # encoder_model.cuda()
    # encoder_model.eval()

    query_model.cuda()
    evidence_model.cuda()
    query_model.eval()
    evidence_model.eval()

    # get evidence embedding and normalise
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        to_cuda(batch)
        # evidence_last = encoder_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = torch.nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()

    out_data = {}
    for batch in tqdm(dataloader):
        to_cuda(batch)
        # query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=args.evidence_num, dim=1).indices.tolist()
        for idx, data in enumerate(batch["datas"]):
            data["evidences"] = [evidence_ids[i] for i in topk_ids[idx]]
            out_data[batch["claim_ids"][idx]] = data
    fout = open("data/retrieval-test-claims_new.json", 'w')
    json.dump(out_data, fout)
    fout.close()

def run(args):
    dpr_setting(args)
    using_negative = args.using_negative
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    val_set = ValDataset("dev", tok, args.max_length)
    evidence_set = EvidenceDataset(tok, args.max_length)

    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn)
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)
    evidence_texts = [data[1] for (i,data) in enumerate(evidence_set)]
    # build models
    # encoder_model = AutoModel.from_pretrained(args.model_type)

    query_model = AutoModel.from_pretrained(args.model_type)
    evidence_model = AutoModel.from_pretrained(args.model_type)

    if len(args.model_pt) > 0:
        query_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "query_ckpt.bin")))
        evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "evidence_ckpt.bin")))

    # encoder_model.cuda()
    query_model.cuda()
    evidence_model.cuda()
    query_model.eval()
    evidence_model.eval()

    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    save_dir = f"./cache/{date}_combined"
    os.makedirs(save_dir, exist_ok=True)

    query_optimizer = optim.Adam(query_model.parameters())
    evidence_optimizer = optim.Adam(evidence_model.parameters())

    for param_group in query_optimizer.param_groups:
        param_group['lr'] = args.max_lr
    for param_group in evidence_optimizer.param_groups:
        param_group['lr'] = args.max_lr

    query_optimizer.zero_grad()
    evidence_optimizer.zero_grad()
    using_nagtive = False
    step_cnt = 0
    all_step_cnt = 0
    avg_loss = 0
    maximum_f_score = 0

    print("\nEvaluate:\n")
    last_evidence_embeddings, last_evidence_ids = get_evidence_embeddings(evidence_dataloader, evidence_model)
    torch.save(last_evidence_embeddings, "temp_data/evidence_embeddings")
    torch.save(last_evidence_ids, "temp_data/evidence_ids")
    last_evidence_embeddings = torch.load("temp_data/evidence_embeddings")
    last_evidence_ids = torch.load("temp_data/evidence_ids")

    f_score = validate(val_dataloader, last_evidence_embeddings, last_evidence_ids, query_model)
    wandb.log({"f_score": f_score}, step=all_step_cnt)

    print("\n")
    print("maximum_f_score", f_score)
    print("\n")
    # assert 0 == 1
    for epoch in range(args.epoch):
        epoch_step = 0

        generate_train_neg_samples(args, tok, query_model, last_evidence_embeddings, last_evidence_ids,evidence_texts)

        train_set = TrainDataset("train", tok, args.evidence_samples, args.max_length,using_negative=using_negative)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)

        for (i, batch) in enumerate(tqdm(dataloader)):
            to_cuda(batch)
            step_cnt += 1
            if using_nagtive:
                triplets = generate_triplets(batch["queries"], batch["positive_evidences"], batch["negative_evidences"], args.batch_size // 2)

                # 计算三元组损失并反向传播
                for triplet in triplets:
                    query = triplet[0]
                    positive_evidence = triplet[1]
                    negative_evidence = triplet[2]

                    # 编码查询、正向证据和负向证据
                    query_embedding = query_model(input_ids=query["input_ids"], attention_mask=query["attention_mask"]).last_hidden_state[:, 0, :]
                    positive_embedding = query_model(input_ids=positive_evidence["input_ids"], attention_mask=positive_evidence["attention_mask"]).last_hidden_state[:, 0, :]
                    negative_embedding = query_model(input_ids=negative_evidence["input_ids"], attention_mask=negative_evidence["attention_mask"]).last_hidden_state[:, 0, :]

                    # 计算三元组损失并反向传播
                    loss = triplet_loss(query_embedding, positive_embedding, negative_embedding)
                    loss.backward()
                    query_model.step()
                    query_model.zero_grad()
            else:
            # forward pass
                query_embeddings = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
                evidence_embeddings = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
                query_embeddings = query_embeddings[:, 0, :]
                evidence_embeddings = evidence_embeddings[:, 0, :]

                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                evidence_embeddings = torch.nn.functional.normalize(evidence_embeddings, p=2, dim=1)

                cos_sims = torch.mm(query_embeddings, evidence_embeddings.t())
                scores = - torch.nn.functional.log_softmax(cos_sims / 0.001, dim=1)

                loss = []
                start_idx = 0
                for idx, label in enumerate(batch["labels"]):
                    end_idx = start_idx + label
                    cur_loss = torch.mean(scores[idx, start_idx:end_idx])
                    loss.append(cur_loss)
                    start_idx = end_idx

                loss = torch.stack(loss).mean()
                loss = loss / args.accumulate_step
                loss.backward()
            avg_loss += loss.item()
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    # nn.utils.clip_grad_norm_(encoder_model.parameters(), args.grad_norm)
                    nn.utils.clip_grad_norm_(query_model.parameters(), args.grad_norm)
                    nn.utils.clip_grad_norm_(evidence_model.parameters(), args.grad_norm)

                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 1e-8

                for param_group in query_optimizer.param_groups:
                    param_group['lr'] = lr
                for param_group in evidence_optimizer.param_groups:
                    param_group['lr'] = lr

                query_optimizer.step()
                evidence_optimizer.step()
                query_optimizer.zero_grad()
                evidence_optimizer.zero_grad()

            if all_step_cnt % args.report_freq == 0 and step_cnt == 0:
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 1e-8

                wandb.log({"learning_rate": lr}, step=all_step_cnt)
                wandb.log({"loss": avg_loss / args.report_freq}, step=all_step_cnt)
                # report stats
                print("\n")
                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_step, avg_loss / args.report_freq))
                print(f"learning rate: {lr:.6f}")
                print("\n")

                avg_loss = 0
            del loss, cos_sims, query_embeddings, evidence_embeddings

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("\nEvaluate:\n")
                # f_score = validate(val_dataloader, evidence_dataloader, query_model, evidence_model)
                evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, evidence_model)
                f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, query_model)
                wandb.log({"f_score": f_score}, step=all_step_cnt)
                last_evidence_embeddings, last_evidence_ids = evidence_embeddings, evidence_ids

                if f_score > maximum_f_score:
                    maximum_f_score = f_score
                    # torch.save(encoder_model.state_dict(), os.path.join(save_dir, "best_ckpt.bin"))
                    torch.save(query_model.state_dict(), os.path.join(save_dir, "query_ckpt.bin"))
                    torch.save(evidence_model.state_dict(), os.path.join(save_dir, "evidence_ckpt.bin"))

                    torch.save(last_evidence_embeddings, os.path.join(save_dir, "evidence_embeddings"))
                    torch.save(last_evidence_ids, os.path.join(save_dir, "evidence_ids"))
                    print("\n")
                    print("best val loss - epoch: %d, epoch_step: %d" % (epoch, epoch_step))
                    print("maximum_f_score", f_score)
                    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-p", "--predict", action="store_true", help="predict test using the best model")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("-n",'--using_negative', default=False, action='store_true', 
                    help='Use negative evidence samples in training')
    args = parser.parse_args()

    if args.predict:
        predict_test(args)
    else:
        run(args)

# nohup python -u dpr/main_update.py >train.out 2>&1 &
# nohup python -u dpr/main_update.py --model_pt 23-04-21 >train.out 2>&1 &
# nohup python -u dpr/main_update.py -p --model_pt dpr >test_dpr.out 2>&1 &

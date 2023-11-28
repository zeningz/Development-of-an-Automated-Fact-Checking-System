from torch.utils.data import Dataset
import json
import random
import nltk
from nltk.corpus import stopwords
import string

# Define the punctuation marks to exclude
exclude_punctuation = string.punctuation.replace(':', '').replace('°', '')
# Download the stop words corpus (run this once)


def process(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation marks
    text = ''.join(ch for ch in text if ch not in exclude_punctuation)

    # Tokenize the text into words
    words = text.split()


    # Join the words back into a single string
    processed_text = ' '.join(words)

    return processed_text



def to_cuda(batch):
    for n in batch.keys():
        if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
            batch[n] = batch[n].cuda()



class TrainDataset(Dataset):
    def __init__(self, mode, tok, evidence_samples, max_length=512, using_negative=True):
        self.max_length = max_length
        if using_negative:
            f = open("data/train-claims-with-negatives_combine.json", "r")
        else:
            f = open("data/{}-claims.json".format(mode), "r")

        self.dataset = json.load(f)
        f.close()
        self.using_negative = using_negative
        f = open("data/evidence.json", "r")
        f = open("temp_data/reduced-evidences_combined.json", "r")

        self.evidences = json.load(f)
        f.close()

        self.tokenizer = tok
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode
        self.evidence_samples = evidence_samples
        self.evidence_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):

        data = self.dataset[self.claim_ids[idx]]
        processed_query = process(data["claim_text"])
        evidences = []
        for evidence_id in data["evidences"]:
            evidences.append(evidence_id)
        if self.using_negative:
            negative_evidences = data["negative_evidences"]
            return [processed_query, evidences, negative_evidences]
        else:
            return [processed_query, evidences]

    def collate_fn(self, batch):
        queries = []
        evidences = []
        labels = []
        if self.using_negative:
            negative_evidences = []
            for query, evidence, negative_evidence in batch:
                queries.append(query)
                evidences.extend(evidence)
                negative_evidences.extend(negative_evidence)
                labels.append(len(evidence))
            negative_evidences2 = [x for x in negative_evidences if x not in evidences]
            evidences.extend(negative_evidences2)
        else:
            for query, evidence in batch:
                queries.append(query)
                evidences.extend(evidence)
                labels.append(len(evidence))
        cnt = len(evidences)
        if cnt > self.evidence_samples:
            evidences = evidences[:self.evidence_samples]
        evidences_text = [process(self.evidences[evidence_id]) for evidence_id in evidences]
        while cnt < self.evidence_samples:
            evidence_id = random.choice(self.evidence_ids)
            while evidence_id in evidences:
                evidence_id = random.choice(self.evidence_ids)
            evidences.append(evidence_id)
            evidences_text.append(process(self.evidences[evidence_id]))
            cnt += 1

        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        evidence_text = self.tokenizer(
            evidences_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["evidence_input_ids"] = evidence_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask
        batch_encoding["evidence_attention_mask"] = evidence_text.attention_mask
        batch_encoding["labels"] = labels
        return batch_encoding


class EvidenceDataset(Dataset):
    def __init__(self, tok, max_length=512):
        self.max_length = max_length


        f = open("data/evidence.json", "r")

        self.evidences = json.load(f)
        f.close()

        self.tokenizer = tok
        self.evidences_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.evidences_ids)

    def __getitem__(self, idx):
        evidences_id = self.evidences_ids[idx]
        evidence = self.evidences[evidences_id]
        return [evidences_id, evidence]

    def collate_fn(self, batch):
        evidences_ids = []
        evidences = []

        for evidences_id, evidence in batch:
            evidences_ids.append(evidences_id)
            evidences.append(process(evidence))

        evidences_text = self.tokenizer(
            evidences,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["evidence_input_ids"] = evidences_text.input_ids
        batch_encoding["evidence_attention_mask"] = evidences_text.attention_mask
        batch_encoding["evidences_ids"] = evidences_ids
        return batch_encoding


class ValDataset(Dataset):
    def __init__(self, mode, tok, max_length=512):
        self.max_length = max_length
        if mode != "test":
            f = open("data/{}-claims.json".format(mode), "r")

        else:
            f = open("data/test-claims-unlabelled.json", "r")
        self.dataset = json.load(f)
        f.close()

        self.tokenizer = tok
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        processed_text = process(data["claim_text"])
        return [processed_text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        queries = []
        datas = []
        evidences = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)
            if self.mode != "test":
                evidences.append(data["evidences"])
            claim_ids.append(claim_id)

        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask

        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids
        if self.mode != "test":
            batch_encoding["evidences"] = evidences
        return batch_encoding





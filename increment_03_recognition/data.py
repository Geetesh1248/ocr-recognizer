import unicodedata
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from common import image_read


class Vocabulary:
    def __init__(self, alphabet):
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet must contain unique characters")
        self.alphabet = alphabet
        self.index = {character: i+1 for i, character in enumerate(alphabet)}

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        missing = sorted(set(text)-set(self.alphabet))
        if missing:
            raise ValueError(f"Characters absent from vocabulary: {missing}")
        return [self.index[c] for c in text]


def line_tensor(image, height=32, max_width=1024):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = max(4, min(max_width, round(gray.shape[1]*height/gray.shape[0])))
    # Width is a multiple of four; the CNN downsamples horizontally by four.
    width = min(max_width, ((width+3)//4)*4)
    resized = cv2.resize(gray, (width, height))
    return torch.from_numpy(resized.copy()).float()[None]/127.5-1, width//4


def minimum_ctc_steps(tokens):
    return len(tokens)+sum(a == b for a, b in zip(tokens, tokens[1:]))


class RecognitionDataset(Dataset):
    def __init__(self, records, vocabulary, height=32, max_width=1024, augment=False):
        self.records, self.vocabulary = records, vocabulary
        self.height, self.max_width, self.augment = height, max_width, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if not isinstance(record.get("text"), str):
            raise ValueError("Recognition examples require text labels")
        label = unicodedata.normalize("NFC", record["text"])
        tokens = self.vocabulary.encode(label)
        image = image_read(record["image"])
        if self.augment:
            image = np.clip(image.astype(float)*np.random.uniform(.8, 1.2), 0, 255).astype(np.uint8)
            if np.random.rand() < .2:
                image = cv2.GaussianBlur(image, (3, 3), 0)
        tensor, steps = line_tensor(image, self.height, self.max_width)
        if steps < minimum_ctc_steps(tokens):
            raise ValueError(f"Crop too narrow for its CTC label: {record['image']}. Increase width or correct label/crop.")
        return tensor, torch.tensor(tokens, dtype=torch.long), steps, label


def collate(samples):
    width = max(t.shape[-1] for t, _, _, _ in samples)
    batch = torch.ones(len(samples), 1, samples[0][0].shape[1], width)
    for index, (image, _, _, _) in enumerate(samples):
        batch[index, :, :, :image.shape[-1]] = image
    return batch, torch.cat([s[1] for s in samples]), torch.tensor([s[2] for s in samples]), \
        torch.tensor([len(s[1]) for s in samples]), [s[3] for s in samples]

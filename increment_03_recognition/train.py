import torch
from torch.utils.data import DataLoader
from common import manifest, check_splits, seed_all, save_json, sha256, distance
from checkpoints import save
from .model import TextRecognizer
from .data import Vocabulary, RecognitionDataset, collate
from .decoder import greedy_ctc


def train(train_path, val_path, output, alphabet, epochs=30, batch_size=8, learning_rate=.001, seed=42, device=None, height=32, max_width=1024):
    if epochs < 1 or batch_size < 1 or height < 16 or height % 8 or max_width < 4 or max_width % 4 or learning_rate <= 0:
        raise ValueError("Invalid recognition training settings")
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    records, validation = manifest(train_path), manifest(val_path)
    check_splits(records, validation)
    seed_all(seed)
    vocabulary = Vocabulary(alphabet)
    # Validate every label before an expensive training run.
    for record in records+validation:
        vocabulary.encode(record["text"])
    model = TextRecognizer(len(alphabet)+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    loaders = [DataLoader(RecognitionDataset(records, vocabulary, height, max_width, True), batch_size=batch_size, shuffle=True, collate_fn=collate),
               DataLoader(RecognitionDataset(validation, vocabulary, height, max_width), batch_size=batch_size, collate_fn=collate)]
    best, history = float("inf"), []
    for epoch in range(1, epochs+1):
        losses, character_errors, characters = [], 0, 0
        for training, loader in zip([True, False], loaders):
            model.train(training)
            total, count = 0., 0
            for images, targets, lengths, target_lengths, labels in loader:
                with torch.set_grad_enabled(training):
                    logits = model(images.to(device), lengths)
                    loss = criterion(logits.log_softmax(-1), targets.to(device), lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite CTC loss; check crop widths and labels")
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                else:
                    probabilities = logits.detach().softmax(-1).cpu().numpy()
                    for index, label in enumerate(labels):
                        predicted, _ = greedy_ctc(probabilities[:int(lengths[index]), index], alphabet)
                        character_errors += distance(label, predicted)
                        characters += len(label)
                total += float(loss.detach())*len(images)
                count += len(images)
            losses.append(total/count)
        cer = character_errors/max(1, characters)
        history.append({"epoch": epoch, "train_loss": losses[0], "validation_loss": losses[1], "validation_cer": cer})
        print(history[-1], flush=True)
        if cer < best:
            best = cer
            save(output, {"format_version": 1, "kind": "recognizer", "epochs": epoch,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "alphabet": alphabet, "height": height, "max_width": max_width, "seed": seed,
                          "validation_cer": best, "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path)})
        save_json(str(output)+".history.json", history)
    return history

import torch
from checkpoints import load
from common import image_read
from .model import TextRecognizer
from .data import line_tensor
from .decoder import greedy_ctc, beam_search_ctc, CharNgramLM


class Recognizer:
    def __init__(self, checkpoint, decoder="greedy", beam_width=5, lm_weight=0.0, lm_corpus_dir="corpus"):
        payload = load(checkpoint, "recognizer")
        self.alphabet = payload["alphabet"]
        self.height, self.max_width = payload["height"], payload["max_width"]
        version = payload.get("model_version")
        if version is None:
            # Detect by state_dict keys: V1 has cnn.0.weight with 32 out_channels, V2 has 64
            sd = payload.get("state_dict", {})
            if "cnn.0.weight" in sd and sd["cnn.0.weight"].shape[0] == 32:
                version = 1
            else:
                version = 2
        from .model import build_recognizer
        self.model = build_recognizer(len(self.alphabet)+1, version=version)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.decoder = decoder
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.lm_corpus_dir = lm_corpus_dir
        self.lm = None
        if self.lm_weight > 0 and lm_corpus_dir:
            try:
                self.lm = CharNgramLM.from_corpus(lm_corpus_dir, self.alphabet)
            except Exception:
                self.lm = None

    @torch.inference_mode()
    def __call__(self, regions, decoder=None, beam_width=None, lm_weight=None):
        lines = []
        dec = decoder if decoder is not None else self.decoder
        bw = beam_width if beam_width is not None else self.beam_width
        lm_w = lm_weight if lm_weight is not None else self.lm_weight
        lm = self.lm
        if lm_w > 0 and lm is None and self.lm_corpus_dir:
            try:
                self.lm = CharNgramLM.from_corpus(self.lm_corpus_dir, self.alphabet)
                lm = self.lm
            except Exception:
                pass

        for region in regions:
            item = dict(region)
            try:
                image, steps = line_tensor(image_read(region["crop"]), self.height, self.max_width)
                probabilities = self.model(image[None], torch.tensor([steps])).softmax(-1)[:, 0].numpy()
                if dec == "beam" or bw > 1:
                    item["text"], item["confidence"] = beam_search_ctc(
                        probabilities, self.alphabet, beam_width=bw, lm_weight=lm_w, lm=lm
                    )
                else:
                    item["text"], item["confidence"] = greedy_ctc(probabilities, self.alphabet)
                item["status"] = "completed" if item["text"] else "empty"
            except Exception as error:
                item.update(text="", confidence=None, status="failed", error=f"{type(error).__name__}: {error}")
            lines.append(item)
        return lines

# Methods behind this implementation

The Python architectures, training loops and processing code are written in this project. We use established methods rather than claim to have invented a new OCR research architecture.

- [Differentiable Binarization paper](https://arxiv.org/abs/1911.08947): informs the segmentation-detector direction. Our current detector uses shrunk polygon targets and fixed-threshold post-processing; it does not implement the paper's learned threshold branch and should not be called a full DBNet reproduction.
- [CRNN paper](https://arxiv.org/abs/1507.05717): informs the CNN + recurrent sequence recognizer. We implement a compact variant using our own layer definitions, vocabulary and greedy decoder.
- [PyTorch CTC loss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html): the framework's primitive CTC loss supplies gradient calculation; our data alignment checks, architecture, training loop and decoding are implemented here.

OpenCV supplies elementary transforms, contour/morphology operations and interpolation. NumPy supplies numerical operations. PDFium supplies PDF decoding. Those libraries do not perform text recognition for this project. PyClipper supplies polygon offsets/intersections, not an OCR detector.

No pretrained weights, ready-made OCR service or generative model are used. Future datasets and trained assets still need recorded source/usage terms.

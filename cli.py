"""Train our models or run our own five processing increments."""
import argparse
from pathlib import Path
import sys
from common import ROOT, read_json, save_json


def build_parser():
    parser = argparse.ArgumentParser(description="Self-made OCR: algorithms and trainable networks implemented in this project")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("run")
    item.add_argument("input")
    item.add_argument("--until", type=int, choices=range(1, 6), default=3)
    item.add_argument("--output", required=True)
    item.add_argument("--pages")
    item.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    for name, epochs, batch in [("train-detector", 20, 2), ("train-recognizer", 30, 8), ("train-orientation", 15, 8), ("train-line-orientation", 15, 8)]:
        item = sub.add_parser(name)
        item.add_argument("--train", required=True)
        item.add_argument("--val", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--epochs", type=int, default=epochs)
        item.add_argument("--batch-size", type=int, default=batch)
        item.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        item.add_argument("--seed", type=int, default=42)
        if name == "train-detector":
            item.add_argument("--size", type=int, default=512)
    item = sub.add_parser("evaluate-text")
    item.add_argument("result")
    item.add_argument("labels")
    item.add_argument("--output", required=True)
    item = sub.add_parser("evaluate-detector")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--labels", required=True)
    item.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if args.command == "run":
        from pipeline import run
        result = run(args.input, args.output, args.until, config,
                     [int(n) for n in args.pages.split(",")] if args.pages else None, args.rotation)
        print(result["status"], Path(args.output).resolve())
        return 0 if result["status"] == "completed" else 2
    if args.command.startswith("train-"):
        import torch
        torch.set_num_threads(config["cpu_threads"])
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable in this environment; select CPU or prepare a GPU training environment")
            device = "cuda"
        else:
            device = "cpu"
        options = dict(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=device)
        if args.command == "train-detector":
            from increment_02_detection.train import train
            options["size"] = args.size
        elif args.command == "train-recognizer":
            from increment_03_recognition.train import train
            options.update(alphabet=config["recognizer"]["alphabet"], height=config["recognizer"]["height"], max_width=config["recognizer"]["max_width"])
        else:
            from increment_01_preparation.orientation import train
            if args.command == "train-line-orientation":
                options["kind"] = "line_orientation"
        train(args.train, args.val, args.output, **options)
        return 0
    from evaluate import text_score, detection_score
    if args.command == 'evaluate-detector':
        import torch
        torch.set_num_threads(config['cpu_threads'])
    result = text_score(read_json(args.result), read_json(args.labels)) if args.command == "evaluate-text" else \
        detection_score(args.checkpoint, args.labels, config["detector"])
    save_json(args.output, result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)

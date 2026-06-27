"""Pseudo-label and fine-tune a lightweight general-object detector.

The external image folder supplied for this experiment contains images only.
Consequently, this script uses a pretrained YOLO model to create conservative
pseudo-labels, freezes the backbone, fine-tunes briefly, and reports consistency
against a held-out pseudo-labelled split. These are not ground-truth accuracy
metrics and must never be presented as manually annotated evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder containing unlabelled images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/general_object_training"),
        help="Experiment output directory.",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained Ultralytics model or path.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.35, help="Pseudo-label confidence threshold.")
    parser.add_argument("--seed", type=int, default=435)
    parser.add_argument(
        "--trained-weights",
        type=Path,
        default=None,
        help="Evaluate an existing fine-tuned checkpoint instead of training again.",
    )
    return parser.parse_args()


def discover_images(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Image folder not found: {source}")
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) < 20:
        raise ValueError(f"Expected at least 20 images, found {len(images)} in {source}")
    unreadable = [str(path) for path in images if cv2.imread(str(path)) is None]
    if unreadable:
        raise ValueError(f"Unreadable images found: {unreadable[:5]}")
    return images


def reset_dataset(output: Path) -> Path:
    dataset = output / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    return dataset


def split_images(images: list[Path], seed: int, validation_fraction: float = 0.2) -> dict[str, list[Path]]:
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    return {"train": shuffled[validation_count:], "val": shuffled[:validation_count]}


def pseudo_label_dataset(
    model: YOLO,
    splits: dict[str, list[Path]],
    dataset: Path,
    confidence: float,
    image_size: int,
) -> tuple[Counter[int], dict[str, int], list[float]]:
    class_counts: Counter[int] = Counter()
    labelled_images = {"train": 0, "val": 0}
    confidences: list[float] = []
    for split, paths in splits.items():
        results = model.predict(
            source=[str(path) for path in paths],
            conf=confidence,
            imgsz=image_size,
            device="cpu",
            stream=True,
            verbose=False,
        )
        for source_path, result in zip(paths, results):
            target_image = dataset / "images" / split / source_path.name
            target_label = dataset / "labels" / split / f"{source_path.stem}.txt"
            shutil.copy2(source_path, target_image)
            label_lines: list[str] = []
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy().astype(float)
                boxes = result.boxes.xywhn.cpu().numpy().astype(float)
                for class_id, score, (x, y, width, height) in zip(classes, scores, boxes):
                    label_lines.append(f"{class_id} {x:.7f} {y:.7f} {width:.7f} {height:.7f}")
                    class_counts[int(class_id)] += 1
                    confidences.append(float(score))
            if label_lines:
                labelled_images[split] += 1
            target_label.write_text("\n".join(label_lines), encoding="utf-8")
    return class_counts, labelled_images, confidences


def write_dataset_yaml(dataset: Path, names: dict[int, str]) -> Path:
    config = {
        "path": str(dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": [names[index] for index in range(len(names))],
    }
    path = dataset / "dataset.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def validation_metrics(model: YOLO, data_yaml: Path, image_size: int, batch: int) -> dict:
    result = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=image_size,
        batch=batch,
        device="cpu",
        workers=0,
        verbose=False,
        plots=False,
    )
    metrics = {
        "map50_95": float(result.box.map),
        "map50": float(result.box.map50),
        "map75": float(result.box.map75),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
    }
    class_ids = [int(class_id) for class_id in result.box.ap_class_index]
    if 0 in class_ids:
        precision, recall, map50, map50_95 = result.box.class_result(class_ids.index(0))
        metrics["person"] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    return metrics


def benchmark(model: YOLO, images: list[Path], image_size: int) -> dict[str, float]:
    sample = images[: min(24, len(images))]
    model.predict(str(sample[0]), imgsz=image_size, device="cpu", verbose=False)
    start = time.perf_counter()
    list(
        model.predict(
            source=[str(path) for path in sample],
            imgsz=image_size,
            device="cpu",
            stream=True,
            verbose=False,
        )
    )
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000 / len(sample)
    return {"images": len(sample), "mean_latency_ms": latency_ms, "fps": 1000 / latency_ms}


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    images = discover_images(args.source)
    splits = split_images(images, args.seed)
    dataset = reset_dataset(args.output)

    print(f"Found {len(images)} readable images: {len(splits['train'])} train / {len(splits['val'])} validation")
    base_model = YOLO(args.model)
    class_counts, labelled_images, confidences = pseudo_label_dataset(
        base_model,
        splits,
        dataset,
        args.confidence,
        args.imgsz,
    )
    data_yaml = write_dataset_yaml(dataset, base_model.names)
    named_counts = {base_model.names[class_id]: count for class_id, count in class_counts.most_common()}
    print("Pseudo-label class counts:", json.dumps(named_counts, indent=2))

    baseline_metrics = validation_metrics(base_model, data_yaml, args.imgsz, args.batch)
    baseline_speed = benchmark(base_model, splits["val"], args.imgsz)

    if args.trained_weights is None:
        run_project = (args.output / "runs").resolve()
        trained = YOLO(args.model)
        trained.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device="cpu",
            workers=0,
            seed=args.seed,
            deterministic=True,
            freeze=10,
            patience=3,
            project=str(run_project),
            name="fine_tune",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        best_source = Path(trained.trainer.best)
    else:
        best_source = args.trained_weights.resolve()
    if not best_source.exists():
        raise FileNotFoundError(f"Training completed without a best checkpoint: {best_source}")
    results_csv = best_source.parent.parent / "results.csv"
    completed_epochs = 0
    best_checkpoint_epoch = 0
    if results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed_epochs = len(rows)
        if rows:
            metric_key = next(key for key in rows[0] if "mAP50-95" in key)
            best_checkpoint_epoch = int(float(max(rows, key=lambda row: float(row[metric_key]))["epoch"]))
    final_weights = args.output / "general_objects_pseudo_finetuned.pt"
    shutil.copy2(best_source, final_weights)
    fine_model = YOLO(str(final_weights))
    fine_metrics = validation_metrics(fine_model, data_yaml, args.imgsz, args.batch)
    fine_speed = benchmark(fine_model, splits["val"], args.imgsz)

    summary = {
        "warning": "All labels were generated by the pretrained baseline. Metrics measure pseudo-label consistency, not ground-truth accuracy.",
        "source": str(args.source.resolve()),
        "images": {"total": len(images), "train": len(splits["train"]), "validation": len(splits["val"])},
        "pseudo_labels": {
            "confidence_threshold": args.confidence,
            "labelled_images": labelled_images,
            "total_boxes": int(sum(class_counts.values())),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "class_counts": named_counts,
        },
        "training": {
            "base_model": args.model,
            "requested_epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "best_checkpoint_epoch": best_checkpoint_epoch,
            "early_stopping_used": 0 < completed_epochs < args.epochs,
            "image_size": args.imgsz,
            "batch": args.batch,
            "backbone_frozen_through_layer": 10,
            "weights": str(final_weights.resolve()),
        },
        "baseline": {"validation": baseline_metrics, "speed": baseline_speed},
        "fine_tuned": {"validation": fine_metrics, "speed": fine_speed},
    }
    summary_path = args.output / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved weights: {final_weights}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

"""Launch the CSCI435 Gradio application using the already-trained classifier.

This launcher does not execute the notebook or retrain any model. It imports the
tested function/class definitions from the notebook source, loads the committed
joblib model, and starts the same Gradio interface used by the notebook.
"""

from __future__ import annotations

import ast
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "CSCI435_Project.ipynb"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "custom_marker_classifier.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

LABELS = ["stop", "warning", "safe", "other"]
TARGET_LABELS = {"stop", "warning", "safe"}
MAX_WIDTH = 640
MIN_CANDIDATE_AREA = 300


def _notebook_code_containing(marker: str) -> str:
    """Return the source of the notebook code cell containing ``marker``."""
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Required notebook not found: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one notebook code cell containing {marker!r}; found {len(matches)}.")
    return matches[0]


def _import_notebook_definitions(
    marker: str,
    names: set[str],
    assignment_names: set[str] | None = None,
) -> None:
    """Execute only selected definitions from a notebook cell, never its workflow statements."""
    assignment_names = assignment_names or set()
    tree = ast.parse(_notebook_code_containing(marker), filename=str(NOTEBOOK_PATH))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & assignment_names:
                selected.append(node)
    missing = names - {
        node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if missing:
        raise RuntimeError(f"Notebook definitions missing: {', '.join(sorted(missing))}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), globals())


def _load_runtime() -> None:
    """Load pipeline definitions and the existing trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run CSCI435_Project.ipynb once to create it."
        )

    _import_notebook_definitions(
        "def canonicalise_marker(",
        {"canonicalise_marker", "extract_features"},
        assignment_names={"HOG"},
    )
    _import_notebook_definitions(
        "class VisionSystem:",
        {
            "MarkerDetection",
            "FrameResult",
            "validate_and_resize",
            "enhance_image",
            "adaptive_canny",
            "create_edge_view",
            "colour_candidate_mask",
            "detect_markers",
            "CentroidTracker",
            "MotionDetector",
            "horizontal_position",
            "build_guidance",
            "VisionSystem",
        },
    )
    _import_notebook_definitions(
        "def process_video(",
        {"open_video_writer", "process_video"},
    )

    globals()["classifier"] = joblib.load(MODEL_PATH)
    globals()["SYSTEM"] = VisionSystem(classifier)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    dataset_metrics = metrics.get("dataset", {})
    globals()["selected_model_name"] = metrics.get("selected_model", "Saved RBF SVM")
    globals()["test_accuracy"] = float(metrics.get("held_out_accuracy", 0.0))
    globals()["images"] = [None] * int(dataset_metrics.get("total_images", 560))
    globals()["sample_scene_path"] = ARTIFACTS / "sample_scene.png"
    globals()["sample_input_video"] = ARTIFACTS / "sample_input_video.mp4"

    _import_notebook_definitions(
        "def build_gradio_app():",
        {"bgr_to_rgb", "ui_process_image", "ui_process_video", "build_gradio_app"},
    )


_load_runtime()
APP = build_gradio_app()


if __name__ == "__main__":
    if os.environ.get("CSCI435_QA") == "1":
        print(f"QA passed: loaded {selected_model_name} from {MODEL_PATH.relative_to(ROOT)} and built Gradio UI.")
    else:
        port = int(os.environ.get("CSCI435_PORT", "7860"))
        print(f"Loaded existing model: {MODEL_PATH.relative_to(ROOT)}")
        print(f"Opening local application at http://127.0.0.1:{port}")
        APP.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=True,
            show_error=True,
        )

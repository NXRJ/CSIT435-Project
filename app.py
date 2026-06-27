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
GENERAL_MODEL_PATH = ARTIFACTS / "general_object_training" / "general_objects_pseudo_finetuned.pt"

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
            "colour_candidate_masks",
            "colour_candidate_mask",
            "shape_colour_hint",
            "box_iou",
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


def _add_general_object_tab(app: gr.Blocks) -> None:
    """Add an opt-in people/general-object detector without changing the marker pipeline."""
    if not GENERAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Experimental general-object weights not found: {GENERAL_MODEL_PATH}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "General-object mode requires Ultralytics. Install requirements-general.txt first."
        ) from exc

    general_model = YOLO(str(GENERAL_MODEL_PATH))

    def process_general_image(image_rgb: np.ndarray | None):
        if image_rgb is None:
            raise gr.Error("Upload an image or capture a webcam frame first.")
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        marker_result = SYSTEM.process_frame(frame_bgr)
        start = time.perf_counter()
        prediction = general_model.predict(
            source=frame_bgr,
            conf=0.35,
            imgsz=512,
            device="cpu",
            verbose=False,
        )[0]
        general_latency_ms = (time.perf_counter() - start) * 1000
        annotated = marker_result.annotated.copy()
        counts: dict[str, int] = {}
        if prediction.boxes is not None:
            boxes = prediction.boxes.xyxy.cpu().numpy().astype(int)
            classes = prediction.boxes.cls.cpu().numpy().astype(int)
            confidences = prediction.boxes.conf.cpu().numpy().astype(float)
            for (x1, y1, x2, y2), class_id, confidence in zip(boxes, classes, confidences):
                label = str(prediction.names[int(class_id)])
                counts[label] = counts.get(label, 0) + 1
                colour = (210, 60, 220) if label == "person" else (230, 170, 30)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    annotated,
                    f"{label.upper()} {confidence:.2f}",
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
        object_summary = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
        if not object_summary:
            object_summary = "no general objects above 35% confidence"
        total_latency = marker_result.latency_ms + general_latency_ms
        guidance = f"{marker_result.summary} Experimental general objects: {object_summary}."
        metrics = (
            f"**Combined latency:** {total_latency:.1f} ms  |  **Estimated throughput:** "
            f"{1000 / max(total_latency, 1e-6):.1f} FPS  |  **General detections:** {sum(counts.values())}"
        )
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), guidance, metrics

    with app:
        with gr.Tab("General Objects - Experimental"):
            gr.Markdown(
                "This optional branch combines the original accessibility-marker SVM with a pseudo-label-fine-tuned "
                "YOLO11n detector for people and everyday COCO objects. Its pseudo-label metrics are not ground-truth accuracy."
            )
            with gr.Row():
                general_input = gr.Image(label="Input", sources=["upload", "webcam"], type="numpy")
                general_output = gr.Image(label="Combined marker and general-object result", type="numpy")
            general_button = gr.Button("Analyse markers and general objects", variant="primary")
            general_guidance = gr.Textbox(label="Combined scene guidance", lines=3)
            general_metrics = gr.Markdown()
            general_button.click(
                process_general_image,
                inputs=[general_input],
                outputs=[general_output, general_guidance, general_metrics],
            )


if os.environ.get("CSCI435_ENABLE_GENERAL") == "1":
    _add_general_object_tab(APP)


if __name__ == "__main__":
    if os.environ.get("CSCI435_QA") == "1":
        mode = "marker + experimental general-object" if os.environ.get("CSCI435_ENABLE_GENERAL") == "1" else "marker"
        print(f"QA passed: loaded {selected_model_name} and built {mode} Gradio UI.")
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

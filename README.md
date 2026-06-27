# CSCI435 Accessibility Scene Hazard Assistant

This repository contains a notebook-first, locally deployable computer-vision system for **CSCI435 - Computer Vision Algorithms and Systems**. The application supports image/webcam input and uploaded video, integrates six vision capabilities in one workflow, trains a custom recognition model, reports quantitative results, and launches a Gradio web interface.

## Main deliverables

- `CSCI435_Project.ipynb` - self-contained implementation, training, evaluation, benchmarking, sample generation, and web UI.
- `output/pdf/CSCI435_Project_Report.pdf` - submission-ready project report.
- `output/report/CSCI435_Project_Report.docx` - editable report source.
- `artifacts/CSCI435_Demonstration_Video.mp4` - 2 minute 5 second backup demonstration video generated from actual pipeline outputs.
- `DEFENCE_GUIDE.md` - timed live-demonstration plan and likely questions.
- `RUBRIC_COMPLIANCE.md` - requirement-to-evidence traceability matrix.

## Vision capabilities

The integrated pipeline performs:

1. **Image enhancement** using gray-world colour balancing and CLAHE.
2. **Edge detection** using adaptive Canny thresholds and morphology.
3. **Object detection** using colour segmentation and contour proposals.
4. **Object recognition** using a custom-trained SVM classifier for accessibility markers (`stop`, `warning`, `safe`, and `other`).
5. **Moving-object detection** using MOG2 background modelling and morphology.
6. **Object tracking** using persistent centroid track IDs.

The user receives bounding boxes, labels, motion IDs, text guidance, latency, and FPS.

## Quick start (Windows PowerShell)

From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab CSCI435_Project.ipynb
```

In Jupyter, choose **Run > Run All Cells**. The notebook trains the model, evaluates it, creates sample artifacts, and launches the local Gradio interface at the end. If the browser does not open automatically, use the local URL printed by Gradio.

## Fast standalone Gradio launcher

After the model artifact exists, launch the interface without opening the notebook or retraining:

```powershell
cd "C:\Users\PC\Downloads\CSIT435 Project\CSIT435 Project"
.\.venv\Scripts\python.exe app.py
```

The launcher loads `artifacts/custom_marker_classifier.joblib`, constructs the tested pipeline, and opens `http://127.0.0.1:7860`. Stop it with `Ctrl+C`. If port 7860 is occupied, choose another port:

```powershell
$env:CSCI435_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

macOS/Linux equivalents:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab CSCI435_Project.ipynb
```

## How to demonstrate the app

1. Run all notebook cells and wait for the validation summary.
2. Open the **Image / Webcam** tab and use the generated sample or a camera image containing red, yellow, or green marker-like objects.
3. Show the enhanced, edge, and annotated views plus the natural-language guidance.
4. Open **Video** and upload `artifacts/sample_input_video.mp4`.
5. Show motion boxes, track IDs, and measured FPS/latency.
6. Use `artifacts/CSCI435_Demonstration_Video.mp4` as the required 2-3 minute submitted backup video.

## Reproducibility

- Python 3.12 is recommended.
- Random seeds are fixed in the notebook.
- Training data is generated deterministically and stored under `artifacts/custom_dataset/` when the notebook runs.
- Model selection compares SVM, random forest, and k-nearest neighbours on the same stratified split.
- Metrics are written to `artifacts/metrics.json` and CSV files.
- The notebook includes executable assertions for blank inputs, dim lighting, rotation, detection, and video processing.

## Verified results

The committed notebook was executed from top to bottom with no cell errors. Its saved results report:

| Measure | Result |
|---|---:|
| Custom training images | 560 (140 per class) |
| Held-out synthetic crop accuracy | 100.0% |
| Mean end-to-end detection precision | 92.5% |
| Mean end-to-end detection recall | 88.1% |
| Mean end-to-end detection F1 | 89.8% |
| Still-frame mean latency / throughput | 13.47 ms / 74.23 FPS |
| Processed-video mean latency / throughput | 13.27 ms/frame / 75.36 FPS |

The 100% crop result applies only to the constrained synthetic held-out set; it is not a claim of real-world accuracy. Timing is CPU- and load-dependent, but the saved run exceeds the rubric's 10 FPS target by a wide margin.

## Important academic note

The included custom dataset is **synthetic and programmatically generated** so that the submission is self-contained and reproducible. It does not represent all real-world signs or accessibility hazards, and the report states this limitation explicitly. Before submission, the team must verify that the proposed contribution table in the report and `DEFENCE_GUIDE.md` matches the work actually completed.

## Repository submission

This folder should be placed under Git version control and pushed to a private or public GitHub/GitLab repository accessible to the lecturer:

```powershell
git init
git add .
git commit -m "Complete CSCI435 integrated vision system"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Do not commit virtual environments or downloaded caches. The included `.gitignore` excludes them.

## Troubleshooting

- **No camera access:** use image upload or video upload; both satisfy the two-modality requirement together.
- **Port already in use:** change the `server_port` in the final notebook cell or stop the previous Gradio process.
- **Video codec issue:** use the generated MP4; the notebook includes an AVI fallback internally.
- **Slow machine:** reduce `MAX_WIDTH` in the configuration cell from 640 to 480.
- **No coloured object detected:** use saturated red, yellow, or green markers and keep them larger than approximately 20 x 20 pixels.

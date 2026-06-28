# CSCI435 Accessibility Scene Hazard Assistant

This is our CSCI435 computer vision project. We built a local Gradio app that analyses images, webcam captures, and uploaded videos. It recognises our custom stop, warning, and safe-route markers, shows an enhanced and edge-detected view, tracks movement in videos, and reports the processing speed.

The core project is inside [CSCI435_Project.ipynb](CSCI435_Project.ipynb). We also included [app.py](app.py) so we can launch the finished app with the saved model without rerunning the training notebook.

## What the project does

For each image or video frame, our pipeline:

1. validates the input and resizes large frames;
2. balances the colour and improves local contrast using CLAHE;
3. calculates adaptive Canny edges;
4. finds red, yellow, and green candidate regions in HSV colour space;
5. recognises stop, warning, and safe markers with our trained RBF SVM;
6. uses MOG2 and centroid tracking for moving regions in videos; and
7. returns bounding boxes, labels, confidence, scene guidance, latency, and FPS.

We trained the marker classifier on 560 generated images: 140 each for stop, warning, safe, and other. The generator changes rotation, brightness, blur, noise, scale, and occlusion. The other class gives the model negative examples so it does not accept every coloured shape as a marker.

Each candidate is represented using HOG shape features, HSV colour histograms, Hu moments, and basic brightness statistics. We compared an RBF SVM, random forest, and k-nearest neighbours on the same stratified split. All three performed well on the synthetic test crops, and we selected the SVM because it had the fastest practical inference among the top models.

## Main files

- CSCI435_Project.ipynb — training, evaluation, testing, visual results, and the Gradio interface.
- app.py — loads the existing model and starts the app without retraining.
- launch_app.bat — double-click Windows launcher for the finished app.
- artifacts/custom_marker_classifier.joblib — our saved custom marker model.
- artifacts/metrics.json — the final saved metrics.
- artifacts/sample_scene.png — image example used by the interface.
- artifacts/sample_input_video.mp4 — video example used by the interface.
- artifacts/sample_scene_dim.png and artifacts/sample_scene_noisy.png — extra examples with different marker combinations, including the green accessibility tick.
- artifacts/sample_input_video_dim.mp4 and artifacts/sample_input_video_blurred.mp4 — extra motion examples with independent diagonal and curved paths.
- artifacts/CSCI435_Demonstration_Video.mp4 — our 2 minute 5 second demonstration video.
- output/report/CSCI435_Project_Report.docx — editable report.
- output/pdf/CSCI435_Project_Report.pdf — final PDF report.
- scripts/train_general_detector.py — optional general-object training experiment.

## Set up the project on Windows

Open PowerShell in the project folder and run:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

## Run the finished app

The trained SVM weights are already included. On Windows, we can double-click launch_app.bat. On its first run, it creates .venv and installs the packages from requirements.txt; later launches start the app directly.

Processed videos are converted to H.264 before Gradio displays them. The project installs its own FFmpeg binary through imageio-ffmpeg, so another computer does not need a separate system FFmpeg installation.

The PowerShell command is:

~~~powershell
.\.venv\Scripts\python.exe app.py
~~~

Then open http://127.0.0.1:7860 if the browser does not open automatically. Press Ctrl+C in PowerShell to stop the server.

If port 7860 is already being used:

~~~powershell
$env:CSCI435_PORT="7861"
.\.venv\Scripts\python.exe app.py
~~~

## Run the notebook and train again

To reproduce the complete project:

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab CSCI435_Project.ipynb
~~~

Open the notebook and choose **Run > Run All Cells**. It will regenerate artifacts/custom_dataset/, train and compare the classifiers, save the best model, run the robustness tests, recreate the sample outputs, and build the Gradio interface.

We kept all 560 generated images in artifacts/custom_dataset/ so we can open the class folders and explain the training data during the demonstration. The notebook can still regenerate the same dataset because the random seed is fixed.

## Saved results

| Measurement | Result |
|---|---:|
| Generated training images | 560 |
| Held-out synthetic crop accuracy | 100.0% |
| Mean end-to-end precision | 93.0% |
| Mean end-to-end recall | 94.6% |
| Mean end-to-end F1 | 93.1% |
| Still-image processing | 19.96 ms / 50.09 FPS |
| Video processing | 16.75 ms/frame / 59.72 FPS |
| Real stop-sign test | 98.0% confidence |
| Real safe-marker test | 94.6% confidence |

The 100% crop result only applies to our controlled synthetic test set. It should not be read as 100% real-world accuracy. Our more useful end-to-end result is the 93.1% mean F1 across bright, dim, rotated, blurred, noisy, occluded, and cluttered generated scenes.

## Optional people and general-object detector

We also tested an optional YOLO11n branch on 240 unlabelled real-world images. Since the folder did not contain human-made bounding boxes, we used the pretrained model to create pseudo-labels and then fine-tuned on those labels.

This experiment is separate from our main SVM model. Its results measure agreement with generated pseudo-labels, not independent ground-truth accuracy. We kept it optional because it can detect people and common COCO objects, but it is less reliable for our custom warning and safe markers.

To enable the extra tab:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-general.txt
$env:CSCI435_ENABLE_GENERAL="1"
.\.venv\Scripts\python.exe app.py
~~~

The saved experimental weights are in artifacts/general_object_training/general_objects_pseudo_finetuned.pt. The experiment summary and combined benchmark are in the same folder.

YOLO11n and the Ultralytics package are third-party components. We did not create the pretrained COCO weights. Ultralytics publishes the project under AGPL-3.0 and also offers an enterprise licence, so anyone redistributing the optional branch should check the current terms:

- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics)

## Limitations

- The main classifier was trained on generated markers, so there is still a synthetic-to-real gap.
- The candidate stage depends on red, yellow, and green colour ranges.
- The motion detector finds changing regions but does not identify what they are.
- Centroid tracking can switch IDs when objects cross or move too quickly.
- The project does not estimate distance or depth.
- This is a university proof of concept and should not be treated as a real safety device.

## Repository

The project is stored at [NXRJ/CSIT435-Project](https://github.com/NXRJ/CSIT435-Project).

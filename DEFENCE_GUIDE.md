# CSCI435 Live Defence Guide

## Seven-to-eight-minute demonstration

| Time | Speaker/workstream | What to show and say |
|---|---|---|
| 0:00-0:45 | Introduction | State the accessibility problem and user story. Explain that coloured markers are a safe, reproducible proof-of-concept for scene guidance. |
| 0:45-1:35 | Architecture | Show the report architecture diagram: input, preprocessing, parallel vision modules, fusion, UI. Name all six capabilities. |
| 1:35-2:35 | Custom training | Show dataset montage, three candidate models, selected SVM, confusion matrix, and robustness table. Emphasise deterministic custom data and honest limitations. |
| 2:35-4:10 | Image/webcam demo | Process the generated clean scene and then a dim/rotated scene. Point out enhancement, edges, recognition boxes, confidence, text guidance, latency, and graceful empty-scene output. |
| 4:10-5:35 | Video demo | Upload the generated sample video. Point out MOG2 moving regions and persistent centroid track IDs. Confirm the measured FPS against the 10 FPS target. |
| 5:35-6:25 | Results and critical analysis | Show accuracy, detection precision/recall/F1, FPS, and latency. Explain failures: synthetic-to-real domain gap, colour dependence, occlusion, and uncalibrated distance. |
| 6:25-7:15 | Reliability and fallback | Mention input validation, codec fallback, generated test assets, pre-run notebook, and the 2:05 backup video. |
| 7:15-7:45 | Conclusion | Summarise the integrated workflow and future work: real labelled data, audio/haptics, calibration, and mobile deployment. |

## Proposed responsibility split - verify before submission

This is a balanced preparation plan, not a claim about completed historical work.

| Member | Student ID | Coding lead | Documentation lead | Defence lead |
|---|---:|---|---|---|
| Mehdi Leghmizi | 8528834 | UI integration and input handling | User story and requirements | Opening and UI demo |
| Neeraj Santosh | 8329345 | Custom dataset and model training | Training methodology | Model-selection explanation |
| Muhammad Soban | 8555588 | Enhancement, edges, robustness tests | Experiment methodology | Image robustness demo |
| Zachary Bracke | 8947405 | Motion detection, tracking, performance | Results and critical analysis | Video and performance demo |
| Mostafa Shalash | 7391493 | System integration and QA | Architecture, conclusion, references | Architecture and closing |

Every member should make at least one genuine code/documentation contribution and be able to explain one algorithm outside their assigned lead area.

## Likely questions

**Why SVM?** It gave the best held-out balance of accuracy and latency among the evaluated lightweight models. The feature space combines HOG/shape and colour evidence, which suits the deliberately constrained marker classes.

**Is the dataset real?** No. It is custom synthetic data generated with rotation, brightness, blur, noise, and occlusion. This makes the notebook reproducible but creates a domain gap; real labelled images are the next step.

**What makes the tasks integrated?** One frame is enhanced, analysed for edges, searched for object candidates, classified, checked for motion, tracked, and fused into one annotated result and one guidance sentence.

**Why MOG2?** It is an adaptive background model that is fast enough for CPU video and handles gradual background changes better than simple frame differencing.

**How is robustness assessed?** Condition-specific held-out sets measure recognition, and generated scenes measure class-aware detection under dim light, rotation, blur, noise, occlusion, clutter, and blank inputs.

**What were the main measured results?** On the saved run, end-to-end scene detection reached 92.5% mean precision, 88.1% mean recall, and 89.8% mean F1. Still-frame processing averaged 13.47 ms or 74.23 FPS, while processed video averaged 13.27 ms/frame or 75.36 FPS. The 100% crop accuracy belongs only to the constrained synthetic held-out set.

**What happens when nothing is detected?** The UI returns a clear no-marker message rather than failing, while edges and performance information remain available.

**What are the biggest limitations?** The colour-based proposal stage can fail under extreme illumination or colour cast; synthetic signs do not cover natural objects; motion boxes are not semantic; and no metric depth is estimated.

**How would this become a real assistive product?** Collect consented real data, train a modern detector, quantify false negatives, add calibrated depth, test with target users, introduce audio/haptic feedback, and complete privacy/safety validation.

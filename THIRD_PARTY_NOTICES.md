# Third-party notices

The core accessibility-marker pipeline is original project code built with cited public libraries including OpenCV, scikit-learn, Gradio, NumPy, pandas and Matplotlib.

The optional people/general-object experiment uses Ultralytics YOLO11n and a COCO-pretrained `yolo11n.pt` checkpoint. The derived experimental checkpoint is stored at `artifacts/general_object_training/general_objects_pseudo_finetuned.pt` and is not the project's rubric-critical custom SVM.

- Software/model: Ultralytics YOLO11
- Authors cited by Ultralytics: Glenn Jocher and Jing Qiu
- Project: https://github.com/ultralytics/ultralytics
- Documentation: https://docs.ultralytics.com/models/yolo11/
- Open-source license identified by Ultralytics: AGPL-3.0

Users redistributing or deploying the optional YOLO component are responsible for following the applicable Ultralytics model and software license terms. The academic project does not claim authorship of the YOLO architecture or COCO-pretrained weights.

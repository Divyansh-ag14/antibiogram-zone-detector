
# Detection and Segmentation using Detectron2

## Business Objective

Antibiograms are often used by clinicians to assess local susceptibility rates to select
empiric antibiotic therapy, and monitor resistance trends over time. The testing occurs
in a medical laboratory and uses culture methods that expose bacteria to antibiotics
to test if bacteria have genes that confer resistance. Culture methods often involve
measuring the diameter of areas without bacterial growth, called zones of inhibition.
The zone of inhibition is a circular area around the antibiotic spot in which the bacteria
colonies do not grow. The zone of inhibition can be used to measure the susceptibility
of the bacteria to the antibiotic. The process of locating the zones and related
inhibitions can be automated using image detection and segmentation.
Image Classification helps us classify what is contained in an image, whereas object
Detection specifies the location of multiple objects in the image. On the other hand,
image Segmentation will create a pixel-wise mask of each object in the images, which
helps identify the shapes of different objects in the image.

This project uses Facebook AI's [Detectron2](https://github.com/facebookresearch/detectron2) to perform object detection and instance segmentation for analyzing antibiogram plates. The goal is to detect and classify two object types: `zone` and `disk`.

---

## Project Structure

```
.
├── ML_Pipeline/
│   ├── training.py         # Handles training with custom dataset
│   ├── inference.py        # Performs inference on test images
│   ├── admin.py            # Defines training and output directories
│
├── Engine.py               # Script to run training or inference
│
├── Input/
│   └── Data/
│       ├── annotations/
│       │   └── annotations.json  # COCO-style annotation file
│       └── images/
│           ├── train/            # Training images
│           └── test/             # Inference images
│
└── output/                 # Stores model weights and inference outputs

```

---

## Training Pipeline

### File: `ML_Pipeline/training.py`

This script performs the following:

- Registers the COCO-style dataset:
  ```python
  register_coco_instances("detection_segmentaion", {}, "annotations/annotations.json", "images/train")
  ```

- Configures a Detectron2 model with:
  - Mask R-CNN backbone
  - Custom number of classes: `zone`, `disk`
  - Training on CPU
  - Pretrained weights from Detectron2 Model Zoo

- Saves the final trained model to:
  ```
  output/model_final.pth
  ```

- Example Config:
  ```python
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
  cfg.SOLVER.MAX_ITER = 100
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  ```

---

## Inference Pipeline

### File: `ML_Pipeline/inference.py`

- Loads the trained weights from `output/model_final.pth`
- Uses `DefaultPredictor` for making predictions on test images
- Visualizes predictions using Detectron2's `Visualizer` class
- Saves the output to `output/output_model/test_image_1_inference.jpg`

---

## Class Mapping

Ensure that the COCO annotations use category IDs:
- `0 → disk`
- `1 → zone`

For correct display of class labels, set metadata:

```python
MetadataCatalog.get("detection_segmentaion").thing_classes = ["zone", "disk"]
```

---

## Running Inference

Example code in `Engine.py`:

```python
from ML_Pipeline.inference import Detectron2Infer

image_path = "images/test/01-auto.jpg"
infer = Detectron2Infer()
outputs = infer.infer(image_path)
print(outputs["instances"])
```

---

## 🚀 Requirements

- Python 3.8+
- Detectron2
- OpenCV
- PyTorch
- Matplotlib

Install Detectron2:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

---

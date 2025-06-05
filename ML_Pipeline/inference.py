from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from ML_Pipeline.admin import output_path
import cv2
import os


class Detectron2Infer:
    def __init__(self):
        self.cfg = get_cfg()

        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = 4

        self.cfg.SOLVER.IMS_PER_BATCH = 2  # images per batch
        self.cfg.OUTPUT_DIR = output_path

        self.output_infer = os.path.join(output_path, "output_model")
        os.makedirs(self.output_infer, exist_ok=True)

        self.cfg.SOLVER.BASE_LR = 0.02
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.SOLVER.MAX_ITER = 2000
        self.cfg.SOLVER.STEPS = (1000, 1500)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 164
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.TEST.EVAL_PERIOD = 100

        # Register dataset once here (instead of every inference)
        self.register_name = "detection_segmentaion"
        annotation = os.path.join(self.cfg.OUTPUT_DIR, "..", "annotations", "annotations.json")
        image_dir = os.path.join(self.cfg.OUTPUT_DIR, "..", "images", "train")
        if self.register_name not in MetadataCatalog.list():
            register_coco_instances(self.register_name, {}, annotation, image_dir)
            MetadataCatalog.get(self.register_name).thing_classes = ["disk", "zone"]

    def infer(self, image_path):

        metadata = MetadataCatalog.get(self.register_name)

        im = cv2.imread(image_path)

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.cfg.DATASETS.TEST = (self.register_name,)

        predictor = DefaultPredictor(self.cfg)

        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        out_path = os.path.join(self.output_infer, 'test_image_1_inference.jpg')
        out.save(out_path)

        return outputs

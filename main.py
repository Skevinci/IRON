import detectron2
import numpy as np
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
import sys
import os
import json
import cv2
import random
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "./", "OFA"))
from utils.eval_utils import eval_step
setup_logger()


class IRON():
    def __init__(self):
        self.img_path = "/home/skevinci/research/iron/img/test.png"

    def mask_rcnn(self):
        """Mask R-CNN"""
        im = cv2.imread(self.img_path)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("Img", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    pipeline = IRON()
    pipeline.mask_rcnn()

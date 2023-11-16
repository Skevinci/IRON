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
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), "./", "OFA"))
from transformers.models.ofa.generate import sequence_generator
from transformers import OFATokenizer, OFAModel
setup_logger()


class IRON():
    def __init__(self):
        self.img_path = "/home/skevinci/research/iron/img/test.png"
        self.output_path = "/home/skevinci/research/iron/img/output"
        self.ofa_ckpt_path = "/home/skevinci/research/iron/OFA-large-caption/"
        
        self.bbox = None
        self.original_img = Image.open(self.img_path)

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
        # print(outputs["instances"].pred_classes)
        print("Predicted Boxes: ", outputs["instances"].pred_boxes)
        print("==========Mask R-CNN Finished==========")
        self.bbox = outputs["instances"].pred_boxes.tensor.to("cpu").numpy()
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("Img", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    def crop(self):
        """Crop image using bbox"""
        for i in range(len(self.bbox)):
            self.original_img.crop(self.bbox[i]).save(self.output_path + str(i) + ".png")

    def ofa(self):
        """OFA"""
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        tokenizer = OFATokenizer.from_pretrained(
            self.ofa_ckpt_path, use_cache=True)
        txt = "what does the image describe?"
        inputs = tokenizer([txt], return_tensors="pt").input_ids
        for i in range(len(self.bbox)):
            img = Image.open(self.output_path + str(i) + ".png")
            patch_img = patch_resize_transform(img).unsqueeze(0)

            model = OFAModel.from_pretrained(self.ofa_ckpt_path, use_cache=True)
            generator = sequence_generator.SequenceGenerator(
                tokenizer=tokenizer,
                beam_size=5,
                max_len_b=16,
                min_len=0,
                no_repeat_ngram_size=3,
            )
            data = {}
            data["net_input"] = {
                "input_ids": inputs, 'patch_images': patch_img, 'patch_masks': torch.tensor([True])}
            gen_output = generator.generate([model], data)
            gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

            model = OFAModel.from_pretrained(self.ofa_ckpt_path, use_cache=False)
            gen = model.generate(inputs, patch_images=patch_img,
                                num_beams=5, no_repeat_ngram_size=3)

            print(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print("==========OFA Finished==========")
        
    def execute(self):
        # get bbox using mask rcnn
        self.mask_rcnn()
        
        # pass rgb crop of each bbox to ofa and get caption
        self.crop()
        self.ofa()


if __name__ == '__main__':
    pipeline = IRON()
    pipeline.execute()

import numpy as np
from scipy.optimize import linear_sum_assignment

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
import clip
import sys
import os
import json
import cv2
import random
import shutil
import base64
# import hanlp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image
from io import BytesIO
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), "./", "OFA"))
from transformers.models.ofa.generate import sequence_generator
from transformers import OFATokenizer, OFAModel
from openai import OpenAI
from hanlp_restful import HanLPClient
from collections import Counter
import pickle

from draw import *

setup_logger()


class IRON:
    def __init__(self):
        self.img_path = "img/test1.jpg"
        self.crop_img_path = "img/crop/"
        self.gen_img_path = "img/generated/"
        self.gen_crop_img_path = "img/generated/crop/"
        self.ofa_ckpt_path = "OFA-large-caption/"
        self.save_path = None
        self.client = OpenAI()
        self.predictor = None  # Mask R-CNN
        self.tokenizer = None  # OFA
        self.inputs = None  # OFA
        self.model_cache = None  # OFA
        self.model_notcache = None  # OFA
        self.generator = None  # OFA
        self.clip_model = None  # CLIP
        self.clip_preprocess = None  # CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.bbox = None
        self.mask = None
        self.original_img = Image.open(self.img_path)
        self.caption = []
        self.img_feature = []
        
        self.prompt = ""
        self.num_generated = 4  # number of images to generate
        self.gen_bbox = []
        self.gen_mask = []
        self.gen_caption = {}
        self.gen_img_feature = {}

        self.debug = True  # test without calling Dall-E
        
    def initDir(self):
        if os.path.exists(self.crop_img_path):
            shutil.rmtree(self.crop_img_path)
        if os.path.exists(self.gen_img_path):
            shutil.rmtree(self.gen_img_path)
        if os.path.exists(self.gen_crop_img_path):
            shutil.rmtree(self.gen_crop_img_path)
        os.mkdir(self.crop_img_path)
        os.mkdir(self.gen_img_path)
        os.mkdir(self.gen_crop_img_path)

        if self.debug:
            print("###### Test Mode ######")
            shutil.copy("img/generated_test/0.jpg", "img/generated/0.jpg")
            shutil.copy("img/generated_test/1.jpg", "img/generated/1.jpg")
            shutil.copy("img/generated_test/2.jpg", "img/generated/2.jpg")
            shutil.copy("img/generated_test/3.jpg", "img/generated/3.jpg")

    def cfg_init(self):
        """Config"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
    
    def mask_rcnn(self, is_initial=False, count=0):
        """Mask R-CNN"""
        if is_initial:
            self.cfg_init()
            im = cv2.imread(self.img_path)
        else:
            im = cv2.imread(self.gen_img_path + f"{count}.jpg")

        outputs = self.predictor(im)
        print("Original output:", outputs["instances"].pred_boxes.tensor.to("cpu").numpy())

        # Filter crop boxes that are largely overlapping with selected ones
        img_box = np.array([0, 0, im.shape[0], im.shape[1]])
        overlap_threshold = min(im.shape[0], im.shape[1]) * 0.07  # By experiment
        bbox_candidates = outputs["instances"].pred_boxes.tensor.to("cpu").numpy()
        candidate_idx = []
        for i in range(bbox_candidates.shape[0]):
            if np.linalg.norm(bbox_candidates[i] - img_box) < overlap_threshold:
                print("Almost Full Size:", np.linalg.norm(bbox_candidates[i] - img_box))
                continue  # The crop box is almost identical to the original image
            if not candidate_idx:
                candidate_idx.append(i)
            else:
                is_overlap = False
                for j in candidate_idx:
                    if np.linalg.norm(bbox_candidates[i] - bbox_candidates[j]) < overlap_threshold:
                        print("Smaller than threshold:", np.linalg.norm(bbox_candidates[i] - bbox_candidates[j]))
                        is_overlap = True  # The current crop box is similar to the selected crop box_j
                if not is_overlap:
                    candidate_idx.append(i)

        if is_initial:
            self.bbox = bbox_candidates[candidate_idx]
            self.mask = outputs["instances"].pred_masks.to("cpu").numpy()[candidate_idx]
            draw_box(im, self.bbox, range(4), "img/out_test.jpg")
            print("Predicted Boxes: ", self.bbox)
        else:
            self.gen_bbox.append(bbox_candidates[candidate_idx])
            self.gen_mask.append(outputs["instances"].pred_masks.to("cpu").numpy()[candidate_idx])
            # draw_box(im, self.gen_bbox[-1], range(len(self.gen_bbox[-1])), f"img/out_test{count}.jpg")
            print("Predicted Boxes: ", self.gen_bbox[-1])

        print("==========Mask R-CNN Finished==========")
        # print(self.mask.shape)
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("Img", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    def crop(self, is_initial=False, count=0):
        """Crop image using bbox"""
        if is_initial:
            for i in range(len(self.bbox)):
                expanded_mask = np.expand_dims(self.mask[i, :, :], axis=-1)
                masked_img = self.original_img * expanded_mask
                masked_img = Image.fromarray(masked_img)
                masked_img.crop(self.bbox[i]).save(self.crop_img_path + str(i) + "m.png")
                self.original_img.crop(self.bbox[i]).save(self.crop_img_path + str(i) + ".png")
        else:
            os.mkdir(self.gen_crop_img_path + "/" + str(count) + "/")
            self.save_path = self.gen_crop_img_path + "/" + str(count) + "/"
            for i in range(len(self.gen_bbox[count])):
                crop_img = Image.open(self.gen_img_path + str(count) + ".jpg")
                expanded_mask = np.expand_dims(self.gen_mask[count][i, :, :], axis=-1)
                masked_img = crop_img * expanded_mask
                masked_img = Image.fromarray(masked_img)
                masked_img.crop(self.gen_bbox[count][i]).save(self.save_path + str(i) + "m.png")
                crop_img.crop(self.gen_bbox[count][i]).save(self.save_path + str(i) + ".png")

    def patch_resize_transform(self, image):
        """Patch Resize Transform for OFA"""
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return patch_resize_transform(image).unsqueeze(0)
    
    def ofa_init(self):
        self.tokenizer = OFATokenizer.from_pretrained(
            self.ofa_ckpt_path, use_cache=True)
        txt = "what does the image describe?"
        self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids
        self.model_cache = OFAModel.from_pretrained(self.ofa_ckpt_path, use_cache=True)
        # self.model_notcache = OFAModel.from_pretrained(self.ofa_ckpt_path, use_cache=False)
        self.generator = sequence_generator.SequenceGenerator(
                tokenizer=self.tokenizer,
                beam_size=5,
                max_len_b=16,
                min_len=0,
                no_repeat_ngram_size=3,
            )
        
    def gen_caption_fn(self, patch_img, i):
        data = {}
        data["net_input"] = {
            "input_ids": self.inputs, 'patch_images': patch_img, 'patch_masks': torch.tensor([True])}
        gen_output = self.generator.generate([self.model_cache], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
        # print(gen)
        # gen = self.model_notcache.generate(self.inputs, patch_images=patch_img,
        #                     num_beams=5, no_repeat_ngram_size=3)
        
        return gen
    
    def ofa(self, is_initial=False, count=0):
        """OFA"""
        if is_initial:
            self.ofa_init()
        
            for i in range(len(self.bbox)):
                img = Image.open(self.crop_img_path + str(i) + ".png")
                patch_img = self.patch_resize_transform(img)
                gen = self.gen_caption_fn(patch_img, i)
                self.caption.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
            
            print(self.caption)
        #
        # else:
        #     for i in range(len(self.gen_bbox[count])):
        #         img = Image.open(self.save_path + str(i) + ".png")
        #         patch_img = self.patch_resize_transform(img)
        #
        #         gen = self.gen_caption_fn(patch_img, i)
        #
        #         self.gen_caption[count] = []
        #         self.gen_caption[count].append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
            
            # print(self.gen_caption[count])
            
        print("==========OFA Finished==========")
    
    def clip_init(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
    def clip(self, is_initial=False, count=0):
        if is_initial:
            self.clip_init()
        
            for i in range(len(self.bbox)):
                img = Image.open(self.crop_img_path + str(i) + "m.png")
                img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    self.img_feature.append(self.clip_model.encode_image(img).cpu().numpy())
        
        else:
            self.gen_img_feature[count] = []
            for i in range(len(self.gen_bbox[count])):
                img = Image.open(self.save_path + str(i) + "m.png")
                img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    self.gen_img_feature[count].append(self.clip_model.encode_image(img).cpu().numpy())
        
    def img2representation(self, is_initial=False, count=0):
        """Convert image to representation"""
        # get bbox using mask rcnn
        self.mask_rcnn(is_initial, count)
        
        # pass rgb crop of each bbox to ofa and get caption
        self.crop(is_initial, count)
        #
        # pass caption to tagging model and get nouns if it is for prompt
        if is_initial:
            self.ofa(is_initial, count)
            self.gen_prompt()
        #
        # pass rgb crop of each bbox to clip and get feature
        self.clip(is_initial, count)
        
        # print
        if is_initial:
            print("==========Initial Image Caption==========", self.caption)
            print("==========Initial Image Feature Num==========", len(self.img_feature))
        else:
            # print(f"==========Generated Image{count} Caption==========", self.gen_caption[count])
            print(f"==========Generated Image{count} Feature Num==========", len(self.gen_img_feature[count]))
        
    def gen_prompt(self):
        HanLP = HanLPClient('https://www.hanlp.com/api', auth='', language='mul') # auth is your API key, which can applied from HanLP
        results = HanLP(self.caption, tasks='ud')
        tokenized = results['tok']
        pos = results['pos']

        # get nouns
        noun_counts = Counter()
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                if pos[i][j] == 'NOUN':
                    if pos[i][j - 1] == 'ADJ':
                        noun_counts[f"{tokenized[i][j - 1]} {tokenized[i][j]}"] += 1
                    else:
                        noun_counts[tokenized[i][j]] += 1

        prompt_list = []
        for tkn, cnt in noun_counts.items():
            if cnt == 1:
                prompt_list.append(f"a {tkn}")
            elif cnt <= len(pos) / 2:
                prompt_list.append(f"{cnt} {tkn}")

        prompt_list.append("on a wooden table, well arranged, no overlapping, top-down view")
        
        self.prompt = ", ".join(prompt_list)
        print(self.prompt)
    
    def save_b64(self, b64_str, count):
        img = Image.open(BytesIO(base64.b64decode(b64_str)))
        img.save(self.gen_img_path + str(count) + ".jpg")
        
    def dalle(self):
        if self.debug:
            for i in range(self.num_generated):
                print(f"==========Processing Generated Image{i}...==========")
                self.img2representation(is_initial=False, count=i)
        else:
            response = self.client.images.generate(
                model="dall-e-2",
                prompt=self.prompt,
                size="256x256",
                quality="standard",
                n=self.num_generated,
                response_format="b64_json",
            )

            # print(response)
            for i in range(self.num_generated):
                print(f"==========Processing Generated Image{i}...==========")
                b64_str = response.data[i].b64_json
                self.save_b64(b64_str, i)
                self.img2representation(is_initial=False, count=i)

        print("==========Generated Images Saved==========")
        
    def execute(self):
        """Execute"""
        # First, process initial image
        self.img2representation(is_initial=True)
        print("==========Finished Converting Initial Image==========")
        self.dalle()
        print("==========Finished Dall-E Generation==========")
        self.best_match()
        # self.savetemp()

    def savetemp(self):
        with open("img_feature.pkl", "wb") as f:
            pickle.dump(self.img_feature, f)
        with open("gen_feature.pkl", "wb") as f:
            pickle.dump(self.gen_img_feature, f)

    def best_match(self):
        def match_algorithm(cost_matrix):
            row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)
            return row_idx, col_idx, cost_matrix[row_idx, col_idx].sum()

        def similarity(v1, v2):
            return np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        best_img_idx = -1
        best_cos_sim = 0
        best_match_idx = []  # row_idx, col_idx

        for idx, m in self.gen_img_feature.items():
            if len(m) != len(self.img_feature):
                pass
            else:
                num = len(m)
                weight_matrix = np.zeros((num, num))
                for i in range(num):
                    for j in range(num):
                        weight_matrix[i, j] = similarity(self.img_feature[i], m[j])  # origin is row, candidate is col
                print(weight_matrix)
                row_ind, col_ind, max_sim_sum = match_algorithm(weight_matrix)
                if max_sim_sum > best_cos_sim:
                    best_img_idx = idx
                    best_cos_sim = max_sim_sum
                    best_match_idx = [row_ind, col_ind]

        if best_img_idx != -1:
            print(best_img_idx, best_cos_sim, best_match_idx)
            im = cv2.imread(self.gen_img_path + f"{best_img_idx}.jpg")
            draw_box(im, self.gen_bbox[best_img_idx], best_match_idx[1], "img/best_match.jpg")
        else:
            print("No best match!")


if __name__ == '__main__':
    pipeline = IRON()
    pipeline.initDir()
    pipeline.execute()
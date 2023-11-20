# Image geneRation for Open-world object RearraNgement (IRON)

Our idea comes from [DALL-E-Bot]🤖 and we are trying to reproduce part of its pipeline as described...

⛏️Coming soon!

## Objectives
- [x] Convert initial RGB observation into 
  - [x] bbox, masks and class labels using [Mask-RCNN] from the [Detectron 2] library
  - [x] text descriptions by passing bbox crops to [OFA]
  - [x] semantic feature vectors by pass bbox crops to [CLIP]
- [ ] Goal image generation
  - [x] integrate [DALL-E 2] api
  - [ ] text prompt design by passing text descriptions of objects to [Part-of-Speech tagging model] from the [Flair NLP] library (Optional)
  - [ ] image generation editing using masks

## Mask-RCNN (Detectron 2)

Using COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml from detectron 2.

Current packages change due to the installation of fairseq: 

1. omegaconf==2.0.6 (comment "from omegaconf import SCMode" and the related SCMode code.)
2. hydra-core==1.0.7

## OFA

~~Changes from official installment:~~

~~1. Delete directory fairseq, and using "pip install fairseq==0.12.2" instead.
2. Comment "pycocotools==2.0.4" in the requirements.txt since it conflicts with that of in Detectron2.
3. Downgrade protobuf using "pip install protobuf==3.20"~~

Follow https://huggingface.co/OFA-Sys/ofa-large-caption.

## CLIP

Follow https://github.com/openai/CLIP and use *model.encode_image* to get the semantic feature vector.

## Current Work Flow

Successfully implement Mask R-CNN, OFA and CLIP and get each object's mask, caption and feature vector. 

Next step: Pass masks and captions to the diffusion model and generate the picture we want. Save generated images as png file.

[DALL-E-Bot]: https://arxiv.org/abs/2210.02438
[DALL-E 2]: https://openai.com/dall-e-2
[Detectron 2]: https://github.com/facebookresearch/detectron2
[OFA]: https://arxiv.org/abs/2202.03052
[Mask-RCNN]: https://arxiv.org/abs/1703.06870
[CLIP]: https://openai.com/research/clip
[Part-of-Speech tagging model]: https://aclanthology.org/C18-1139.pdf
[Flair NLP]: https://aclanthology.org/N19-4010/
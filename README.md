# Image geneRation for Open-world object rearraNgement (IRON)

Our idea comes from [DALL-E-Bot]🤖 and we are trying to reproduce part of its pipeline as described...

⛏️Coming soon!

## Objectives
- [x] Convert initial RGB observation into 
  - [x] bbox, masks and class labels using [Mask-RCNN] from the [Detectron 2] library
  - [x] text descriptions by passing bbox crops to [OFA]
  - [x] semantic feature vectors by pass bbox crops to [CLIP]
- [ ] Goal image generation
  - [x] integrate [DALL-E 2] api
  - [x] text prompt design by passing text descriptions of objects to [Part-of-Speech tagging model] from the ~~[Flair NLP]~~ [Hanlp] library (Optional)
  - [ ] image generation editing using masks
- [ ] Goal image selection
  - [x] Convert every generated image into representations that are the same as the first part
  - [ ] Filter generated images by comparing the number of objects and whether movable objects overlap
  - [ ] Use [Hungarian Matching algorithm] to compute an assignment of each object in the initial image to an object in the generated image, such that the total cosine similarity score is maximized.

## Current Work Flow

Next step: 
1. Pass generated prompt to DALL-E 2 to generate images and do the image editing.
2. Filter generated images.
3. Implement Hungarian Matching Algorithm.

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

## DALL-E 2
```pip install openai```

Setup openai secret key in the environment.

Reference: 
- https://github.com/openai/openai-python
- https://platform.openai.com/docs/guides/images/introduction
- https://platform.openai.com/docs/api-reference/images/create

## Flair NLP (Use Hanlp instead)
~~```pip install flair```~~

~~Reference:~~
~~- https://flairnlp.github.io/docs/tutorial-basics/part-of-speech-tagging~~
~~- https://github.com/flairNLP/flair~~

## Hanlp
Follow https://github.com/hankcs/HanLP and use hanlp's RESTful API to do tokenize and Part of Speech in order to generate prompt.


[DALL-E-Bot]: https://arxiv.org/abs/2210.02438
[DALL-E 2]: https://openai.com/dall-e-2
[Detectron 2]: https://github.com/facebookresearch/detectron2
[OFA]: https://arxiv.org/abs/2202.03052
[Mask-RCNN]: https://arxiv.org/abs/1703.06870
[CLIP]: https://openai.com/research/clip
[Part-of-Speech tagging model]: https://aclanthology.org/C18-1139.pdf
[Flair NLP]: https://github.com/flairNLP/flair
[Hungarian Matching algorithm]: https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800020109
[Hanlp]: https://github.com/hankcs/HanLP
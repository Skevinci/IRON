**Detectron 2**

Using COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml from detectron 2.

Current packages change due to the installation of fairseq: 

1. omegaconf==2.0.6 (comment "from omegaconf import SCMode" and the related SCMode code.)
2. hydra-core==1.0.7

**OFA**

~~Changes from official installment:~~

~~1. Delete directory fairseq, and using "pip install fairseq==0.12.2" instead.
2. Comment "pycocotools==2.0.4" in the requirements.txt since it conflicts with that of in Detectron2.
3. Downgrade protobuf using "pip install protobuf==3.20"~~

Follow https://huggingface.co/OFA-Sys/ofa-large-caption.

**CLIP**

Follow https://github.com/openai/CLIP and use *model.encode_image* to get the image feature vector.

**Current Work Flow**

Successfully implement Mask R-CNN, OFA and CLIP and get each object's mask, caption and feature vector. 

Next step: Pass masks and captions to the diffusion model and generate the picture we want. Save generated images as png file.

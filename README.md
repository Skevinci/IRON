**Detectron 2**

Using COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml from detectron 2.

Current packages change due to the installation of fairseq: 

1. omegaconf==2.0.6 (comment "from omegaconf import SCMode" and the related SCMode code.)
2. hydra-core==1.0.7

**OFA**

Changes from official installment:

1. Delete directory fairseq, and using "pip install fairseq==0.12.2" instead.
2. Comment "pycocotools==2.0.4" in the requirements.txt since it conflicts with that of in Detectron2.
3. Downgrade protobuf using "pip install protobuf==3.20"

**Current Work Flow**

Successfully import OFA and run mask-RCNN. 

Next step: Pass rgb crops of bounding boxes after running mask-RCNN to OFA and get the caption of each rgb crop.

import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from datetime import datetime
from detectron2.engine import default_setup

'''
    Backbone model:
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 
''' 

def setup_rotated_cfg(args):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)  
    cfg.DATASETS.TRAIN = (['train'])
    # cfg.DATASETS.TEST = (["Test"])

    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RRPN'
    cfg.MODEL.RPN.HEAD_NAME = 'StandardRPNHead'
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1,1,1,1,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = 'RotatedAnchorGenerator'
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-60,-30,0,30,60]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NAME = 'RROIHeads'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #this is far lower than usual.  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = 'ROIAlignRotated'
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
    cfg.SOLVER.IMS_PER_BATCH = 5 #can be up to  24 for a p100 (6 default)
    cfg.SOLVER.CHECKPOINT_PERIOD=1000
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.STEPS=[1000,2000,4000,8000, 12000]
    cfg.SOLVER.MAX_ITER=10000

    cfg.INPUT.MIN_SIZE_TRAIN = (300,)
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MIN_SIZE_TEST = 300
    cfg.INPUT.MAX_SIZE_TEST = 400

    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
    cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD=0.01
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#lets just check our output dir exists
    cfg.MODEL.BACKBONE.FREEZE_AT=6

    # Setup Logging folder
    curTime = datetime.now()
    cfg.OUTPUT_DIR = "./output/" + curTime.strftime("%m%d%Y%H%M%S")
    if not os.path.exists(os.getcwd() + "/output/"):
        os.makedirs(os.getcwd() + "/output/", exist_ok=True)
        print("Done creating folder output!")
    else:
        print("Folder output/ existed!")
    if not os.path.exists(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S")):    
        os.makedirs(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S"), exist_ok=True)

    cfg.freeze()                    # make the configuration unchangeable during the training process
    default_setup(cfg, args)
    return cfg

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)  

    # dataset configuration  
    cfg.DATASETS.TRAIN = (['train'])
    # cfg.DATASETS.TEST = (["Test"])
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 5                    # 2 GPUs --> each GPU will see 25 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,8,16,32,64]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 category (roadway stress)

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.INPUT.MIN_SIZE_TRAIN = (300,)
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MIN_SIZE_TEST = 300
    cfg.INPUT.MAX_SIZE_TEST = 400

    # Setup Logging folder
    curTime = datetime.now()
    cfg.OUTPUT_DIR = "./output/" + curTime.strftime("%m%d%Y%H%M%S")
    if not os.path.exists(os.getcwd() + "/output/"):
        os.makedirs(os.getcwd() + "/output/", exist_ok=True)
        print("Done creating folder output!")
    else:
        print("Folder output/ existed!")
    if not os.path.exists(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S")):    
        os.makedirs(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S"), exist_ok=True)

    cfg.freeze()                    # make the configuration unchangeable during the training process
    default_setup(cfg, args)
    return cfg

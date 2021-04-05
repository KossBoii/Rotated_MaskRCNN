import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

def setup_rotated_cfg():
    cfg = get_cfg()

    cfg.OUTPUT_DIR = './output'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
    cfg.DATASETS.TRAIN = (['train'])
    # cfg.DATASETS.TEST = (["Test"])

    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RRPN'
    cfg.MODEL.RPN.HEAD_NAME = 'StandardRPNHead'
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1,1,1,1,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = 'RotatedAnchorGenerator'
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NAME = 'RROIHeads'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #this is far lower than usual.  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = 'ROIAlignRotated'
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
    cfg.SOLVER.IMS_PER_BATCH = 25 #can be up to  24 for a p100 (6 default)
    cfg.SOLVER.CHECKPOINT_PERIOD=1000
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.STEPS=[1000,2000,4000,8000, 12000]
    cfg.SOLVER.MAX_ITER=10000


    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
    cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD=0.01
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#lets just check our output dir exists
    cfg.MODEL.BACKBONE.FREEZE_AT=6
    return cfg

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")    

    # dataset configuration  
    cfg.DATASETS.TRAIN = (['train'])
    # cfg.DATASETS.TEST = (["Test"])
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 25                    # 2 GPUs --> each GPU will see 1 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,8,16,32,64,128]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 category (roadway stress)

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 800
    return cfg

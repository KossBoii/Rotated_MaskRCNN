from rotatedmodel import *
from configs import *
from utils import get_dataset, get_rotated_dataset
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import json
import math
import argparse
import torch
import sys
from detectron2.engine import launch
import logging
import os
logger = logging.getLogger("detectron2")
from torch.utils.cpp_extension import CUDA_HOME

def main(args):
    print(torch.cuda.is_available(), CUDA_HOME)
    ROOT_DIR = os.getcwd()
    dataset_path = os.path.join(ROOT_DIR, 'dataset')

    # Register the dataset
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    # setup config
    if args.model == 'normal':
        for d in ['train']:
            DatasetCatalog.register(d, lambda d=d: get_dataset(os.path.join(dataset_path, d)))
            MetadataCatalog.get(d).set(thing_classes=['car'])

        cfg = setup_cfg(args)
        cfg.dump()

        # training starts
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    elif args.model == 'rotate':
        for d in ['train']:
            DatasetCatalog.register(d, lambda d=d: get_rotated_dataset(os.path.join(dataset_path, d)))
            MetadataCatalog.get(d).set(thing_classes=['car'])
        
        cfg = setup_rotated_cfg(args)
        cfg.dump()

        # training starts
        trainer = RotatedTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    # predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    
    # visualize the result
    img = cv2.imread('./dataset/train/imgs/DJI_0002.JPG')
    outputs = predictor(img)
    vis = RotatedVisualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get("Train"), 
        scale=1.0
    )

    out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imsave('result.jpeg', out.get_image()[:, :, ::-1])

def custom_default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default='normal', help='type of the model (normal/rotate)')
    parser.add_argument(
        "--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="backbone model"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = custom_default_argument_parser().parse_args()
    print("Command Line Args:", args)
    torch.backends.cudnn.benchmark = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


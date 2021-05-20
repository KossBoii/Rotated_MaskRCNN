from rotatedmodel import *
from configs import *
from utils import get_dataset, get_rotated_dataset
from detectron2.engine import DefaultPredictor
import os
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import cv2
import json
import math
import argparse
import torch
import sys
from detectron2.engine import launch
import logging

logger = logging.getLogger("detectron2")
from torch.utils.cpp_extension import CUDA_HOME
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators, inference_on_dataset, coco_evaluation,DatasetEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

class MyRotatedCOCOEvaluator(RotatedCOCOEvaluator):
  def _eval_predictions(self, tasks, predictions, img_ids=None):
    super()._eval_predictions(tasks, predictions)

def custom_default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

def main(args):
    print(torch.cuda.is_available(), CUDA_HOME)
    ROOT_DIR = os.getcwd()
    dataset_path = os.path.join(ROOT_DIR, 'dataset')

    # Register the dataset
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    # List of models' model ID
    model_list = [
        '04292021100459', '04302021103509', '04292021213751', '05032021065307',
        '05012021092356', '05022021202935', '05012021164640', '05022021161415',
        '05022021164657', '05032021185038', 
    ]
    # '05052021175642', '05052021170048',

    # setup config
    for d in ['train']:
        DatasetCatalog.register(d, lambda d=d: get_rotated_dataset(os.path.join(dataset_path, d)))
        MetadataCatalog.get(d).set(thing_classes=['car'])

    for model_id in model_list:
        print('Model ', model_id)
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join('./output/' + model_id, 'config.yaml'))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.DATASETS.TEST = (['train'])
        
        model = build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)

        # trainer = RotatedTrainer(cfg)

        evaluator = RotatedCOCOEvaluator("train", cfg, False, output_dir=cfg.OUTPUT_DIR)
        # evaluator = MyRotatedCOCOEvaluator("train", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "train", mapper=custom_rotated_mapper) 
        outputs = inference_on_dataset(model, val_loader, evaluator)

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

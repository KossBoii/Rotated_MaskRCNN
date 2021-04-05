from configs import setup_cfg
from rotatedmodel import *
from configs import *
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import json
import math
import argparse
import torch
import sys
from detectron2.engine import launch

id_to_category = {
    0: 'car'
}

category_to_id = {
    'car': 0
}

def get_rotated_dataset(dataset_path):
    # Load and read json file stores information about annotations
    json_file = os.path.join(dataset_path, 'segmask/annotations.json')
    with open(json_file) as f:
        imgs_annos = json.load(f)

    dataset_dicts = []          # list of annotations info for every images in the dataset
    for idx, v in enumerate(imgs_annos.values()):                   # loop through every image
        if(v["regions"]):
            record = {}         # a dictionary to store all necessary info of each image in the dataset
            
            # open the image to get the height and width
            filename = os.path.join(dataset_path, 'imgs/' + v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            # parsing rotated bounding box
            rotated_bboxes = []
            rbbox_filename = os.path.join(dataset_path, 'rbbox/%s.txt' % v["filename"][:-4])
            with open(rbbox_filename) as bbox_file:
                lines = bbox_file.readlines()
                for line in lines:
                    temp = line.split()
                    assert len(temp) == 6
                    cx = float(temp[1])
                    cy = float(temp[2])
                    w = float(temp[3])
                    h = float(temp[4])
                    a = 180 - math.degrees(float(temp[5]))
                    rbbox = [cx, cy, w, h, a]
                    rotated_bboxes.append(rbbox)

            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            for anno_id, anno in enumerate(annos):
                class_name = anno['region_attributes']['name']
                anno = anno['shape_attributes']
                px = anno['all_points_x']
                py = anno['all_points_y']
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    'bbox': rotated_bboxes[anno_id],
                    'bbox_mode': BoxMode.XYWHA_ABS,
                    'segmentation': [poly],
                    'category_id': category_to_id[class_name],
                    'iscrowd': 0
                }
                objs.append(obj)
            record['annotations'] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def get_dataset(dataset_path):
    # Load and read json file stores information about annotations
    json_file = os.path.join(dataset_path, "bbox/via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []          # list of annotations info for every images in the dataset
    for idx, v in enumerate(imgs_anns.values()):
        if(v["regions"]):
            record = {}         # a dictionary to store all necessary info of each image in the dataset
            
            # open the image to get the height and width
            filename = os.path.join(dataset_path, 'imgs/' + v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            for anno in annos:
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def main(args):
    ROOT_DIR = os.getcwd()
    dataset_path = os.path.join(ROOT_DIR, 'dataset')

    # Register the dataset
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    for d in ['train']:
        DatasetCatalog.register(d, lambda d=d: get_dataset(os.path.join(dataset_path, d)))
        MetadataCatalog.get(d).set(thing_classes=['car'])

    # setup config
    cfg = setup_cfg()
    cfg.dump()

    # training starts
    if args.model == 'normal':
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()
    elif args.model == 'rotate':
        trainer = RotatedTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()

    # predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    
    # visualize the result
    img = cv2.imread('./dataset/train/imgs/DJI_0064.JPG')
    outputs = predictor(img)
    vis = RotatedVisualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get("Train"), 
        scale=1.0
    )

    out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imsave('result.jpeg', out.get_image()[:, :, ::-1])

def custom_default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
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


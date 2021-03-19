from configs import setup_cfg
from rotatedmodel import *
from configs import *
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import json
import math

id_to_category = {
    0: 'car'
}

category_to_id = {
    'car': 0
}

def get_dataset(dataset_path):
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
            rbbox_filename = os.path.join(dataset_path, 'rrbox/%s.txt' % v["filename"][:-4])
            with open(rbbox_filename) as bbox_file:
                lines = bbox_file.readlines()
                for line in lines:
                    temp = line.split()
                    assert len(temp) == 6
                    cx = temp[1]
                    cy = temp[2]
                    w = temp[3]
                    h = temp[4]
                    a = 180 - math.degrees(temp[5])
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

def main():
    ROOT_DIR = os.getcwd()
    dataset_path = os.path.join(ROOT_DIR, 'dataset')

    # Register the dataset
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    for d in ['train']:
        DatasetCatalog.register(d, lambda d=d: get_dataset(d))
        MetadataCatalog.get(d).set(thing_classes=['car'])

    # setup config
    cfg = setup_cfg()
    cfg.dump()

    # training starts
    trainer = RotatedTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    
    # visualize the result
    img = cv2.imread('test_img.jpeg')
    outputs = predictor(img)
    vis = RotatedVisualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get("Train"), 
        scale=0.5
    )

    out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imsave('result.jpeg', out.get_image()[:, :, ::-1])


if __name__ == '__main__':
    main()


import detectron2
import copy
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torch
import numpy as np
import os
import json
import cv2
import math

def transform_instance_annotations_rotated(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:        # rotated bbox
        annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation

def custom_rotated_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
                    T.Resize((300, 400)),
                    T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                    # T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
                    # T.RandomBrightness(0.9, 1.1),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        transform_instance_annotations_rotated(obj, transforms, image.shape[:2]) 
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

def custom_mapper(dataset_dict):
  dataset_dict = copy.deepcopy(dataset_dict)
  image = utils.read_image(dataset_dict["file_name"], format="BGR")

  transform_list = [
                    T.Resize((300, 400)),
                    T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
                    T.RandomBrightness(0.9, 1.1),
                    ]
  image, transforms = T.apply_transform_gens(transform_list, image)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
  annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)
  return dataset_dict


    # dataset_dict = copy.deepcopy(dataset_dict)
    # image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # # Data Augmentation 
    # augs = T.AugmentationList([
    #     T.Resize((300, 400)),
    #     T.RandomBrightness(0.9, 1.1),
    #     T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
    #     T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
    # ])

    # input = T.AugInput(image)
    # transforms = augs(input)
    # # image = torch.from_numpy(input.image.transpose(2,0,1))
    # image = torch.as_tensor(input.image.transpose(2, 0, 1))

    # annos = [
    #     transform_instance_annotations_rotated(annotation, transforms, image.shape[1:])
    #     for annotation in dataset_dict.pop('annotations')
    # ]
    # instances = utils.annotations_to_instances_rotated(annos, image.shape[1:])

    # return {
    #     'image': image,
    #     'instances': utils.filter_empty_instances(instances)
    # }

id_to_category = {
    0: 'car'
}

category_to_id = {
    'car': 0
}

def get_rotated_dataset(dataset_path):
    # Load and read json file stores information about annotations
    json_file = os.path.join(dataset_path, 'segmask/via_export_json.json')
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
            count=0
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
                    count+= 1

            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            # print('count: %d' % count)
            # print('anno_id: %d' % len(annos))
            for anno_id, anno in enumerate(annos):
                class_name = anno['region_attributes']['name']
                anno = anno['shape_attributes']
                px = anno['all_points_x']
                py = anno['all_points_y']
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                try:
                    obj = {
                        'bbox': rotated_bboxes[anno_id],
                        'bbox_mode': BoxMode.XYWHA_ABS,
                        'segmentation': [poly],
                        'category_id': category_to_id[class_name],
                        'iscrowd': 0
                    }
                except IndexError:
                    print(v["filename"])
                except KeyError:
                    print(v["filename"])
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


import detectron2
import copy
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torch

def transform_instance_annotations_rotated(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:        # rotated bbox
        annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

def custom_rotated_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # Data Augmentation 
    augs = T.AugmentationList([
        T.Resize((600, 800)),
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
    ])

    input = T.AugInput(image)
    transforms = augs(input)
    # image = torch.from_numpy(input.image.transpose(2,0,1))
    image = torch.as_tensor(input.image.transpose(2, 0, 1))

    annos = [
        transform_instance_annotations_rotated(annotation, transforms, image.shape[1:])
        for annotation in dataset_dict.pop('annotations')
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[1:])

    return {
        'image': image,
        'instances': utils.filter_empty_instances(instances)
    }

def custom_mapper(dataset_dict):
  dataset_dict = copy.deepcopy(dataset_dict)
  image = utils.read_image(dataset_dict["file_name"], format="BGR")

  transform_list = [
                    T.Resize((600, 800)),
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
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

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # Data Augmentation 
    augs = T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.5),
        T.RandomCrop("absolute", (640, 640))
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

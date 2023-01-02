# Data
## Detectron2
#### Setting up DataLoaders
```
trainloader = detectron2.data.build_detection_train_loader(
    total_batch_size, 
    mapper,
    dataset,
    sampler
)

mapper = detectron2.data.DatasetMapper(is_train, augmentations, image_format)
is_train: bool
augmentations: list of transforms i.e [T.RandomBrightness(), T.RandomFlip()]
image_format: 'RGB', 'BGR', ...

sampler = detectron2.data.samplers.TrainingSampler(shuffle, size)
# detectron sampler infinitely streams samples from a given dataset 
# iteration training (each batch is an iteration), doesn't support epoch training
size: int (size of the dataset to sample from, typically len(dataset))
shuffle: bool

dataset = detectron2.data.build.get_detection_dataset_dicts(name)
name: str i.e 'coco_2017_train'
```
**Note**: when mapper is None, bounding boxes (assuming raw coco annotations) are in XYWH format. However, detectron2.data.DatasetMapper converts annotations to XYXY (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format. Also returns images, bounding boxes, and classes in Tensor format.

#### Registering datasets (COCO)
Detectron has predefined names for the `get_detection_dataset_dicts` function with some wack filepaths. Can set custom filepaths to dataset folders like so:
```
detectron2.data.datasets.register_coco_instances(
    name,
    {},
    annotations,
    image_folder
)
# example:
name = 'my_coco_train2017'
annotations = 'Users/sleepy/Documents/Data/coco/annotations/instances_train2017.json'
image_folder = 'Users/sleepy/Documents/Data/coco/images/train2017/'

dataset = detectron2.data.build.get_detection_dataset_dicts('my_coco_train2017')

```
# mmdetection-lib

## directory structure
```
├── backbones
│   ├── backbone_1
│   │    ├── ...
│   │    └── ...
│   ├── backbone_2
│   │    ├── ...
│   │    └── ...
│   └── ...
│
├── necks
│   ├── neck_1
│   │    ├── ...
│   │    └── ...
│   ├── neck_2
│   │    ├── ...
│   │    └── ...
│   └── ...
│
└── heads
│   ├── head_1
│   │    ├── ...
│   │    └── ...
│   ├── head_2
│   │    ├── ...
│   │    └── ...
│   └── ...
```
## how to use
copy paste the model file into the respective directory in the mmdetection library i.e `mmdetection/mmdet/models/necks/slpy_dyhead.py` from `necks/dyead/slyp_dyhead.py`<br> <br>
add the relevant model to `__init__.py` i.e 
```
# mmdet models
from .rfp import RFP
from .fpn import FPN
...
# custom models
from .sleepy_dyhead import SleepyDyHead
```
initializing the model is straightforward and employs the same method as mmdetection's (config files)

```
# EXAMPLE: initializing an entire detector with FPN + self-implemented dyhead neck,
# resnet50 backbone, and ATSS head
from mmdet.models.detectors.atss import ATSS

backbone_cfg = dict(...)
head_cfg = dict(...)
neck_cfg = [
    dict(type='FPN', ...),
    dict(type='SleepyDyHead', *args)
]
train_cfg, test_cfg = edict(...), edict(...)

detector = ATSS(backbone_cfg, neck_cfg, head_cfg, train_cfg, test_cfg, ...)
```
**may need to run pip install -v -e . in the mmdetection folder again**
## notes
all custom models are prefixed with `slpy_modelname.py` to avoid conflicting file names in the case of similar/reimplemented models <br>
this repo is primarily used for reimplementing and trying new necks/heads (backbones are boring XD)
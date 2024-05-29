import numpy as np
import os.path as osp
from collections import namedtuple
from src.datasets import IGNORE_LABEL as IGNORE


########################################################################
#                         Download information                         #
########################################################################

CVLIBS_URL = 'http://www.cvlibs.net/datasets/kitti-360/download.php'
BASE_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360'
DATA_3D_SEMANTICS_URL = osp.join(BASE_URL, '6489aabd632d115c4280b978b2dcf72cb0142ad9/data_3d_semantics.zip')
DATA_3D_SEMANTICS_TEST_URL = osp.join(BASE_URL, '6489aabd632d115c4280b978b2dcf72cb0142ad9/data_3d_semantics_test.zip')
CALIBRATION_URL = osp.join(BASE_URL, '384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip')
DATA_POSES_URL = osp.join(BASE_URL, '89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip')
DATA_3D_SEMANTICS_ZIP_NAME = 'data_3d_semantics.zip'
DATA_3D_SEMANTICS_TEST_ZIP_NAME = 'data_3d_semantics_test.zip'
UNZIP_NAME = 'data_3d_semantics'

########################################################################
#                              Data splits                             #
########################################################################

# These train and validation splits were extracted from:
#   - 'data_3d_semantics/2013_05_28_drive_train.txt'
#   - 'data_3d_semantics/2013_05_28_drive_val.txt'
WINDOWS = {
    'train': ['scenes/fox', 'scenes/counter', 'scenes/room'],
    'val': ['scenes/fox'],
    'test': ['scenes/fox']
    }

SEQUENCES = {
    k: list(set(osp.dirname(x) for x in v)) for k, v in WINDOWS.items()}


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/autonomousvision/kitti360Scripts

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'kittiId',  # An integer ID that is associated with this label for KITTI-360
    # NOT FOR RELEASING

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'ignoreInInst',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations of instance segmentation or not

    'color',  # The color of this label
])

# A list of all labels
# NB:
#   Compared to the default KITTI360 implementation, we set all classes to be
#   ignored at train time to IGNORE. Besides, for 3D semantic segmentation, the
#   'train', 'bus', 'rider' and 'sky' classes are absent from evaluationn so we
#   adapt 'ignoreInEval', 'ignoreInInst' and 'trainId' accordingly. Finally, it
#   seems, from the official website, the that only 'building' and 'car' are
#   actually evaluated for 3D instance segmentation, so we set 'ignoreInInst'
#   accordingly for all other classes.
#
#   See:
#   https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalPointLevelSemanticLabeling.py

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , True         , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , True         , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,    IGNORE , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,    IGNORE , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , True         , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , True         , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,    IGNORE , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,    IGNORE , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,    IGNORE , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,    IGNORE , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , True         , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , True         , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,    IGNORE , 'sky'             , 5       , False        , True         , True         , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        10 , 'human'           , 6       , True         , False        , True         , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,    IGNORE , 'human'           , 6       , True         , True         , True         , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        11 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        12 , 'vehicle'         , 7       , True         , False        , True         , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        13 , 'vehicle'         , 7       , True         , False        , True         , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        14 , 'vehicle'         , 7       , True         , False        , True         , (119, 11, 32) ),
    Label(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,    IGNORE , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,    IGNORE , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,    IGNORE , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,    IGNORE , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,    IGNORE , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,    IGNORE , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

# Dictionaries for a fast lookup
NAME2LABEL = {label.name: label for label in labels}
ID2LABEL = {label.id: label for label in labels}
TRAINID2LABEL = {label.trainId: label for label in reversed(labels)}
KITTIID2LABEL = {label.kittiId: label for label in labels}  # KITTI-360 ID to cityscapes ID
CATEGORY2LABELS = {}
for label in labels:
    category = label.category
    if category in CATEGORY2LABELS:
        CATEGORY2LABELS[category].append(label)
    else:
        CATEGORY2LABELS[category] = [label]
KITTI360_NUM_CLASSES = len(TRAINID2LABEL) - 1  # 15 classes for 3D semantic segmentation
INV_OBJECT_LABEL = {k: TRAINID2LABEL[k].name for k in range(KITTI360_NUM_CLASSES)}
OBJECT_COLOR = np.asarray([TRAINID2LABEL[k].color for k in range(KITTI360_NUM_CLASSES)])
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}
ID2TRAINID = np.array([label.trainId for label in labels])
TRAINID2ID = np.asarray([TRAINID2LABEL[c].id for c in range(KITTI360_NUM_CLASSES)] + [0])
CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(KITTI360_NUM_CLASSES)] + ['ignored']
CLASS_COLORS = np.append(OBJECT_COLOR, np.zeros((1, 3), dtype=np.uint8), axis=0)

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [label.trainId for label in labels if not label.ignoreInInst]
STUFF_CLASSES = [i for i in range(KITTI360_NUM_CLASSES) if not i in THING_CLASSES]

########################################################################
#                        Gaussian Splatting Path                       #
########################################################################
GS_PATH = "/workspace/gaussian_splatting/output"
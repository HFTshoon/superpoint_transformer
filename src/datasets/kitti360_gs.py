import os
import sys
import glob
import torch
import shutil
import logging
from zipfile import ZipFile
from plyfile import PlyData
from torch_geometric.data.extract import extract_zip
from src.datasets import BaseDataset
from src.data import Data, InstanceData
from src.datasets.kitti360_config_gs import *
from src.utils.neighbors import knn_2
from src.utils.color import to_float_rgb
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from tqdm.auto import tqdm as tq
from src.data import NAG
from src.transforms import NAGSelectByKey, NAGRemoveKeys, SampleXYTiling, \
    SampleRecursiveMainXYAxisTiling

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with KITTI360 on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['KITTI360GS', 'MiniKITTI360GS']


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_window(
        filepath, xyz=True, rgb=True, semantic=True, instance=True,
        remap=False):
    """Read a KITTI-360 window –i.e. a tile– saved as PLY.

    :param filepath: str
        Absolute path to the PLY file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their KITTI-360 ID
        to their train ID. For more details, see:
        https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalPointLevelSemanticLabeling.py
    """
    data = Data()
    with open(filepath, "rb") as f:
        window = PlyData.read(f)
        attributes = [p.name for p in window['vertex'].properties]

        if xyz:
            pos = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["x", "y", "z"]], dim=-1)
            pos_offset = pos[0]
            data.pos = pos - pos_offset
            data.pos_offset = pos_offset

        if rgb:
            data.rgb = to_float_rgb(torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["red", "green", "blue"]], dim=-1))

        if semantic and 'semantic' in attributes:
            y = torch.LongTensor(window["vertex"]['semantic'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance and 'instance' in attributes:
            idx = torch.arange(data.num_points)
            obj = torch.LongTensor(window["vertex"]['instance'])
            # is_stuff = obj % 1000 == 0
            # obj[is_stuff] = 0
            obj = consecutive_cluster(obj)[0]
            count = torch.ones_like(obj)
            y = torch.LongTensor(window["vertex"]['semantic'])
            y = torch.from_numpy(ID2TRAINID)[y] if remap else y
            data.obj = InstanceData(idx, obj, count, y, dense=True)

    return data


########################################################################
#                              KITTI360GS                              #
########################################################################

class KITTI360GS(BaseDataset):
    """KITTI360 dataset.

    Dataset website: http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = CVLIBS_URL
    _trainval_zip_name = DATA_3D_SEMANTICS_ZIP_NAME
    _test_zip_name = DATA_3D_SEMANTICS_TEST_ZIP_NAME
    _unzip_name = UNZIP_NAME
    
    _gs_path = GS_PATH

    @property
    def class_names(self):
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return KITTI360_NUM_CLASSES

    @property
    def stuff_classes(self):
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES

    @property
    def class_colors(self):
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return WINDOWS

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── raw/
            └── data_3d_semantics/
                └── 2013_05_28_drive_{{seq:0>4}}_sync/
                    └── static/
                        └── {{start_frame:0>10}}_{{end_frame:0>10}}.ply
            """

    def process(self):
        # If some stages have mixed clouds (they rely on the same cloud
        # files and the split is operated at reading time by
        # `on_device_transform`), we create symlinks between the
        # necessary folders, to avoid duplicate preprocessing
        # computation
        hash_dir = self.pre_transform_hash
        train_dir = osp.join(self.processed_dir, 'train', hash_dir)
        val_dir = osp.join(self.processed_dir, 'val', hash_dir)
        test_dir = osp.join(self.processed_dir, 'test', hash_dir)
        if not osp.exists(train_dir):
            os.makedirs(train_dir, exist_ok=True)
        if not osp.exists(val_dir):
            if self.val_mixed_in_train:
                os.makedirs(osp.dirname(val_dir), exist_ok=True)
                os.symlink(train_dir, val_dir, target_is_directory=True)
            else:
                os.makedirs(val_dir, exist_ok=True)
        if not osp.exists(test_dir):
            if self.test_mixed_in_val:
                os.makedirs(osp.dirname(test_dir), exist_ok=True)
                os.symlink(val_dir, test_dir, target_is_directory=True)
            else:
                os.makedirs(test_dir, exist_ok=True)

        # Process clouds one by one
        for p in tq(self.processed_paths):
            self._process_single_cloud(p)

    def _process_single_cloud(self, cloud_path):
        """Internal method called by `self.process` to preprocess a
        single cloud of 3D points.
        """
        # If required files exist, skip processing
        if osp.exists(cloud_path):
            return

        # Create necessary parent folders if need be
        os.makedirs(osp.dirname(cloud_path), exist_ok=True)

        # Read the raw cloud corresponding to the final processed
        # `cloud_path` and convert it to a Data object
        raw_path = self.processed_to_raw_path(cloud_path)
        data = self.sanitized_read_single_raw_cloud(raw_path)

        # If the cloud path indicates a tiling is needed, apply it here
        if self.xy_tiling is not None:
            tile = self.get_tile_from_path(cloud_path)[0]
            data = SampleXYTiling(x=tile[0], y=tile[1], tiling=tile[2])(data)
        elif self.pc_tiling is not None:
            tile = self.get_tile_from_path(cloud_path)[0]
            data = SampleRecursiveMainXYAxisTiling(x=tile[0], steps=tile[1])(data)

        # Apply pre_transform
        if self.pre_transform is not None:
            nag = self.pre_transform(data)
        else:
            nag = NAG([data])

        # To save some disk space, we discard some level-0 attributes
        if self.point_save_keys is not None:
            keys = set(nag[0].keys) - set(self.point_save_keys)
            nag = NAGRemoveKeys(level=0, keys=keys)(nag)
        elif self.point_no_save_keys is not None:
            nag = NAGRemoveKeys(level=0, keys=self.point_no_save_keys)(nag)
        if self.segment_save_keys is not None:
            keys = set(nag[1].keys) - set(self.segment_save_keys)
            nag = NAGRemoveKeys(level='1+', keys=keys)(nag)
        elif self.segment_no_save_keys is not None:
            nag = NAGRemoveKeys(level=0, keys=self.segment_no_save_keys)(nag)

        # Save pre_transformed data to the processed dir/<path>
        # TODO: is you do not throw away level-0 neighbors, make sure
        #  that they contain no '-1' empty neighborhoods, because if
        #  you load them for batching, the pyg reindexing mechanism will
        #  break indices will not index update
        nag.save(
            cloud_path,
            y_to_csr=self.save_y_to_csr,
            pos_dtype=self.save_pos_dtype,
            fp_dtype=self.save_fp_dtype)
        del nag

########################################################################
#                            MiniKITTI360GS                            #
########################################################################

class MiniKITTI360GS(KITTI360GS):
    """A mini version of KITTI360 with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()

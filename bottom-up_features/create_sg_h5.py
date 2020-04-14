import argparse
import os
import sys
try:
    from lib.config import cfg as sgg_cfg
    from lib.data.transforms import build_transforms
    from lib.scene_parser.parser import build_scene_parser
    from lib.scene_parser.rcnn.utils.model_serialization import load_state_dict
except ModuleNotFoundError:
    print("""ERROR: Could not import libraries from `graph-rcnnn.pytorch `.
Please run this script with the environment variable PYTHONPATH including the path to `graph-rcnn.pytorch`.
Example:
PYTHONPATH="/path/to/graph-rcnn.pytorch" python create_sg_h5.py.""")
    sys.exit(1)
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset
import utils


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.imgids, self.imagepaths = utils.load_imageinfo(image_dir)
        self.imgids = list(self.imgids)

    def __len__(self) -> int:
        return len(self.imgids)

    def __getitem__(self, index: int) -> dict:
        """
        :param index: The index of the item to retrieve
        :return: One data pair (image and caption).
        """
        imgid = self.imgids[index]
        image_path = self.imagepaths[imgid]
        image = utils.pil_loader(image_path)  # self.load_image(image_path)
        return {
            'image': image,
            'id': imgid,
            'image_file': os.path.basename(image_path),
        }


def get_input_hook(module, input, output):
    """
    A hook to assign to modules to get there output later
    :param module: the module
    :param input: the input of the module
    :param output: the output of the module
    """
    setattr(module, "_input_value_hook", input)


def collect_sgg_features(dataset, buffer_size=1):
    sgg_cfg.merge_from_file(os.path.join(sgg_cfg_file))
    sgg_cfg.inference = True
    sgg_cfg.instance = -1
    sgg_cfg.resume = 1
    trans = build_transforms(sgg_cfg, is_train=False)
    scene_parser = build_scene_parser(sgg_cfg)
    scene_parser.to(device)
    scene_parser.rel_heads.rel_predictor.obj_predictor.register_forward_hook(get_input_hook)
    scene_parser.rel_heads.rel_predictor.pred_predictor.register_forward_hook(get_input_hook)
    checkpoint = torch.load(sgg_weight_file)
    if "model" not in checkpoint:
        checkpoint = dict(model=checkpoint)
    load_state_dict(scene_parser, checkpoint.pop("model"))
    scene_parser.eval()
    # create dataloader to loop over the dataset
    start_ = 0
    for _ in range(int(np.ceil(len(dataset)/buffer_size))):
        bs = len(dataset)-start_ if start_+buffer_size > len(dataset) else buffer_size
        buffer = {
            'object_features': np.zeros((bs, 100, 512), dtype=np.float32),
            'relation_features': np.zeros((bs, 2500, 512),
                                          dtype=np.float32),
            'object_mask': np.zeros((bs, 100), dtype=np.int),
            'relation_mask': np.zeros((bs, 2500), dtype=np.int),
            'object_labels': np.zeros((bs, 100), dtype=np.int),
            'relation_labels': np.zeros((bs, 2500), dtype=np.int),
            'object_boxes': np.zeros((bs, 100, 4), dtype=np.float32),
            'relation_boxes': np.zeros((bs, 2500, 8), dtype=np.float32),
            'relation_pairs': np.zeros((bs, 2500, 2), dtype=np.int),
            'ids': [],
            'num_rels': 0
        }
        max_rels = 0
        for b in range(bs):
            image_data = dataset[start_+b]
            if image_data['image'].mode == 'L':
                image_data['image'] = image_data['image'].convert("RGB")
            image, _ = trans(image_data['image'], image_data['image'])
            boxes, rel_boxes = scene_parser(image.to(device))
            boxes, rel_boxes = boxes[0], rel_boxes[0]
            rel_labels = rel_boxes.get_field('scores').argmax(dim=1)
            indices = rel_labels.nonzero(as_tuple=True)
            object_features = scene_parser.rel_heads.rel_predictor.obj_predictor._input_value_hook[0]\
                .squeeze().detach().cpu().numpy()
            relation_features = scene_parser.rel_heads.rel_predictor.pred_predictor._input_value_hook[0][indices]\
                .squeeze().detach().cpu().numpy()
            num_obj = object_features.shape[0]
            num_rels = relation_features.shape[0]
            if num_rels > max_rels:
                max_rels = num_rels
            # add features to buffer
            buffer['object_features'][b, :num_obj] = object_features
            buffer['relation_features'][b, :num_rels] = relation_features
            buffer['object_labels'][b, :num_obj] = boxes.get_field('labels').detach().cpu().numpy()
            buffer['relation_labels'][b, :num_rels] = rel_labels[indices].detach().cpu().numpy()
            buffer['object_mask'][b, :num_obj] = 1
            buffer['relation_mask'][b, :num_rels] = 1
            buffer['object_boxes'][b, :num_obj] = boxes.bbox.detach().cpu().numpy()
            buffer['relation_boxes'][b, :num_rels] = rel_boxes.bbox[indices].detach().cpu().numpy()
            buffer['relation_pairs'][b, :num_rels] = rel_boxes.get_field('idx_pairs')[indices].detach().cpu().numpy()
            buffer['ids'].append(image_data['id'])
        # when buffer_obj is full, return it
        start_ += bs
        buffer['num_rels'] = max_rels
        yield buffer


def construct_sgg_hdf5(hdf5_path, pickle_path, image_folder, buffer_size=1):
    # os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    image_dataset = ImageDataset(image_folder)
    image_h5 = h5py.File(hdf5_path, 'w', libver='latest')
    # create general hdf5 setting
    comp = 9
    shuffle = True
    # Create the hdf5 datasets
    object_features = image_h5.create_dataset(name='object_features',
                                              dtype='f',
                                              shape=(len(image_dataset), 100, 512),
                                              maxshape=(len(image_dataset), 100, 512),
                                              chunks=(1, 100, 512),
                                              compression=comp,
                                              shuffle=shuffle)
    relation_features = image_h5.create_dataset(name='relation_features',
                                                dtype='f',
                                                shape=(len(image_dataset), 100, 512),
                                                maxshape=(len(image_dataset), 2500, 512),
                                                chunks=(1, 100, 512),
                                                compression=comp,
                                                shuffle=shuffle)
    object_features_mask = image_h5.create_dataset(name='object_mask',
                                                   dtype='uint8',
                                                   shape=(len(image_dataset), 100))
    relation_features_mask = image_h5.create_dataset(name='relation_mask',
                                                     dtype='uint8',
                                                     shape=(len(image_dataset), 100),
                                                     maxshape=(len(image_dataset), 2500),
                                                     chunks=(1, 2500),
                                                     compression=comp,
                                                     shuffle=shuffle)
    object_labels = image_h5.create_dataset(name='object_labels',
                                            dtype='i',
                                            shape=(len(image_dataset), 100))
    relation_labels = image_h5.create_dataset(name='relation_labels',
                                              dtype='i',
                                              shape=(len(image_dataset), 100),
                                              maxshape=(len(image_dataset), 2500),
                                              chunks=(1, 2500),
                                              compression=comp,
                                              shuffle=shuffle)
    object_boxes = image_h5.create_dataset(name='object_boxes',
                                           dtype='f',
                                           shape=(len(image_dataset), 100, 4),
                                           maxshape=(len(image_dataset), 100, 4),
                                           chunks=(1, 100, 4),
                                           compression=comp,
                                           shuffle=shuffle)
    relation_boxes = image_h5.create_dataset(name='relation_boxes',
                                             dtype='f',
                                             shape=(len(image_dataset), 100, 8),
                                             maxshape=(len(image_dataset), 2500, 8),
                                             chunks=(1, 2500, 8),
                                             compression=comp,
                                             shuffle=shuffle)
    relation_pair_idx = image_h5.create_dataset(name='relation_pair_idx',
                                                dtype='i',
                                                shape=(len(image_dataset), 100, 2),
                                                maxshape=(len(image_dataset), 2500, 2),
                                                chunks=(1, 2500, 2),
                                                compression=comp,
                                                shuffle=shuffle)
    image2sgg_feat_map = dict()
    i = 0
    for buffer in tqdm(collect_sgg_features(image_dataset, buffer_size=buffer_size), desc='collect sgg features',
                       total=int(np.ceil(len(image_dataset)/buffer_size))):
        bs = buffer['object_features'].shape[0]
        object_features[i:i+bs] = buffer['object_features']
        object_features_mask[i:i+bs] = buffer['object_mask']
        object_labels[i:i+bs] = buffer['object_labels']
        object_boxes[i:i+bs] = buffer['object_boxes']

        if buffer['num_rels'] > relation_labels.shape[1]:
            relation_features.resize(buffer['num_rels'], axis=1)
            relation_features_mask.resize(buffer['num_rels'], axis=1)
            relation_labels.resize(buffer['num_rels'], axis=1)
            relation_boxes.resize(buffer['num_rels'], axis=1)
            relation_pair_idx.resize(buffer['num_rels'], axis=1)
        relation_features[i:i+bs, :buffer['num_rels']] = buffer['relation_features'][:, :buffer['num_rels']]
        relation_features_mask[i:i+bs, :buffer['num_rels']] = buffer['relation_mask'][:, :buffer['num_rels']]
        relation_labels[i:i+bs, :buffer['num_rels']] = buffer['relation_labels'][:, :buffer['num_rels']]
        relation_boxes[i:i+bs, :buffer['num_rels']] = buffer['relation_boxes'][:, :buffer['num_rels']]
        relation_pair_idx[i:i+bs, :buffer['num_rels']] = buffer['relation_pairs'][:, :buffer['num_rels']]
        for id_ in buffer['ids']:
            image2sgg_feat_map[id_] = i
            i += 1
    image_h5.close()
    with open(pickle_path, 'wb') as f:
        pickle.dump(image2sgg_feat_map, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Captioning')
    # Add config file arguments
    parser.add_argument('--buffersize', default=1, type=int, help='how many images to store at a time')
    parser.add_argument('--sgg-cfg-file', default="configs/sgg/sgg.IMP_pretrained.yaml", type=str,
                        help='config for scene graph generator')
    parser.add_argument('--sgg-weight-file', default="weights/sg_imp_step_ckpt.pth", type=str,
                        help='weights for scene graph generator')
    # Parse the arguments
    args = parser.parse_args()
    sgg_cfg_file = args.sgg_cfg_file
    sgg_weight_file = args.sgg_weight_file
    train_data_file = 'train_scene-graph.hdf5'
    val_data_file = 'val_scene-graph.hdf5'
    train_indices_file = 'train_scene-graph_imgid2idx.pkl'
    val_indices_file = 'val_scene-graph_imgid2idx.pkl'
    train_ids_file = 'train_scene-graph_ids.pkl'
    val_ids_file = 'val_scene-graph_ids.pkl'
    train_imgs_path = 'data/train2014/'
    val_imgs_path = 'data/val2014/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    construct_sgg_hdf5(train_data_file, train_indices_file, train_imgs_path, buffer_size=args.buffersize)
    construct_sgg_hdf5(val_data_file, val_indices_file, val_imgs_path, buffer_size=args.buffersize)

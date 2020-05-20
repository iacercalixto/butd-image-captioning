import argparse
import h5py
import pickle
from PIL import Image
from torch.utils.data import Dataset
import utils as locutils
import numpy as np
import torch
from tqdm import tqdm
import os
import random
import yaml
import json
import cv2
from torchvision import transforms

#import factorizable net stuff
from lib import network
import lib.datasets as datasets
from lib.utils.FN_utils import get_model_name, group_features
import lib.utils.general_utils as utils
import models
from models.HDN_v2.utils import interpret_relationships

from lib.rpn_msr.anchor_target_layer import anchor_target_layer


class ImageDataset(Dataset):
    def __init__(self, image_dir, opts, image_set='test', dataset_option='normal', batch_size=1, use_region=False):
        self.imgids, self.imagepaths = locutils.load_imageinfo(image_dir)
        self.imgids = list(self.imgids)
        self.opts = opts
        self.use_region = use_region
        self._name = 'vg_' + dataset_option + '_' + image_set
        self.unknown_token='<unknown>'
        self.start_token='<start>'
        self.end_token='<end>'
        self._set_option = dataset_option
        # self._batch_size = batch_size
        self._image_set = image_set
        self._data_path = os.path.join(self.opts['dir'], 'images')
        # load category names and annotations
        annotation_dir = os.path.join(self.opts['dir'])
        cats = json.load(open(os.path.join(annotation_dir, 'categories.json')))
        dictionary = json.load(open(os.path.join(annotation_dir, 'dict.json')))
        inverse_weight = json.load(open(os.path.join(annotation_dir, 'inverse_weight.json')))
        self.idx2word = dictionary['idx2word']
        self.word2idx = dictionary['word2idx']
        dict_len = len(dictionary['idx2word'])
        self.idx2word.append(self.unknown_token)
        self.idx2word.append(self.start_token)
        self.idx2word.append(self.end_token)
        self.word2idx[self.unknown_token] = dict_len
        self.word2idx[self.start_token] = dict_len + 1
        self.word2idx[self.end_token] = dict_len + 2
        self.voc_sign = {'start': self.word2idx[self.start_token],
                         'null': self.word2idx[self.unknown_token],
                         'end': self.word2idx[self.end_token]}

        self._object_classes = tuple(['__background__'] + cats['object'])
        self._predicate_classes = tuple(['__background__'] + cats['predicate'])
        self._object_class_to_ind = dict(zip(self.object_classes, xrange(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, xrange(self.num_predicate_classes)))
        self.inverse_weight_object = torch.ones(self.num_object_classes)
        # for idx in xrange(1, self.num_object_classes):
        #     self.inverse_weight_object[idx] = inverse_weight['object'][self._object_classes[idx]]
        for idx in xrange(1, self.num_object_classes):
            self.inverse_weight_object[idx] = inverse_weight['object'][idx]
        self.inverse_weight_object = self.inverse_weight_object / self.inverse_weight_object.min()
        # print self.inverse_weight_object
        self.inverse_weight_predicate = torch.ones(self.num_predicate_classes)
        # for idx in xrange(1, self.num_predicate_classes):
        #     self.inverse_weight_predicate[idx] = inverse_weight['predicate'][self._predicate_classes[idx]]
        for idx in xrange(1, min(self.num_predicate_classes, len(inverse_weight['predicate']))):
            self.inverse_weight_predicate[idx] = inverse_weight['predicate'][idx]
        self.inverse_weight_predicate = self.inverse_weight_predicate / self.inverse_weight_predicate.min()
        # print self.inverse_weight_predicate
        ann_file_name = {'vg_normal_train': 'train.json',
                           'vg_normal_test': 'test.json',
                           'vg_small_train': 'train_small.json',
                           'vg_small_test': 'test_small.json',
                           'vg_fat_train': 'train_fat.json',
                           'vg_fat_test': 'test_small.json'}

        # ann_file_path = os.path.join(annotation_dir, ann_file_name[self.name])
        # self.annotations = json.load(open(ann_file_path))
        # self.max_size = 11  # including the <end> token excluding <start> token
        # self.tokenize_annotations(self.max_size)


        # image transformation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.cfg_key = image_set
        self._feat_stride = None
        self._rpn_opts = None

    def __getitem__(self, index):
        imgid = self.imgids[index]
        image_path = self.imagepaths[imgid]

        # Sample random scales to use for each image in this batch
        item = {'rpn_targets': {}, 'id': imgid}

        target_scale = self.opts[self.cfg_key]['SCALES'][np.random.randint(0, high=len(self.opts[self.cfg_key]['SCALES']))]
        img = cv2.imread(image_path)
        img_original_shape = img.shape
        item['path'] = image_path
        img, im_scale = self._image_resize(img, target_scale, self.opts[self.cfg_key]['MAX_SIZE'])
        # restore the [image_height, image_width, scale_factor, max_size]
        item['image_info'] = np.array([img.shape[0], img.shape[1], im_scale,
                    img_original_shape[0], img_original_shape[1]], dtype=np.float)
        item['visual'] = Image.fromarray(img)

        if self.transform is not None:
            item['visual'] = self.transform(item['visual'])

        return item

    @staticmethod
    def collate(items):
        batch_item = {}
        for key in items[0]:
            if key == 'visual':
                batch_item[key] = [x[key].unsqueeze(0) for x in items]
            #     out = None
            #     # If we're in a background process, concatenate directly into a
            #     # shared memory tensor to avoid an extra copy
            #     numel = sum([x[key].numel() for x in items])
            #     storage = items[0][key].storage()._new_shared(numel)
            #     out = items[0][key].new(storage)
            #     batch_item[key] = torch.stack([x[key] for x in items], 0, out=out)
            elif key == 'rpn_targets':
                batch_item[key] = {}
                for subkey in items[0][key]:
                    batch_item[key][subkey] = [x[key][subkey] for x in items]
            elif items[0][key] is not None:
                batch_item[key] = [x[key] for x in items]

        return batch_item


    def __len__(self):
        # return len(self.annotations)
        return len(self.imgids)


    @property
    def voc_size(self):
        return len(self.idx2word)


    def get_regions(self, idx, length_constraint=50):
        boxes = np.array([reg['box'] for reg in self.annotations[idx]['regions']]).astype(np.float32)
        text = []
        mask_ = np.ones(boxes.shape[0], dtype=np.bool)
        for i in range(len(self.annotations[idx]['regions'])):
            reg = self.annotations[idx]['regions'][i]
            if len(reg['phrase']) > length_constraint:
                mask_[i] = False
                continue
            text.append(self.untokenize_single_sentence(reg['phrase']))
        boxes = boxes[mask_]
        return boxes, text

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self.annotations[index]['path']
        image_path = os.path.join(self._data_path, file_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def tokenize_annotations(self, max_size):

        counter = 0
        # print 'Tokenizing annotations...'
        for im in self.annotations:
            for obj in im['objects']:
                obj['class'] = self._object_class_to_ind[obj['class']]
            for rel in im['relationships']:
                rel['predicate'] = self._predicate_class_to_ind[rel['predicate']]
            for region in list(im['regions']):
                region['phrase'] = [self.word2idx[word] if word in self.word2idx else self.word2idx[self.unknown_token] \
                                        for word in (['<start>'] + region['phrase'] + ['<end>'])]
                if len(region['phrase']) < 5 or len(region['phrase']) >= max_size:
                    im['regions'].remove(region)


    def tokenize_sentence(self, sentence):
        return [self.word2idx[word] for word in (sentence.split() + ['<end>'])]

    def untokenize_single_sentence(self, sentence):
        word_sentence = []
        for idx in sentence:
            if idx == self.voc_sign['end']:
                break
            if idx == self.voc_sign['null'] or idx == self.voc_sign['start']:
                continue
            else:
                word_sentence.append(self.idx2word[idx])
        return ' '.join(word_sentence)

    def untokenize_sentence(self, sentence):
        result = []
        keep_id = []
        for i in range(sentence.shape[0]):
            word_sentence = []
            for idx in sentence[i]:
                if idx == self.voc_sign['end']:
                    break
                else:
                    word_sentence.append(self.idx2word[idx])
            if len(word_sentence) > 0:
                result.append(' '.join(word_sentence))
                keep_id.append(i)
        return result, np.array(keep_id, dtype=np.int)


    def _image_resize(self, im, target_size, max_size):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes
#
# class ImageDataset(Dataset):
#     def __init__(self, image_dir):
#         super(ImageDataset, self).__init__()
#         self.imgids, self.imagepaths = locutils.load_imageinfo(image_dir)
#         self.imgids = list(self.imgids)
#
#     def __len__(self):
#         return len(self.imgids)
#
#     def __getitem__(self, index):
#         """
#         :param index: The index of the item to retrieve
#         :return: One data pair (image and caption).
#         """
#         imgid = self.imgids[index]
#         image_path = self.imagepaths[imgid]
#         image = locutils.pil_loader(image_path)  # self.load_image(image_path)
#         return {
#             'image': image,
#             'id': imgid,
#             'image_file': os.path.basename(image_path),
#         }


def get_input_hook(module, input, output):
    """
    A hook to assign to modules to get there output later
    :param module: the module
    :param input: the input of the module
    :param output: the output of the module
    """
    setattr(module, "_input_value_hook", input)


def get_output_hook(module, input, output):
    """
    A hook to assign to modules to get there output later
    :param module: the module
    :param input: the input of the module
    :param output: the output of the module
    """
    setattr(module, "_output_value_hook", output)


def get_dataset_model(image_folder):
    # initialize the factorizableNet stuff
    # Set options
    options = {'data': {'batch_size': 1}}

    if args.path_opt is not None:
        with open(os.path.join(args.path2fact, args.path_opt), 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)
        with open(os.path.join(args.path2fact, options['data']['opts']), 'r') as f:
            data_opts = yaml.load(f)
            options['data']['dataset_version'] = data_opts.get('dataset_version', None)
            options['opts'] = data_opts
    lr = options['optim']['lr']
    options = get_model_name(options)
    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)
    train_set = ImageDataset(image_folder, data_opts, 'train',
                           dataset_option=options['data'].get('dataset_option', 'normal'),
                           use_region=options['data'].get('use_region', False), )
    dataset = ImageDataset(image_folder, data_opts, 'test', dataset_option=options['data'].get('dataset_option', 'normal'),
                           use_region=options['data'].get('use_region', False), )
    options['model']['rpn_opts'] = os.path.join(args.path2fact, options['model']['rpn_opts'])
    model = getattr(models, options['model']['arch'])(train_set, opts=options['model'])
    return dataset, model

def collect_sgg_features(dataset, model, buffer_size=1):
    vgg_features_fix, vgg_features_var, rpn_features, hdn_features, mps_features = group_features(model, has_RPN=True)
    network.set_trainable(model, False)
    # Setting the state of the training model
    model.cuda()
    top_Ns = [50, 100]
    # object features
    model.mps_list[-1].register_forward_hook(get_output_hook)
    # relation features
    model.phrase_inference.register_forward_hook(get_output_hook)
    model.eval()

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
            sample = dataset[start_ + b]
            sample = dataset.collate([sample])
            assert len(sample['visual']) == 1
            input_visual = sample['visual'][0].cuda()
            image_info = sample['image_info']
            # Forward pass
            object_result, predicate_result = model.forward_eval(input_visual, image_info)
            object_features = model.mps_list[-1]._output_value_hook[0]
            relation_features = model.phrase_inference._output_value_hook[0]
            obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, \
            subject_boxes, object_boxes, predicate_inds, \
            sub_assignment, obj_assignment, total_score = interpret_relationships(
                object_result[0], object_result[1], object_result[2], predicate_result[0], predicate_result[1],
                image_info, top_N=max(top_Ns), reranked_score=object_result[3],
                nms=-1,  # nms is an argument, left default according to README
                triplet_nms=0.4)  # default triplet_nms is 0.4, unchanged according to README

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
            buffer['ids'].append(sample['id'])
        # when buffer_obj is full, return it
        start_ += bs
        buffer['num_rels'] = max_rels
        yield buffer


def construct_sgg_hdf5(hdf5_path, pickle_path, image_folder, buffer_size=1):
    # os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    image_dataset, model = get_dataset_model(image_folder)
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
    for buffer in tqdm(collect_sgg_features(image_dataset, model, buffer_size=buffer_size), desc='collect sgg features',
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
    parser = argparse.ArgumentParser('Collect FactorizableNet features')
    parser.add_argument('--path2fact', default='../../FactorizableNet/', type=str,
                        help='path to a factorizablenet code base')
    parser.add_argument('--path_opt', default='options/models/VG-DR-Net.yaml', type=str, help='path to a yaml options file')
    parser.add_argument('--workers', type=int, default=4, help='#idataloader workers')
    # model init
    parser.add_argument('--pretrained_model', default='output/trained_models/Model-VG-DR-Net.h5',
                        type=str, help='path to pretrained_model')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation and test set')
    parser.add_argument('--buffersize', default=1, type=int, help='how many images to store at a time')
    parser.add_argument('--seed', default=42, type=int, help='for randomness')
    args = parser.parse_args()

    train_data_file = 'train_scene-graph.hdf5'
    val_data_file = 'val_scene-graph.hdf5'
    train_indices_file = 'train_scene-graph_imgid2idx.pkl'
    val_indices_file = 'val_scene-graph_imgid2idx.pkl'
    train_ids_file = 'train_scene-graph_ids.pkl'
    val_ids_file = 'val_scene-graph_ids.pkl'
    train_imgs_path = '../data/train2014/'
    val_imgs_path = '../data/val2014/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    construct_sgg_hdf5(train_data_file, train_indices_file, train_imgs_path, buffer_size=args.buffersize)
    # construct_sgg_hdf5(val_data_file, val_indices_file, val_imgs_path, buffer_size=args.buffersize)

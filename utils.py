import os
import numpy as np
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle
import dgl


def collate_fn(batch):
    """ Collate function to be used when iterating captioning datasets.
        Only use with batch size == 1.
    """
    if len(tuple(zip(*batch))) == 4:
        image_features, caps, caplens, orig_caps = zip(*batch)
        r = (torch.stack(image_features), torch.stack(caps), torch.stack(caplens), orig_caps[0])
    else:
        (obj, rel, caps, caplens, orig_caps, obj_mask, rel_mask, pair_idx) = zip(*batch)
        r = (torch.stack(obj), torch.stack(rel), torch.stack(caps), torch.stack(caplens), orig_caps[0],
             torch.stack(obj_mask), torch.stack(rel_mask), torch.stack([torch.as_tensor(p) for p in pair_idx]))
    return r


def create_input_files(dataset,karpathy_json_path,captions_per_image, min_word_freq,output_folder,max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset. Since bottom up features only available for coco, we use only coco
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    with open(os.path.join(output_folder,'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)
        
    with open(os.path.join(output_folder,'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)
    
    # Read image paths and captions for each image
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()
    
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue
        
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])

        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t",train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v",val_data[image_id]))
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v",val_data[image_id]))
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v",val_data[image_id]))
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_det) == len(train_image_captions)
    assert len(val_image_det) == len(val_image_captions)
    assert len(test_image_det) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
   
    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
        
    
    for impaths, imcaps, split in [(train_image_det, train_image_captions, 'TRAIN'),
                                   (val_image_det, val_image_captions, 'VAL'),
                                   (test_image_det, test_image_captions, 'TEST')]:
        orig_captions = []
        enc_captions = []
        caplens = []
        
        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image
            
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                orig_captions.append(c)
                caplens.append(c_len)
        
        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_ORIG_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(orig_captions, j)

        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
    
    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_det, j)
        
    with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_det, j)
        
    with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_det, j)


def create_scene_graph_input_files(dataset, karpathy_json_path, output_folder):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset. Since bottom up features only available for coco, we use only coco
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    with open(os.path.join(output_folder, 'train_scene-graph_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)
    with open(os.path.join(output_folder, 'val_scene-graph_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)

    # Read image paths and captions for each image
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()
    for img in data['images']:
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])

        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t", train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v", val_data[image_id]))
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v", val_data[image_id]))
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v", val_data[image_id]))

    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN_SCENE_GRAPHS_FEATURES_'+dataset+'.json'), 'w') as j:
        json.dump(train_image_det, j)

    with open(os.path.join(output_folder, 'VAL_SCENE_GRAPHS_FEATURES_'+dataset+'.json'), 'w') as j:
        json.dump(val_image_det, j)

    with open(os.path.join(output_folder, 'TEST_SCENE_GRAPHS_FEATURES_'+dataset+'.json'), 'w') as j:
        json.dump(test_image_det, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                    stopping_metric, metric_score, tracking, is_best, outdir, best_epoch):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param decoder: decoder model
    :param decoder_optimizer: optimizer to update decoder's weights
    :param stopping_metric: metric to check stopping
    :param metric_score: validation score for this epoch
    :param tracking: dict with list of eval scores and possible test scores
    :param is_best: is this checkpoint the best so far?
    :param outdir: where to store all the files
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'stopping_metric': stopping_metric,
             'metric_score': metric_score,
             'decoder': decoder,
             'decoder_optimizer': decoder_optimizer,
             'tracking': tracking,
             'best_epoch': best_epoch}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, os.path.join(outdir, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(outdir, 'BEST_' + str(epoch) + '_' + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def create_captions_file(im_ids, sentences_tokens, file):
    preds = []
    if sentences_tokens[0] != [] and isinstance(sentences_tokens[0][0], list):
        imgs = []
        cap_id = 0
        for im_id, captions in zip(im_ids, sentences_tokens):
            for caption in captions:
                imgs.append({'id': im_id})
                pred = dict()
                pred['id'] = cap_id
                pred['image_id'] = im_id
                pred['caption'] = ' '.join(caption)
                cap_id += 1
                preds.append(pred)
        preds = {'images': imgs, 'annotations': preds}
    else:
        for im_id, caption in zip(im_ids, sentences_tokens):
            prediction = dict()
            prediction['image_id'] = im_id
            prediction['caption'] = ' '.join(caption)
            preds.append(prediction)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(preds, f)


def create_batched_graphs(o, om, r, rm, pairs, beam_size=1):
    bsz = o.size(0)
    graphs = []
    pairs = pairs.detach().cpu().numpy()
    for b in range(bsz):
        for k in range(beam_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(num=om[b].sum())
            graph.ndata['F_n'] = o[b, om[b]]
            cpu_mask = rm[b].detach().cpu()
            graph.add_edges(pairs[b][cpu_mask, 0], pairs[b][cpu_mask, 1])
            graph.edata['F_e'] = r[b, rm[b]]
            graphs.append(graph)
    b_graphs = dgl.batch(graphs)
    return b_graphs

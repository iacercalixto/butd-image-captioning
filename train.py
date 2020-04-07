import argparse
import shutil
import time
import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DecoderWithAttention
from datasets import CaptionDataset
from utils import collate_fn, save_checkpoint, AverageMeter, adjust_learning_rate, accuracy, create_captions_file
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from eval import beam_evaluate_butd

word_map = word_map_inv = None


def main():
    """
    Training and validation.
    """

    global word_map, word_map_inv

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        # create inverse word map
    word_map_inv = {v: k for k, v in word_map.items()}

    # Initialize / load checkpoint
    if args.checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=args.dropout)

        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
        tracking = {'eval': [], 'test': None}
        start_epoch = 0
        best_epoch = -1
        epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation
        best_stopping_score = 0.  # stopping_score right now
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        args.stopping_metric = checkpoint['stopping_metric'],
        best_stopping_score = checkpoint['metric_score'],
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer'],
        tracking = checkpoint['tracking'],
        best_epoch = checkpoint['best_epoch']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(CaptionDataset(args.data_folder, args.data_name, 'TRAIN'),
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(CaptionDataset(args.data_folder, args.data_name, 'VAL'),
                                             collate_fn=collate_fn,
                                             # use our specially designed collate function with valid/test only
                                             batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    #    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # # One epoch's validation
        # recent_bleu4 = validate(val_loader=val_loader,
        #                         decoder=decoder,
        #                         criterion_ce=criterion_ce,
        #                         criterion_dis=criterion_dis,
        #                         epoch=epoch)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              criterion_dis=criterion_dis,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_results = validate(val_loader=val_loader,
                                  decoder=decoder,
                                  criterion_ce=criterion_ce,
                                  criterion_dis=criterion_dis,
                                  epoch=epoch)
        tracking['eval'] = recent_results
        recent_stopping_score = recent_results[args.stopping_metric]

        # Check if there was an improvement
        is_best = recent_stopping_score > best_stopping_score
        best_stopping_score = max(recent_stopping_score, best_stopping_score)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            best_epoch = epoch

        # Save checkpoint
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                        args.stopping_metric, best_stopping_score, tracking, is_best, args.outdir, best_epoch)

    # if needed, run an beamsearch evaluation on the test set
    if args.test_at_end:
        checkpoint_file = 'BEST_' + str(best_epoch) + '_' + 'checkpoint_' + args.data_name + '.pth.tar'
        results = beam_evaluate_butd(args.data_name, checkpoint_file, args.data_folder, args.beam_size, args.outdir)
        tracking['test'] = results
    with open(os.path.join(args.outdir, 'TRACKING.'+args.data_name+'.pkl'), 'wb') as f:
        pickle.dump(tracking, f)


def train(train_loader, decoder, criterion_ce, criterion_dis, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # Max-pooling across predicted words across time steps for discriminative supervision
        scores_d = scores_d.max(1)[0]

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0), scores_d.size(1)).to(device)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:, :length - 1] = targets[:, :length - 1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
        #scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss_d = criterion_dis(scores_d, targets_d.long())
        loss_g = criterion_ce(scores, targets)
        loss = loss_g + (10 * loss_d)

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, decoder, criterion_ce, criterion_dis, epoch):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        # for i, (imgs, caps, caplens,allcaps) in enumerate(val_loader):
        for i, (imgs, caps, caplens, orig_caps) in enumerate(val_loader):

            if i % 5 != 0:
                # only decode every 5th caption, starting from idx 0.
                # this is because the iterator iterates over all captions in the dataset, not all images.
                if i % args.print_freq_val == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                    batch_time=batch_time,
                                                                                    loss=losses, top5=top5accs))
                continue

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0), scores_d.size(1)).to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:, :length - 1] = targets[:, :length - 1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
            #scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            #targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss_d = criterion_dis(scores_d, targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss = loss_g + (10 * loss_d)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq_val == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            assert (len(sort_ind) == 1), "Cannot have batch_size>1 for validation."
            # a reference is a list of lists:
            # [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['a', 'cat', 'on', 'the', 'mat']]
            references.append(orig_caps)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            preds_idxs_no_pads = list()
            for j, p in enumerate(preds):
                preds_idxs_no_pads.append(preds[j][:decode_lengths[j]])  # remove pads
                preds_idxs_no_pads = list(map(lambda c: [w for w in c if w not in {word_map['<start>'],
                                                                                   word_map['<pad>']}],
                                              preds_idxs_no_pads))
            temp_preds = list()
            # remove <start> and pads and convert idxs to string
            for hyp in preds_idxs_no_pads:
                temp_preds.append([])
                for w in hyp:
                    assert (not w == word_map['pad']), "Should have removed all pads."
                    if not w == word_map['<start>']:
                        temp_preds[-1].append(word_map_inv[w])
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # bleu4 = corpus_bleu(references, hypotheses)
    # bleu4 = round(bleu4, 4)
    # compute the metrics
    hypotheses_file = os.path.join(args.outdir, 'hypotheses', 'Epoch{:0>3d}.Hypotheses.json'.format(epoch))
    references_file = os.path.join(args.outdir, 'references', 'Epoch{:0>3d}.References.json'.format(epoch))
    create_captions_file(range(len(hypotheses)), hypotheses, hypotheses_file)
    create_captions_file(range(len(references)), references, references_file)
    coco = COCO(references_file)
    # add the predicted results to the object
    coco_results = coco.loadRes(hypotheses_file)
    # create the evaluation object with both the ground-truth and the predictions
    coco_eval = COCOEvalCap(coco, coco_results)
    # change to use the image ids in the results object, not those from the ground-truth
    coco_eval.params['image_id'] = coco_results.getImgIds()
    # run the evaluation
    coco_eval.evaluate(verbose=False, metrics=['bleu', 'meteor', 'rouge', 'cider'])
    # Results contains: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"
    results = coco_eval.eval
    results['loss'] = losses.avg
    results['top5'] = top5accs.avg

    for k, v in results.items():
        print(k+':\t'+str(v))
    # print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}, CIDEr - {cider}\n'
    #       .format(loss=losses, top5=top5accs, bleu=round(results['Bleu_4'], 4), cider=round(results['CIDEr'], 1)))
    return results


if __name__ == '__main__':
    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "loss", "top5"]
    parser = argparse.ArgumentParser('Image Captioning')
    # Add config file arguments
    parser.add_argument('--data_folder', default='final_dataset', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--print_freq', default=100, type=int, help='print training stats every __ batches')
    parser.add_argument('--print_freq_val', default=1000, type=int, help='print validation stats every __ batches')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, None if none')
    parser.add_argument('--outdir', default='outputs', type=str,
                        help='path to location where to save outputs. Empty for current working dir')
    parser.add_argument('--workers', default=1, type=int,
                        help='for data-loading; right now, only 1 works with h5py '
                             '(OUTDATED, h5py can have multiple reads, right)')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--seed', default=42, type=int, help='The random seed that will be used.')
    parser.add_argument('--emb_dim', default=1024, type=int, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', default=1024, type=int, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', default=1024, type=int, help='dropout probability')
    parser.add_argument('--dropout', default=0.5, type=float, help='dimension of decoder RNN')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--architecture', default='bottomup_topdown', type=str, choices=['bottomup_topdown'],
                        help='which architecture to use')
    parser.add_argument('--stopping_metric', default='Bleu_4', type=str, choices=metrics,
                        help='which metric to use for early stopping')
    parser.add_argument('--test_at_end', default=True, type=bool, help='If there should be tested on the test split')
    parser.add_argument('--beam_size', default=5, type=int, help='If test at end, beam size to use for testing.')
    # Parse the arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    # setup initial stuff for reproducability
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size otherwise lot of computational overhead
    cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)

    # Training parameters
    args.outdir = os.path.join(args.outdir,
                               args.architecture,
                               'batch_size-{bs}_epochs-{ep}_dropout-{drop}'.format(bs=args.batch_size, ep=args.epochs,
                                                                                   drop=args.dropout),
                               'emb-{emb}_att-{att}_dec-{dec}'.format(emb=args.emb_dim, att=args.attention_dim,
                                                                      dec=args.decoder_dim),
                               'seed-{}'.format(args.seed))
    if os.path.exists(args.outdir) and args.checkpoint is None:
        answer = input("\n\t!! WARNING !! \nthe specified --outdir already exists, "
                       "probably from previous experiments: \n\t{}\n"
                       "Ist it okay to delete it and all its content for current experiment? "
                       "(Yes/No) .. ".format(args.outdir))
        if answer.lower() == "yes":
            print('SAVE_DIR will be deleted ...')
            shutil.rmtree(args.outdir)
            os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
            os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
        else:
            print('To run this experiment and preserve the other one, change some settings, like the --seed.\n'
                  '\tExiting Program...')
            exit(0)
    elif os.path.exists(args.outdir) and args.checkpoint is not None:
        print('continueing from checkpoint {} in {}...'.format(args.checkpoint, args.outdir))
    elif not os.path.exists(args.outdir) and args.checkpoint is not None:
        print('set a checkpoint to continue from, but the save directory from --outdir {} does not exist. '
              'setting --checkpoint to None'.format(args.outdir))
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    main()

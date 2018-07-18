'''
Based on Open-ReID framework. https://github.com/Cysu/open-reid
'''
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from collections import OrderedDict
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
sys.path.append(osp.abspath(osp.abspath(__file__) + "/../.."))
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import DeepLoss
from reid.trainers import Trainer
from reid.evaluators import pairwise_distance, evaluate_all
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.meters import AverageMeter
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, batch_size, num_instances, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(256, 128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(256, 128),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, 'num_instances should divide batch_size'

    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.batch_size, args.num_instances, args.workers)

    # Create model
    print('num_features: %d, features:%d ' % (args.num_features, num_classes))
    model = models.create("deepperson", num_features=args.num_features,
                          dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluate(model, val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluate(model, test_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    criterion = DeepLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, add_soft=args.add_soft)
        if epoch < args.start_save:
            continue
        top1 = evaluate(model, val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with last model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluate(model, test_loader, dataset.query, dataset.gallery)


def evaluate(model, data_loader, query, gallery):
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()

    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        # normal imgs
        normal_inputs = Variable(imgs.cuda(), volatile=True)
        normal_outputs = model(normal_inputs)[-1]
        normal_outputs = normal_outputs.data.cpu()

        # fliped imgs
        inv_idx = torch.arange(imgs.size(3) - 1, -1, -1).long()  # N x C x H x W
        flip_imgs = imgs.index_select(3, inv_idx)
        flip_inputs = Variable(flip_imgs.cuda(), volatile=True)
        flip_outputs = model(flip_inputs)[-1]
        flip_outputs = flip_outputs.data.cpu()

        outputs = F.normalize(normal_outputs + flip_outputs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

    distmat = pairwise_distance(features, query, gallery)
    return evaluate_all(distmat, query, gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepPerson")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('-K', '--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 16")
    # model
    parser.add_argument('--num-features', type=int, default=2048)

    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--add-soft', type=int, default=30, help="add softmax loss after add_soft epochs")
    # optimizer
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start-save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)

    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(osp.expanduser('~'), 'datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

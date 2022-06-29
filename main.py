import argparse
import os
import time
import pickle
#import cPickle as pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import Model
from data_load import ImageList
import random
import warnings
import logging
import numpy as np
import json


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='N', help='GPU NO. (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--split', default=0, type=int)
args = parser.parse_args()

name = 'traffic_net'
ckpts = 'ckpts/cdnn/'  # save model
if not os.path.exists(ckpts):
    os.makedirs(ckpts)

log_file = os.path.join(ckpts + "/train_log_%s.txt" % (name, ))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


def main():
    #global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()

    params = model.parameters()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(params, args.lr,
                                 weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # traffic_frames root
    root = r'/home/yupengcheng/project/CDNN/CDNN-traffic-saliency/traffic_frames/'

    train_imgs = [json.loads(line) for line in open(root + 'train.json')]

    valid_imgs = [json.loads(line) for line in open(root + 'valid.json')]

    test_imgs = [json.loads(line) for line in open(root + 'test.json')]
    # print len(train_imgs),train_imgs[0]
    # # print train_imgs
    # exit(0)

    train_loader = DataLoader(
        ImageList(root, train_imgs, for_train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
        pin_memory=False)

    valid_loader = DataLoader(
        ImageList(root, valid_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=False)
    test_loader = DataLoader(
        ImageList(root, test_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    criterion = nn.BCELoss().cuda()

    logging.info(
        '-------------- New training session, LR = %f ----------------' % (args.lr, ))
    logging.info('-- length of training images = %d--length of valid images = %d--' %
                 (len(train_imgs), len(valid_imgs)))
    logging.info('-- length of test images = %d--' % (len(test_imgs)))
    best_loss = float('inf')
    file_name = os.path.join(ckpts, 'model_best_%s.tar' % (name, ))
    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch)
        # print train_loss
        # exit(0)
        # evaluate on validation set
        valid_loss = validate(
            valid_loader, model, criterion)

        # remember best lost and save checkpoint
        best_loss = min(valid_loss, best_loss)
        file_name_last = os.path.join(
            ckpts, 'model_epoch_%d.tar' % (epoch + 1, ))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
        }, file_name_last)

        if valid_loss == best_loss:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, file_name)

        msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f}'.format(
            epoch+1, train_loss, valid_loss)
        logging.info(msg)

    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state_dict'])
    outputs, targets = predict(test_loader, model)
    np.savez(ckpts + 'test.npz', p=outputs, t=targets)
    with open(ckpts + 'test.pkl', 'wb') as f:
        pickle.dump(test_imgs, f)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    # print('train_loader[0]',train_loader[0])
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # input = input.cuda(async=True)
        # target = target.cuda(async=True)
        #print('shape of input',input.shape)
        #print('shape of target', target.shape)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # m = nn.Sigmoid()
        # loss = criterion(m(output), m(target_var))
        # print('shape of output',output.shape)
        # print('shape of target_var',target_var.shape)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 1000 == 0:
            msg = 'Training Epoch {:03d}  Iter {:03d} Loss {:.6f} in {:.3f}s'.format(
                epoch+1, i+1, losses.avg, time.time() - start)
            start = time.time()
            logging.info(msg)
            print(msg)

    return losses.avg


def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    for i, (input, target) in enumerate(valid_loader):
        input = input.cuda()
        target = target.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        msg = 'Validating Iter {:03d} Loss {:.6f} in {:.3f}s'.format(
            i+1, losses.avg, time.time() - start)
        start = time.time()
        # logging.info(msg)
        print(msg)

    return losses.avg


def predict(valid_loader, model):

    # switch to evaluate mode
    model.eval()

    targets = []
    outputs = []

    for i, (input, target) in enumerate(valid_loader):

        targets.append(target.numpy().squeeze(1))

        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = model(input_var)
        outputs.append(output.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets


def test_set(test_loader, model):
    model.eval()
    #inputs = []
    #targets = []
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        p_output = nn.functional.softmax(target, dim=1)
    return p_output


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

import time
import torch
import math
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def train(train_loader, model, criterion, epoch, optimizer, lr_scheduler, device, batch_split_size, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_loader (torch.utils.data.DataLoader): The trainset dataloader
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        lr_scheduler (Training.LearningRateScheduler): class implementing learning rate schedules
        device (str): device name where data is transferred to
        batch_split_size (int): size of smaller split of batch to
            calculate sequentially if too little memory is available
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and batch_size (int).
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    factor = args.batch_size // batch_split_size
    last_batch = int(math.ceil(len(train_loader.dataset) / float(batch_split_size)))
    optimizer.zero_grad()

    for i, (inp, target) in enumerate(train_loader):
        inp, target = inp.to(device), target.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust the learning rate
        if i % factor == 0:
            lr_scheduler.adjust_learning_rate(optimizer, i // factor + 1)

        # compute output
        output = model(inp)

        loss = criterion(output, target) * inp.size(0) / float(args.batch_size)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item()*float(args.batch_size)/inp.size(0), inp.size(0))
        top1.update(prec1.item(), inp.size(0))
        top5.update(prec5.item(), inp.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if (i + 1) % factor == 0 or i == (last_batch - 1):
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq*factor) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(' * Train: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg
import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def validate(val_loader, model, criterion, epoch, device, args):
    """
    Evaluates/validates the model

    Parameters:
        val_loader (torch.utils.data.DataLoader): The validation or testset dataloader
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int).

    Returns:
        float: top1 accuracy
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            # compute output
            output = model(inp)

            # compute loss
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))
            top5.update(prec5.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validate: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch+1, i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Validation: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

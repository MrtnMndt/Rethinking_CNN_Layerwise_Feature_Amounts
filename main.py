########################
# Importing libraries
########################
# Custom library
import lib.Models.architectures as architectures
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit
from lib.cmdparser import parser
from lib.Models.skew_normal_models import SkewNormalModels
from lib.Utility.utils import GPUMem
from lib.Training.learning_rate_scheduling import LearningRateScheduler
from lib.Training.train import train
from lib.Training.validate import validate

# Logging to csv
import pandas as pd

# System libraries
import os
from time import gmtime, strftime

# Torch libraries
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


def main():
    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    num_GPUs = torch.cuda.device_count()

    # If save directory for runs doesn't exist then create it
    if not os.path.exists('runs'):
        os.mkdir('runs')

    # Create a time-stamped save path for individual experiment
    save_path = 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + \
                ';' + args.dataset + ';' + args.architecture
    os.mkdir(save_path)

    # List of values to log to csv
    columns_list = ['Filters', 'Parameters', 'Mean', 'Variance', 'Skew', 'BestVal',
                    'BestValsTrain', 'BestEpoch', 'LastValPrec', 'LastTrainPrec',
                    'AllTrain', 'AllVal']
    df = pd.DataFrame(columns=columns_list)

    # Dataset loading
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)

    # get the amount of color channels in the input images
    net_input, _ = next(iter(dataset.train_loader))
    num_colors = net_input.size(1)

    # import model from architectures class
    net_init_method = getattr(architectures, args.architecture)

    # Get the parameters for all valid skewed models
    SNModels = SkewNormalModels(depth=args.vgg_depth, num_classes=dataset.num_classes, patch_size=args.patch_size)
    skew_model_params = SNModels.get_valid_models()
    print("Total number of models: ", len(skew_model_params["filters"]))

    # Weight-init method
    WeightInitializer = WeightInit(args.weight_init)

    # Optionally resume a previous experiment
    current_id = args.resume_model_id
    for i in range(len(skew_model_params["filters"]) - current_id):
        print("Model filters: ", skew_model_params["filters"][i + current_id])
        print("Model parameters: ", skew_model_params["total_params"][i + current_id],
              " mean: ", skew_model_params["means"][i + current_id],
              " var: ", skew_model_params["vars"][i + current_id],
              " skew: ", skew_model_params["skews"][i + current_id])

        model = net_init_method(device, dataset.num_classes, num_colors, args,
                                skew_model_params["filters"][i + current_id], custom_filters=True)

        # Parallel container for multi GPU use and cast to available device
        model = torch.nn.DataParallel(model).to(device)
        print(model)

        # Initialize the weights of the model
        print("Initializing networks with: " + args.weight_init)
        WeightInitializer.init_model(model)

        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

        # Initialize SGDWR learning rate scheduler
        lr_scheduler = LearningRateScheduler(args.lr_wr_epochs, len(dataset.train_loader.dataset), args.batch_size,
                                             args.learning_rate, args.lr_wr_mul, args.lr_wr_min)

        # Get estimated GPU memory usage of the model and split batch if too little memory is available
        if torch.cuda.is_available():
            GPUMemory = GPUMem(torch.cuda.is_available())
            print('available:{}'.format(
                (GPUMemory.total_mem - GPUMemory.total_mem * GPUMemory.get_mem_util()) / 1024.))
            print('required per gpu with buffer: {}'.format((4. / float(num_GPUs) * model.module.gpu_usage) + 1.))

            # calculate smaller chunk size to split batch into sequential computations
            mem_scale_factor = 4.0  # TODO: WEIRD factor... why is this necessary and where does it come from?
            # TODO: the + 1 Gb should be taken from the cache allocator
            if ((GPUMemory.total_mem - GPUMemory.total_mem * GPUMemory.get_mem_util()) / 1024.) < (
                    (mem_scale_factor / float(num_GPUs) * model.module.gpu_usage) + 1.):

                # code for variable batch size implementation as per gpu constraint; remove for old code
                approx_small_batch_size = (((GPUMemory.total_mem - GPUMemory.total_mem * GPUMemory.get_mem_util()) / 1024.
                                            - 1.) * float(num_GPUs) / mem_scale_factor) //\
                                          (model.module.gpu_usage / float(args.batch_size))

                diff = float('inf')
                temp_small_batch_size = approx_small_batch_size
                for j in range(1, (args.batch_size // 2) + 1):
                    if args.batch_size % j == 0 and abs(j - approx_small_batch_size) < diff:
                        diff = abs(j - approx_small_batch_size)
                        temp_small_batch_size = j
                batch_seq_split_size = temp_small_batch_size
            else:
                batch_seq_split_size = args.batch_size
        else:
            batch_seq_split_size = args.batch_size

        # Get training and validation dataset loaders
        dataset.train_loader, dataset.val_loader = dataset.get_dataset_loader(batch_seq_split_size,
                                                                              args.workers, device)

        print('sequential batch size split size:{}'.format(batch_seq_split_size))

        epoch = 0
        best_epoch = 0
        best_prec = 0
        best_val_train_prec = 0
        all_train = []
        all_val = []

        while epoch < args.epochs:
            # train for one epoch
            train_prec = train(dataset.train_loader, model, criterion, epoch,
                               optimizer, lr_scheduler, device, batch_seq_split_size, args)
            # evaluate on validation set
            prec = validate(dataset.val_loader, model, criterion, epoch, device, args)

            all_train.append(train_prec)
            all_val.append(prec)

            # remember best prec@1 and save checkpoint
            is_best = prec > best_prec
            if is_best:
                best_epoch = epoch
                best_val_train_prec = train_prec
                best_prec = prec

            # if architecture doesn't train at all skip it
            if epoch == args.lr_wr_epochs - 1 and train_prec < (2 * 100.0 / dataset.num_classes):
                break

            # increment epoch counters
            epoch += 1
            lr_scheduler.scheduler_epoch += 1

        # append architecture results to csv
        df = df.append(pd.DataFrame([[skew_model_params["filters"][i + current_id],
                                      skew_model_params["total_params"][i + current_id],
                                      skew_model_params["means"][i + current_id],
                                      skew_model_params["vars"][i + current_id],
                                      skew_model_params["skews"][i + current_id], best_prec,
                                      best_val_train_prec, best_epoch, prec, train_prec, all_train, all_val]],
                                    columns=columns_list), ignore_index=True)
        df.to_csv(save_path + '/model_%03d' % (i+1+current_id) + '.csv')

        del model
        del optimizer


if __name__ == '__main__':
    main()

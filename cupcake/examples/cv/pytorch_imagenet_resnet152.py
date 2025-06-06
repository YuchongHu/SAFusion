import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
# import horovod.torch as hvd
import os
import math
from tqdm import tqdm

import lib.hv_distributed_optimizer as hvd
from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer

from utils_model import get_network
import numpy as np
import matplotlib.pyplot as plt
import time
import os



os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'

from compression import compressors
from utils_model import get_network


import timeit
import numpy as np
from profiling import benchmark




# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar100 + ResNet-152  Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Horovod
# parser.add_argument('--net', default='resnet50',type=str, required=True, help='net type')
parser.add_argument('--model-net', default='resnet152',type=str, help='net type')
parser.add_argument('--train-dir', default=os.path.expanduser('/data/dataset/cv/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/data/dataset/cv/imagenet/val'),
                    help='path to validation data')
# parser.add_argument('--train-dir', default=os.path.expanduser('/data/dataset/cv/mini-imagenet/images/train'),
#                     help='path to training data')
# parser.add_argument('--val-dir', default=os.path.expanduser('/data/dataset/cv/mini-imagenet/images/val'),
#                     help='path to validation data')
                    
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./pytorch_checkpoints/cifar100_resnet50/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')


# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')

parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')


# MG_WFBP
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
# parser.add_argument('--batch-size', type=int, default=32,
#                     help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=20,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')

# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')

# parser.add_argument('--use-adasum', action='store_true', default=False,
#                     help='use adasum algorithm to do reduction')

parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')


# parser.add_argument('--threshold', type=int, default=2370520, help='Set threshold if mgwfbp is False')

parser.add_argument('--threshold', type=int, default=2370520, help='Set threshold if mgwfbp is False')





parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')


parser.add_argument('--compressor', type=str, default='dgc', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')





y_loss = {}  # loss history
y_loss['train'] = []
y_loss['test'] = []
y_acc = {}
y_acc['train'] = []
y_acc['test'] = []
x_test_epoch_time = []
x_train_epoch_time = []
x_epoch = []

def draw_curve(epoch):
    fig = plt.figure(figsize=(18, 14))

    ax0 = fig.add_subplot(221, title="train loss", xlabel = 'epoch', ylabel = 'loss')
    ax1 = fig.add_subplot(222, title="test loss")

    ax2 = fig.add_subplot(223, title="train top1_accuracy", xlabel = 'epoch', ylabel = 'accuracy')
    ax3 = fig.add_subplot(224, title="test top1_accuracy")

    x_epoch = range(1, epoch + 1)
    ax0.plot(x_epoch, y_loss['train'], 'ro-', label='train')
    ax1.plot(x_epoch, y_loss['test'], 'ro-', label='test')

    ax2.plot(x_epoch, y_acc['train'], 'bo-', label='train')
    ax3.plot(x_epoch, y_acc['test'],  'bo-', label='test')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    dataset_model = '/imagenet_resnet152'
    date = '1120'    
    comp_type = '/merge_layer_wise'
    comp = '/merge_layer_wise'
    comp_type = '/merge_number_noef'
    comp = '/merge_number_noef'
      
    figpath = './mgwfbp/result' + dataset_model
    datapath = './mgwfbp/result' + dataset_model + comp_type
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    fig.savefig(figpath + comp + '_epoch' + str(epoch) + '_' + date + '.jpg')

    np.savetxt(datapath + comp + "_e" + str(epoch) + "_ytest_acc_" + date + ".txt", y_acc['test'])

    np.savetxt(datapath + comp + "_e" + str(epoch) + "_xtrain_time_" + date + ".txt", x_train_epoch_time)
    np.savetxt(datapath + comp + "_e" + str(epoch) + "_ytrain_acc_" + date + ".txt", y_acc['train'])




def train(epoch):
    bias_gaussiank_array=[]
    bias_dgc_array=[]
    bias_redsync_array=[]
    
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    optimizer._compression.compress_time=[]
    optimizer._compression.threshold_time=[]
    
    optimizer.synchronize_time= []
    optimizer.para_update_time= []
    optimizer.hook_time= []    

    io_time_array= []
    forward_backforward_time_array= []
    forward_time_array= []
    backward_time_array= []
    step_time_array= []
    update_time_array= []
    
    optimizer.handle_synchronize_time= []
    optimizer_synchronize_time_array= []
    
    optimizer._compression.epoch=epoch
    
    # optimizer._communicator.compressor.epoch=epoch

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            
            # optimizer._communicator.compressor.iteration=batch_idx
            s_time=time.time()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            io_time_array.append(time.time()-s_time)
            
            e_time=time.time()

            optimizer.zero_grad()
            
            
            
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            
            forward_time_array.append(time.time()-e_time)
            
            b_time=time.time()
            loss.backward()
            backward_time_array.append(time.time()-b_time)
            
            
            
            s_time=time.time()             
            # Gradient is applied across all ranks
            optimizer.step()
            
            
            step_time_array.append(time.time()-s_time)
            
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            
            u_time=time.time() 
            t.update(1)
            update_time_array.append(time.time()-u_time)
            
            optimizer_synchronize_time_array.append(optimizer.handle_synchronize_time)
            optimizer.handle_synchronize_time= []
            



    y_loss['train'].append(train_loss.avg.item())
    y_acc['train'].append(train_accuracy.avg.item())
    end_time_epoch = time.time()
    x_train_epoch_time.append(end_time_epoch - modified_time)    
    io_time=sum(io_time_array)
    # forward_backforward_time=sum(forward_backforward_time_array)
    forward_time=sum(forward_time_array)
    backward_time=sum(backward_time_array)
    step_time=sum(step_time_array)
    update_time=sum(update_time_array)
    
    compress_time_array =optimizer._compression.compress_time
    threshold_time_array =optimizer._compression.threshold_time
    compress_time=sum(compress_time_array)
    threshold_time=sum(threshold_time_array)    
    
    synchronize_time=sum(optimizer.synchronize_time)
    para_update_time=sum(optimizer.para_update_time)
    hook_time=sum(optimizer.hook_time)
    
    if hvd.rank() == 0:

        print('compress_time = ', compress_time)
        print('threshold_time = ', threshold_time)

        print('io_time = ', io_time)        
        print('forward_time = ', forward_time)
        print('backward_time = ', backward_time-compress_time)        
        print('step_time = ', step_time)
        # print('update_time = ', update_time)
        print('communication_time = ', synchronize_time)
        print('para_update_time = ', para_update_time)
        print('hook_time = ', hook_time)
        # print('buffer_time = ', buffer_time) 

        print('optimizer_synchronize_time_array= ', optimizer_synchronize_time_array)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    val_start_time = time.time()

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    

    y_loss['test'].append(val_loss.avg.item())
    y_acc['test'].append(val_accuracy.avg.item())
    end_time_epoch = time.time()
    val_time = end_time_epoch - val_start_time
    global modified_time
    modified_time += val_time
    x_test_epoch_time.append(end_time_epoch - modified_time)

    


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    

    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.set_default_dtype(torch.float16)
    os.environ['HOROVOD_NUM_NCCL_STREAMS'] = str(args.nstreams)

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    # for try_epoch in range(args.epochs, 0, -1):
    #     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
    #         resume_from_epoch = try_epoch
    #         break

    # # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # # checkpoints) to other ranks.
    # resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
    #                                   name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    # log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    
    CIFAR100_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    CIFAR100_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up standard ResNet-50 model.
    model = models.resnet152(num_classes=1000)
    # model=get_network(args)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)
    
     # Set up fixed fake data
    image_size = 224
    if args.model == 'inception_v3':
        image_size = 227
    data = torch.randn(args.batch_size, 3, image_size, image_size)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    if args.mgwfbp:
        seq_layernames, layerwise_times, _ = benchmark(model, (data, target), F.cross_entropy, task='imagenet')
        layerwise_times =hvd.broadcast(layerwise_times,root_rank=0)
        # layerwise_times = comm.bcast(layerwise_times, root=0)
    else:
        seq_layernames, layerwise_times = None, None
        
    
    
    
    

    
    # Horovod    
    optimizer = hvd.DistributedOptimizer(args.model_net, 
                                         optimizer, 
                                         named_parameters=model.named_parameters(), 
                                         compression=compressors[args.compressor](), 
                                         is_sparse=args.density<1, 
                                         density=args.density, 
                                         seq_layernames=seq_layernames, 
                                         layerwise_times=layerwise_times, norm_clip=None,
                                         threshold=args.threshold, 
                                         writer=None, 
                                         gradient_path='./', 
                                         momentum_correction=False, 
                                         fp16=args.fp16, 
                                         mgwfbp=args.mgwfbp, 
                                         rdma=args.rdma, 
                                         asc=args.asc)


    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    

    

    start_time = time.time()
    modified_time = start_time
    if hvd.rank() == 0:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('start_time_str = ', start_time_str)

    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)
        
        

    if hvd.rank() == 0:
        # torch.cuda.synchronize()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('end_time_str = ', end_time_str)
        print('end_time - start_time = ', end_time - start_time)
        training_time = end_time - modified_time
        print('end_time - modified_time = ', training_time)
        print(f"Samples per second: GPU * epochs * iter * batch-size / time = {8 * args.epochs * 5005 * 32 / training_time}")
        draw_curve(args.epochs)  
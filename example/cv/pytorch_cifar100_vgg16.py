import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
# import horovod.torch as hvd
import os
import math
from tqdm import tqdm
# from adtopk_lib.helper import get_communicator

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 环境变量HOROVOD_FUSION_THRESHOLD实际上以字节为单位.
# 然而, 当使用horovodrun时, 有一个--fusion-threshold-mb以MB为单位的参数.
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'


import sys
sys.path.append("../..") 
import hv_distributed_optimizer as hvd
from compression import compressors
from utils_model import get_network

import timeit
import numpy as np
from profiling import benchmark



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar100 + VGG-16 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Horovod
# parser.add_argument('--net', default='resnet50',type=str, required=True, help='net type')
# parser.add_argument('--model-net', default='resnet50',type=str, help='net type')
parser.add_argument('--model-net', default='vgg16',type=str, help='net type')

parser.add_argument('--train-dir', default=os.path.expanduser('~/cifar100/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/cifar100/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./pytorch_checkpoints/cifar100_vgg16/checkpoint-{epoch}.pth.tar',
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


# Gradient Merging
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='vgg16',
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

# 设置合并的阈值大小,default=23705252为ResNet50所有层梯度元素数量的总和
# parser.add_argument('--threshold', type=int, default=536870912, help='Set threshold if mgwfbp is False')
# parser.add_argument('--threshold', type=int, default=671080, help='Set threshold if mgwfbp is False')
parser.add_argument('--threshold', type=int, default=2370520, help='Set threshold if mgwfbp is False')

# parser.add_argument('--threshold', type=int, default=1000520, help='Set threshold if mgwfbp is False')


# parser.add_argument('--threshold', type=int, default=23705252, help='ResNet-50 Set threshold if mgwfbp is False')

parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')

# Baseline
# parser.add_argument('--compressor', type=str, default='none', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
# parser.add_argument('--density', type=float, default=1, help='Density for sparsification')

# Top-k + EF
parser.add_argument('--compressor', type=str, default='eftopk', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=0.1, help='Density for sparsification')
# parser.add_argument('--density', type=float, default=0.0101, help='Density for sparsification')
# parser.add_argument('--density', type=float, default=0.0099, help='Density for sparsification')


# Gaussiank + EF
# parser.add_argument('--compressor', type=str, default='gaussian', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
# parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')



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

    dataset_model = '/cifar100_vgg16'
    date = '0106'

    # comp_type = '/merge_threshold_ef'
    # comp = '/merge_threshold_ef'

    # comp_type = '/merge_number_noef_wait_001_4'
    # comp = '/merge_number_noef_wait_001_4'

    # comp_type = '/merge_number_noef_wait_005_9_1215_3'
    # comp = '/merge_number_noef_wait_005_9_1215_3'

    # comp_type = '/merge_threshold_noef_wait_005'
    # comp = '/merge_threshold_noef_wait_005'
    
    # comp_type = '/fc_conv5_split_001_ef'
    # comp = '/fc_conv5_split_001_ef'
    
    # comp_type = '/fc_conv5_split_001_fc_nocompression_ef'
    # comp = '/fc_conv5_split_001_fc_nocompression_ef'

    comp_type = '/cifar100_vgg16_'+str(args.compressor)+'_ef_epoch_'+str(args.epochs)+'_'+str(args.density).replace('.','')
    comp = '/cifar100_vgg16_'+str(args.compressor)+'_ef_epoch_'+str(args.epochs)+'_'+str(args.density).replace('.','')
    
    # resnet50 resnet101 Vgg16,  Vgg19
    # 0.1 0.01 0.05
    # topk, gaussiank, readsync dgc randomk , sidoc

    
    save_dir = '/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result_train_ours'
    figpath = save_dir + dataset_model
    datapath = save_dir + dataset_model + comp_type
    
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    # fig.savefig(figpath + comp + '_epoch' + str(epoch) + '_' + date + '.jpg')

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
    
    optimizer._compression.topk_time=[]
    optimizer._compression.threshold_time=[]
    
    optimizer.synchronize_time= []
    optimizer.para_update_time= []
    optimizer.hook_time= []
    
    # optimizer._communicator.compressor.bias_gaussiank=[]
    # optimizer._communicator.compressor.bias_dgc=[]
    # optimizer._communicator.compressor.bias_redsync=[]
    
    # optimizer._communicator.compression_time_array=[]
    # optimizer._communicator.decompression_time_array=[]
    # optimizer._communicator.send_time_array=[]
    # optimizer._communicator.receive_time_array=[]
    # optimizer._communicator.synchronize_time_array=[]

    io_time_array= []
    forward_backforward_time_array= []
    forward_time_array= []
    backward_time_array= []
    step_time_array= []
    update_time_array= []
    
    optimizer.handle_synchronize_time= []
    optimizer_synchronize_time_array= []
    
    optimizer._compression.epoch=epoch
    
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
            
            # Split data into sub-batches of size batch_size
            # for i in range(0, len(data), args.batch_size):
            #     data_batch = data[i:i + args.batch_size]
            #     target_batch = target[i:i + args.batch_size]
            #     output = model(data_batch)
            #     train_accuracy.update(accuracy(output, target_batch))
            #     loss = F.cross_entropy(output, target_batch)
            #     train_loss.update(loss)
            #     # Average gradients among sub-batches
            #     loss.div_(math.ceil(float(len(data)) / args.batch_size))
            #     loss.backward()
            
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            
            forward_time_array.append(time.time()-e_time)
            
            b_time=time.time()
            loss.backward()
            backward_time_array.append(time.time()-b_time)
            
            # if hvd.rank() == 0 and batch_idx == 5 and epoch < 5:
            #     # tensor_np_array=[]
            #     index=1
            #     for name_1, param_1 in model.named_parameters():
            #         tensor = param_1.grad
            #         tensor = tensor.flatten()
            #         tensor_np = tensor.cpu().numpy()
            #         # tensor_np_array.append(tensor_np)
            #         np.savetxt("./examples/torch/resnet_layer_gradient/epoch" + str(epoch+1) + "/resnet18_grad_tensor_5_"+str(index)+".txt", tensor_np)
            #         index = index + 1
            
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
            
            # bias_gaussiank=optimizer._communicator.compressor.bias_gaussiank  
            # bias_dgc=optimizer._communicator.compressor.bias_dgc
            # bias_redsync=optimizer._communicator.compressor.bias_redsync

            # gaussiank_sum= sum(bias_gaussiank)/len(bias_gaussiank)
            # dgc_sum= sum(bias_dgc)/len(bias_dgc)
            # redsync_sum= sum(bias_redsync)/len(bias_dgc)

            # bias_gaussiank_array.append(gaussiank_sum)
            # bias_dgc_array.append(dgc_sum)
            # bias_redsync_array.append(redsync_sum)

            # optimizer._communicator.compressor.bias_gaussiank=[]
            # optimizer._communicator.compressor.bias_dgc=[]
            # optimizer._communicator.compressor.bias_redsync=[]
            
            # topk_time_array =optimizer._communicator.compressor.topk_time
            # threshold_time_array =optimizer._communicator.compressor.threshold_time
            # if hvd.rank() == 0:
            #     datapath='/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/compression_time/'
            #     np.savetxt(datapath + "topk_time/topk_time_"+str(epoch)+"_rank_"+str(hvd.rank())+"_iteration_"+str(batch_idx)+".txt", topk_time_array)
            #     np.savetxt(datapath + "threshold_time/threshold_time_"+str(epoch)+"_rank_"+str(hvd.rank())+"_iteration_"+str(batch_idx)+".txt", threshold_time_array)
                
            # optimizer._communicator.compressor.topk_time=[]
            # optimizer._communicator.compressor.threshold_time=[]
    
    
    # datapath='/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/bias_threshold/'
    # np.savetxt(datapath + "average_inter_worker_bias_gaussiank_array_epoch_"+str(epoch)+"_rank_"+str(hvd.rank())+".txt", bias_gaussiank_array)
    # np.savetxt(datapath + "average_inter_worker_bias_dgc_array_epoch_"+str(epoch)+"_rank_"+str(hvd.rank())+".txt", bias_dgc_array)
    # np.savetxt(datapath + "average_inter_worker_bias_redsync_array_epoch_"+str(epoch)+"_rank_"+str(hvd.rank())+".txt", bias_redsync_array)


    # if hvd.rank() == 0:
    #     print('bias_gaussiank_array = ', bias_gaussiank_array)
    #     print('bias_dgc_array = ', bias_dgc_array)
    #     print('bias_redsync_array = ', bias_redsync_array)
    
    # bias_gaussiank_array=[]
    # bias_dgc_array=[]
    # bias_redsync_array=[]

    # if log_writer:
    #     log_writer.add_scalar('train/loss', train_loss.avg, epoch)
    #     log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

    y_loss['train'].append(train_loss.avg.item())
    y_acc['train'].append(train_accuracy.avg.item())
    end_time_epoch = time.time()
    x_train_epoch_time.append(end_time_epoch - modified_time)
    
    
    # compression_time=sum(optimizer._communicator.compression_time_array)
    # decompression_time=sum(optimizer._communicator.decompression_time_array)
    # send_time=sum(optimizer._communicator.send_time_array)
    # receive_time=sum(optimizer._communicator.receive_time_array)
    # synchronize_time=sum(optimizer._communicator.synchronize_time_array)
    
    io_time=sum(io_time_array)
    # forward_backforward_time=sum(forward_backforward_time_array)
    forward_time=sum(forward_time_array)
    backward_time=sum(backward_time_array)
    step_time=sum(step_time_array)
    update_time=sum(update_time_array)
    
    topk_time_array =optimizer._compression.topk_time
    threshold_time_array =optimizer._compression.threshold_time
    topk_time=sum(topk_time_array)
    threshold_time=sum(threshold_time_array)
    
    synchronize_time=sum(optimizer.synchronize_time)
    para_update_time=sum(optimizer.para_update_time)
    hook_time=sum(optimizer.hook_time)
    
    if hvd.rank() == 0:
        # datapath='/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/compression_time/'
        # np.savetxt(datapath + "topk_time/topk_time_"+str(epoch)+"_rank_"+str(hvd.rank())+".txt", topk_time_array)
        # np.savetxt(datapath + "threshold_time/threshold_time_"+str(epoch)+"_rank_"+str(hvd.rank())+".txt", topk_time_array)
        
        # print('compression_time = ', compression_time)
               
        print('topk_time = ', topk_time)
        print('threshold_time = ', threshold_time)
                     
        # print('send_time = ', send_time)        
        # print('decompression_time = ', decompression_time)
        # print('receive_time = ', receive_time)
        # print('synchronize_time = ', synchronize_time)
        
        print('io_time = ', io_time)
        print('forward_time = ', forward_time)
        print('backward_time = ', backward_time-topk_time)
        print('step_time = ', step_time)
        # print('update_time = ', update_time)
        print('communication_time = ', synchronize_time)
        print('para_update_time = ', para_update_time)
        print('hook_time = ', hook_time)
        # print('buffer_time = ', buffer_time)
        
        
        # print('backforward_time = ', forward_backforward_time-(send_time+receive_time+decompression_time+compression_time))
        print('---------------------------------')
        # print('optimizer_synchronize_time_array= ', optimizer_synchronize_time_array[:20])



    # topk_time=sum(optimizer._communicator.compressor.topk_time)
    # threshold_time=sum(optimizer._communicator.compressor.threshold_time)
    # if hvd.rank() == 0:
    #     print('topk_time = ', topk_time)
    #     print('threshold_time = ', threshold_time)

    # if hvd.rank() == 0:
    #     print('\nTrain set: Average loss: {:.4f}, Train Accuracy: {:.2f}%\n'.format(
    #             train_loss.avg.item(), 100. * train_accuracy.avg.item()))


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

    # if log_writer:
    #     log_writer.add_scalar('val/loss', val_loss.avg, epoch)
    #     log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    y_loss['test'].append(val_loss.avg.item())
    y_acc['test'].append(val_accuracy.avg.item())
    end_time_epoch = time.time()
    val_time = end_time_epoch - val_start_time
    global modified_time
    modified_time += val_time
    x_test_epoch_time.append(end_time_epoch - modified_time)

    # if hvd.rank() == 0:
    #     print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}%\n'.format(
    #             val_loss.avg.item(), 100. * val_accuracy.avg.item()))


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
    
    
    # if epoch < 40:
    #     lr_adj = 1e-1
    # elif epoch < 60:
    #     lr_adj = 1e-2
    # elif epoch < 80:
    #     lr_adj = 1e-3
    # else:
    #     lr_adj = 1e-4

    # if epoch < args.warmup_epochs:
    #     epoch += float(batch_idx + 1) / len(train_loader)
    #     lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    # elif epoch < 30:
    #     lr_adj = 1.
    # elif epoch < 60:
    #     lr_adj = 1e-1
    # elif epoch < 80:
    #     lr_adj = 1e-2
    # else:
    #     lr_adj = 1e-3
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
    # imagenet
    # train_dataset = \
    #     datasets.ImageFolder(args.train_dir,
    #                          transform=transforms.Compose([
    #                              transforms.RandomResizedCrop(224),
    #                              transforms.RandomHorizontalFlip(),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                   std=[0.229, 0.224, 0.225])
    #                          ]))
    # CIFAR100
    train_dataset = \
        datasets.CIFAR100(args.train_dir,
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                                      std=CIFAR100_TRAIN_STD)
                             ]))
    
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    
    # Imagenet
    # val_dataset = \
    #     datasets.ImageFolder(args.val_dir,
    #                          transform=transforms.Compose([
    #                              transforms.Resize(256),
    #                              transforms.CenterCrop(224),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                   std=[0.229, 0.224, 0.225])
    #                          ]))    
    # CIFAR100
    val_dataset = \
        datasets.CIFAR100(args.val_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                 # transforms.Resize(256),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                                      std=CIFAR100_TRAIN_STD)
                             ]))    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up standard ResNet-50 model.
    # model = models.resnet50()
    model=get_network(args)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1
    # By default, Adasum doesn't need scaling up learning rate.
    # lr_scaler = hvd.size() if not args.use_adasum else 1

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
    
    
    # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather'}
    # Horovod: wrap optimizer with DistributedOptimizer.
    # 得到一个分布式的SGD优化器
    # optimizer = hvd.DistributedOptimizer(
    #     optimizer, grc, named_parameters=model.named_parameters())


    # Horovod
    # # Allgather
    # # params = {'compressor': 'imbalancetopktime', 'memory': 'residual', 'communicator': 'allgather','model_named_parameters':model.named_parameters()}
    # params = {'compressor': 'imbalancetopktime', 'memory': 'none', 'communicator': 'allgather','model_named_parameters':model.named_parameters()}
    # # params = {'compressor': 'none', 'memory': 'none', 'communicator': 'allgather','model_named_parameters':model.named_parameters()}
    # # params = {'compressor': 'none', 'memory': 'none', 'communicator': 'allreduce','model_named_parameters':model.named_parameters()}
    
    # # params = {'compressor': 'none', 'memory': 'none', 'communicator': 'allgather','model_named_parameters':model.named_parameters()}
    # # params = {'compressor': 'none', 'memory': 'none', 'communicator': 'allreduce','model_named_parameters':model.named_parameters()}
    
    # communicator = get_communicator(params)
    # optimizer = hvd.DistributedOptimizer(
    #     optimizer, communicator, named_parameters=model.named_parameters())
    
    # MG-WFBP
    optimizer = hvd.DistributedOptimizer(args.model_net, optimizer, 
                                         named_parameters=model.named_parameters(), compression=compressors[args.compressor](), is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=args.threshold, writer=None, gradient_path='./', momentum_correction=False, fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, asc=args.asc)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    

    # Horovod: wrap optimizer with DistributedOptimizer.
    # All-reduce
    # optimizer = hvd.DistributedOptimizer(
    #     optimizer, named_parameters=model.named_parameters(),
    #     compression=compression,
    #     backward_passes_per_step=args.batches_per_allreduce,
    #     op=hvd.Adasum if args.use_adasum else hvd.Average,
    #     gradient_predivide_factor=args.gradient_predivide_factor)
    
    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    # if resume_from_epoch > 0 and hvd.rank() == 0:
    #     filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    #     checkpoint = torch.load(filepath)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    start_time = time.time()
    modified_time = start_time
    if hvd.rank() == 0:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('start_time_str = ', start_time_str)

    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)

        # 保存最后一个训练模型
        # if epoch==args.epochs-1:
        #     save_checkpoint(epoch)

    if hvd.rank() == 0:
        # torch.cuda.synchronize()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('end_time_str = ', end_time_str)
        print('end_time - start_time = ', end_time - start_time)
        training_time = end_time - modified_time
        print('end_time - modified_time = ', training_time)
        print(f"Samples per second: GPU * epochs * iter * batch-size / time = {8 * args.epochs * 196 * 32 / training_time}")
        
        
        draw_curve(args.epochs)


# SAFusion: Efficient Tensor Fusion with Sparsification Ahead for High-Performance Distributed DNN Training

__SAFusion__ is a new efficient tensor fusion mechanism for high-performance distributed DNN training. __SAFusion__ first proposes a sparsification-ahead tensor fusion scheme, which performs sparsification on each of the gradient tensors before merging them during tensor fusion, instead of sparsification-behind tensor fusion, so as to avoid gradient tensor missing and thus improve the convergence performance. Then, __SAFusion__ designs an inter-worker gradient alignment fusion scheme that merges the same amount of sparsified gradients across workers to avoid long gradient synchronization waiting, and an intra-worker adaptive buffer sizing scheme that maximizes the overlap of backpropagation and communication time to reduce multiple waiting periods.
This repository contains __SAFusion__'s source code, as well as a set of benchmarking scripts for some popular open-source distributed DNN training systems with state-of-the-art tensor fusion schemes. 

# Introduction
This code repository covers:
### __SAFusion Framework__
- SAF(Naive): Sparsification-ahead tensor fusion
- SAF-Inter: Aligned inter-worker gradient tensor fusion
- SAF-(Inter+Intra): Adaptive intra-worker buffer sizing

### __State-of-the-art tensor fusion schemes__

- [WFBP](https://github.com/horovod/horovod)
- [OkTopk](https://dl.acm.org/doi/pdf/10.1145/3126908.3126912)
- [OMGS](https://github.com/HKBU-HPML/OMGS-SGD)
- [Cupcake](https://github.com/zhuangwang93/Cupcake)

### __State-of-the-art sparsification algorithms__

- [DGC](https://arxiv.org/pdf/1712.01887.pdf)
- [Gaussiank](https://arxiv.org/pdf/1911.08772.pdf)
- [Redsync](https://www.sciencedirect.com/science/article/pii/S0743731518308657)
- [SIDCo](https://proceedings.mlsys.org/paper_files/paper/2021/file/fea47a8aa372e42f3c84327aec9506cf-Paper.pdf)

# Implementation



## **__SAFusion__** System Architecture
We use the [PyTorch](https://github.com/pytorch/pytorch) framework and implemented the prototype system of __SAFusion__ based on the [Horovod](https://github.com/horovod/horovod) framework using NCCL as the communication library. The overview of our system is as follows: 
<!-- ![Overview](Overview.png) -->
<center class ='img'>
<img src="Overview_0208.png" width="600px" />
</center>

In our system of SAFusion, each worker contains a __Generator__ module for generating an efficient sparsification-ahead fusion buffer, a __Controller__ module for controlling a series of operations such as sparsified gradient pushing, pulling, and communication in the fusion buffer, and a __Sparsification Compression__ module for performing layer-wise gradient sparsification during the backward propagation.

## **__SAFusion__** Generator Workflow
The workflow of the __SAFusion__ __Generator__ module：
<center class ='img'>
<img src="Generator_0208.png" width="600px" />
</center>

# Installation


## **Prerequisites**
- CUDA-12.0
- Python >= 3.9
- [NCCL-2.8.3](https://github.com/NVIDIA/nccl)
- [PyTorch-1.3.+](https://github.com/pytorch/pytorch)
- [OpenMPI-4.0.+](https://www-lb.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.28.1+](https://github.com/horovod/horovod)
- [numpy](https://github.com/numpy/numpy)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [openmpi](https://www.open-mpi.org/software/ompi/)

## **Get the code**
```
git clone https://github.com/HPDC25-SAFusion/SAFusion.git
cd SAFusion
pip install -r requirements.txt
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.28.0
```

if pip installation fails, please try to upgrade pip via `pip install --upgrade pip`. If [Horovod](https://github.com/horovod/horovod) installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html).

## **Quick start**
The primary benchmark is provided in `example`. 
For example, we can use the following command to run the benchmark on 8 GPUs, with compression algorithm as dgc, communication primitive as allgather, memory as residual.
 
**To run BERT-large training job:**
```
cd safusion/example/nlp/bert/scripts
bash run_squad_bert.sh
```

**To run GPT2-large training job:**
```
cd safusion/example/nlp/gpt
bash run_clm_no_trainer_hvd_103.sh
```

**To run ResNet-152 training job:**
```
cd safusion/example/cv
bash run_imagenet_resnet152.sh
```

**To run ViT-large training job:**
```
cd safusion/example/cv/vit
bash run_imagenet_no_trainer.sh
```



## **Papers**

SAFusion: Efficient Tensor Fusion with Sparsification Ahead for High-Performance Distributed DNN Training

## **Referred Datasets**

- Wikitex-2/103: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
- CIFAR-100: [https://www.cs.utoronto.ca/~kriz/cifar.html](https://www.cs.utoronto.ca/~kriz/cifar.html)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)

## **License**

See [LICENSE](https://github.com/ATC24-SAFusion/SAFusion/blob/main/LICENSE.txt).

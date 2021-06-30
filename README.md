# A template for fast distributed training using PyTorch-Lightning and Horovod

This template was created for the [Fujitsu Multi-Node GPU
Challenge](https://marketing.global.fujitsu.com/multinode-gpu-challenge_reg.html).

This template provides installation instructions and a short well-commented
template to get started with fast multi-node training on multiple GPUs using
PyTorch and Horovod.

## Benchmarks

This template achieves near linear scaling. 

The following benchmarks were run on two nodes with infiniband and GeForce RTX
2080 Ti GPUs. We report the time to run a single epoch using (details about the
command-line can be found below):

``` sh
horovodrun -np ${num_processes} -H node1:4,node2:4 python template.py --batch_size 2 --max_epochs 3 --profile simple
```
 
| Num processes | Time (seconds) |
|:--------------|:---------------|
| 1             | 45.48          |
| 2             | 23.9           |
| 4             | 12.1           |
| 8             | 6.16           |


## Installation

To run multi-node training, run these installation instructions on all nodes
that you want to use.

### Pip
``` sh
pip3 install torch==1.8.1 torchvision pytorch-lightning==1.3.2 horovod==0.22.0 
```

This will automatically install pytorch for CUDA-toolkit version 10.2. Please
use [the PyTorch installation
instructions](https://pytorch.org/get-started/locally/) if you want to use
CUDA-toolkit version 11.

### Anaconda

Because installing Horovod using conda is not [particularly
easy](https://horovod.readthedocs.io/en/stable/conda.html), I recommend creating
a new environment using the anaconda package manager, and then installing the
packages using pip anyway:

``` sh
conda create -n multi-node python=3.9 
conda activate multi-node
pip3 install torch==1.8.1 torchvision pytorch-lightning==1.3.2 horovod==0.22.0 
```

## Running on a single node

You can run the training code on a single node as follows:
``` sh
horovodrun -np 4 python template.py --batch_size 2 --max_epochs 1
```

The `-np` argument specifies that 4 training processes are started, each using 1
GPU for a total of 4 GPUs. The combined batch size equals `4 x 2 = 8`, i.e.,
each GPU has a local batch size of 2. This script stops after 1 epoch, i.e.,
after going through all training data once. If you remove `--max_epochs`, the
code will continue to run until the validation error starts increasing.

### Additional options

**Profiling** If you want to know how long each step of the script took, run 

``` sh
horovodrun -np 4 python template.py --batch_size 2 --max_epochs 1 --profiler simple
```

This shows a report summarizing how long each step of the training procedure
took.

 
## Running on multiple nodes

### 1. Copy the code to all nodes
If you are on linux, you may use the following snippet to do this

``` sh
for node in node1 node2 ; do rsync -av fujitsu_challenge ${node}: ; done
```

### 2. Log in to a node

Log in to one of the nodes. You might want to use `-o ForwardAgent=true` to make
sure that you can login to other nodes from `node1`:
``` sh
ssh -o ForwardAgent=true node1
```
From node1, make sure you can login to other nodes:
``` sh
ssh node2
```

### 3. Run training code 

The following code runs the training process on 2 nodes using 4 GPUs each:
``` sh
horovodrun -np 8 -H node2:4,node1:4 python template.py --batch_size 2 --max_epochs 1
```

The `-np` argument specifies that a total of 8 processes have to be started. The
`-H` argument lets Horovod know where these processes can be started. This is
explained in the manual (`horovodrun --help`):
``` 
  -H HOSTS, --hosts HOSTS
            List of host names and the number of available slots for running processes on each, of the form:            
            <hostname>:<slots> (e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1, 4 on host2, and  
            1 on host3). If not specified, defaults to using localhost:<np>                                             
```

Now, the batch size equals `16 = 8 x 2`. The `--max_epochs 1` flags causes the
training to stop after 1 epoch. If this parameter is not passed, training can be
stopped using [early
stopping](https://pytorch-lightning.readthedocs.io/en/1.3.2/common/early_stopping.html).

## Common pitfalls

### SSH, SIGHUP and Tmux

If you log into a node using SSH and start a long running job, it will be
terminated as soon as the SSH connection fails. To prevent this, use a terminal
multiplexer like Tmux. The [Tao of
Tmux](https://leanpub.com/the-tao-of-tmux/read) is a great introduction.

### Different file system layouts

The horovodrun program expects that the file system layout is exactly the same
on all nodes. If there is a difference, you might get an error like

``` sh
[3]<stderr>:python: can't open file '/home/projects/fujitsu_challenge/template.py': [Errno 2] No such file or directory
```

In addition, it makes sense to use a shared networked file system to store your
datasets so they can be accessed uniformly from all nodes.

### Installing random versions of packages

The PyTorch and PyTorch-Lightning packages change rapidly. Therefore, always
make sure to pin the versions you install. Like we did in the installation steps
above. You will probably be fine if you use the latest PyTorch version (1.9 at
time of writing). If you want to use the latest version of PyTorch-Lightning,
make sure to check the
[changelog](https://pytorch-lightning.readthedocs.io/en/latest/generated/CHANGELOG.html)
for any breaking changes.


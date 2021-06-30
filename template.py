#!/usr/bin/env python3

from argparse import ArgumentParser
from tqdm import tqdm

# PyTorch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# PyTorch-Lightning and Horovod:
import pytorch_lightning as pl
import horovod.torch as hvd

# Image input/output:
import torchvision

###############################################################################
#                           Define a neural network                           #
###############################################################################

# Define your neural network here, or import a known architecture.


class DnCNN(nn.Module):
    def __init__(self, num_channels, num_layers=20):
        # Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a gaussian
        # denoiser: residual learning of deep CNN for image denoising. IEEE Transactions
        # on Image Processing, 26(7), 3142–3155.
        # http://dx.doi.org/10.1109/tip.2017.2662206

        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


###############################################################################
#                          Load or create dataset(s)                          #
###############################################################################
def create_dataset(n, num_images):
    """Create a dataset.

    The dataset contains `num_images` n x n single-channel images. The
    images are sampled from a Gaussian noise distribution.
    """
    # just generate some random noise
    images = torch.randn(num_images + 1, 1, n, n)

    # x and y contain subsequent random images.
    x = images[:-1]
    y = images[1:]
    dataset = TensorDataset(x, y)
    return dataset


###############################################################################
#                     Define your PyTorch-Lightning Module                    #
###############################################################################

# The Lightning module defines
# - what your network is,
# - how to run it, and
# - how it is trained.

# You can follow along with the code using this tutorial from PyTorch-Lightning:
#
# - https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html


class Denoiser(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # You can define your neural network here, or load a known network
        # architecture as we do in this case.
        # https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html#move-your-computational-code
        self.net = DnCNN(1, num_layers=20)

    def forward(self, x):
        # This defines the prediction/inference actions
        # https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html#move-your-computational-code
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # This defines the train loop. It is independent of forward.
        # https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html#find-the-train-loop-meat
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # In this case, a validation step performs the same computation as the
        # training step. Feel free to use another metric for than the training
        # loss here.
        # For more information, see:
        # https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html#find-the-val-loop-meat
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/1.3.2/starter/converting.html#move-the-optimizer-s-and-schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    ###########################################################################
    #                      Command-line argument parsing                      #
    ###########################################################################

    # Create argument parser and add the PyTorch Lightning command-line
    # arguments to it:
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # You can add additional arguments as follows. For more information, see:
    # - https://docs.python.org/dev/library/argparse.html#module-argparse
    parser.add_argument("--num_images_in_dataset", type=int, default=512)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    ###########################################################################
    #                               Load dataset                              #
    ###########################################################################

    num_images = args.num_images_in_dataset
    n = 512  # Image dimensions are n x n

    # NOTE: Because of horovod, this code runs in each process separately. In
    # this example, the dataset is generated randomly. In general, it is advised
    # to be careful with random numbers in data loading code. In addition, bugs
    # can be introduced by setting num_workers > 0 in the data loader. This is
    # discussed in detail here:
    # - https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    dataset = create_dataset(n, num_images)
    # Split into train and validation set and create data loaders:
    train, val = random_split(dataset, [num_images // 2, num_images // 2])
    dl_train = DataLoader(
        train, num_workers=4, pin_memory=True, batch_size=args.batch_size
    )
    dl_val = DataLoader(val, num_workers=4, pin_memory=True, batch_size=args.batch_size)

    ###########################################################################
    #                                  Train                                  #
    ###########################################################################

    denoiser = Denoiser()
    trainer = pl.Trainer.from_argparse_args(
        args,
        # Use the Horovod backend. For other options, see:
        # - https://pytorch-lightning.readthedocs.io/en/1.3.2/clouds/cluster.html
        # - https://pytorch-lightning.readthedocs.io/en/1.3.2/advanced/multi_gpu.html
        distributed_backend="horovod",
        # Always use 1 GPU for each process:
        gpus=1,
    )
    # Run trainer:
    trainer.fit(denoiser, dl_train, dl_val)

    ###########################################################################
    #                                Post-train                               #
    ###########################################################################

    # Because of Horovod, there are multiple running processes. We only save and
    # reload the weights using one process to prevent them from overwriting each
    # other's data.
    if hvd.rank() == 0:
        # Intermediate weights are automatically saved in the lightning_logs
        # directory. Here, we manually save the final weights:
        trainer.save_checkpoint("./final_weights.ckpt")

        # The weights can be reloaded as follows:
        new_model = Denoiser.load_from_checkpoint(
            checkpoint_path="./final_weights.ckpt"
        )

        # Output of the trained network can be computed as follows.
        x, y = train[0]
        # compute output (adding and removing batch dimension)
        out = new_model(x[None])[0]

        def to_uint_img(img):
            rescaled = (img - img.min()) / (img.max() - img.min())
            return (rescaled * 255).to(torch.uint8).expand(3, -1, -1)

        torchvision.io.write_png(to_uint_img(x), "input.png", compression_level=6)
        torchvision.io.write_png(to_uint_img(y), "target.png", compression_level=6)
        torchvision.io.write_png(to_uint_img(out), "output.png", compression_level=6)


###############################################################################
#                                   Run main                                  #
###############################################################################

# For more info on this snippet, see: https://stackoverflow.com/a/419185
if __name__ == "__main__":
    main()

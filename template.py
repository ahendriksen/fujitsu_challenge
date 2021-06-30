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
    parser.add_argument("--num_generations", type=int, default=512)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--simulate_game_of_life", default=False, action="store_true")
    args = parser.parse_args()

    ###########################################################################
    #                               Load dataset                              #
    ###########################################################################

    num_generations = args.num_generations
    n = 512  # Image dimensions are n x n

    # NOTE: Because of horovod, this code runs in each process separately. The
    # generation of the game of life uses random number generation. In general,
    # it is advised to be careful with random numbers in data loading code. In
    # addition, bugs can be introduced by setting num_workers > 0 in the data
    # loader. This is discussed in detail here:
    # - https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    dataset = create_dataset(n, num_generations, args.simulate_game_of_life)
    # Split into train and validation set and create data loaders:
    train, val = random_split(dataset, [num_generations // 2, num_generations // 2])
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


###############################################################################
#  Appendix: Some code to create a Game of Life simulation in PyTorch         #
###############################################################################

# You probably do not need this :)


def create_dataset(n, num_generations, simulate_game_of_life=False):
    """Create a dataset

    The dataset contains n x n single-channel images. It contains
    `num_generations - 1` such images.
    """
    print("Simulating game of life: ", simulate_game_of_life)
    if simulate_game_of_life:
        initial = (torch.rand(1, 1, n, n) < 0.5).to(torch.int8)
        gol_iterator = iter_game_of_life(initial, num_generations)
        tqdm_gol_iterator = tqdm(gol_iterator, leave=False, desc="Create dataset")
        generations = torch.cat([g for g in tqdm_gol_iterator])
        generations = generations.to(torch.float)
    else:
        # just generate some random noise
        generations = torch.randn(num_generations + 1, 1, n, n)

    # x and y contain subsequent generations of the game of life.
    # x has some noise applied to it.
    x = generations[:-1]
    x = x + 0.5 * torch.randn_like(x)
    y = generations[1:]
    dataset = TensorDataset(x, y)
    return dataset


def iter_game_of_life(world, num_generations):
    # Game of life
    # From wikipedia:
    #
    # These rules, which compare the behavior of the automaton to real life, can
    # be condensed into the following:
    # - Any live cell with two or three live neighbours survives.
    # - Any dead cell with three live neighbours becomes a live cell.
    # - All other live cells die in the next generation. Similarly, all
    #   other dead cells stay dead.
    yield world

    w = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1],], dtype=torch.int8, device=world.device
    )[None, None]

    for _ in range(num_generations):
        neighbours = torch.conv2d(world, w, padding=1)

        alive = world == 1

        world = torch.logical_or(
            neighbours == 3,  # three neighbours and cell is dead or alive
            torch.logical_and(
                alive, neighbours == 2
            ),  # cell is alive and has two live neighbours
        ).to(torch.int8)

        yield world

    return None


###############################################################################
#                                   Run main                                  #
###############################################################################

# For more info on this snippet, see: https://stackoverflow.com/a/419185
if __name__ == "__main__":
    main()

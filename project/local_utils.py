import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union
from typing import Tuple
import tqdm
import matplotlib.pyplot as plt
import platform
import time
import numpy as np


class TimeMeasurement:
    def __init__(self, context_name: str, frames: int) -> None:
        self.context_name: str = context_name
        self.frames: int = frames
        self.begin: float = None
        self.end: float = None

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()

    @property
    def time(self) -> float:
        if self.begin is None or self.end is None:
            raise RuntimeError()
        return self.end - self.begin

    @property
    def fps(self):
        return self.frames / self.time

    def __str__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = int(t - h*60 - min*60)
        ms = int((t - np.floor(t))*1000)

        return f"Execution time: {h}:{min}:{s}:{ms}, processed {self.frames} frames, throughput: {self.fps} fps."

    def __repr__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = np.floor(t - h*60 - min*60)
        ms = np.floor((t - np.floor(t))*1000)

        return f'TimeMeasurement(context="{self.context_name}","{h}:{min}:{s}:{ms}", frames={self.frames}, throughput={self.fps})'


def display_tensor_as_img(t: torch.Tensor, title=''):
    t = t.reshape((1,) + t.shape[-2:])

    for i in range(t.shape[0]):
        plt.imshow(t[i,:,:])
        plt.title(title + str(i))
        plt.show()


class BaseMetic(ABC):

    @abstractmethod
    def __call__(self, y_pred, y_ref) -> Any:
        raise NotImplementedError()


def count_params(model: torch.nn.Module):
    num_of_params = 0
    for p in model.parameters():
        num_of_params += p.view(-1,1).shape[0]

    return num_of_params


def train_test_pass(model: torch.nn.Module,
                    data_generator: Callable,
                    criterion: Callable,
                    metric: BaseMetic,
                    optimizer: torch.optim.Optimizer = None,
                    update_period: int = None,
                    mode: str = 'test',
                    device = torch.device('cpu'),
                    repeat: int = 1) -> Tuple[torch.nn.Module, float, float]:
    """
    Train or test pass generator data through the model.

    :param model: network
    :param data_generator: data loader
    :param criterion: criterion / loss two arg function
    :param metric: metric object - two arg function
    :param optimizer: optimizer object. For test mode use None. Defaults to None
    :param update_period: number of batches of processing to update parameters. For test mode use None. Defaults to None
    :param mode: one of ['train', 'test']. Test mode is for evaluation. Train for training (includes gradient propagation), defaults to 'test'
    :param device: device to execute on.
    :return: model, loss_value, metric_value
    """
    print(f"Running on platform: {platform.platform()}, "
          f"machine: {platform.machine()}, "
          f"python_version: {platform.python_version()}, "
          f"processor: {platform.processor()}, "
          f"system: {platform.system()}, "
          )

    # change model mode to train or test
    if mode == 'train':
        model.train(True)

    elif mode == 'test':
        model.eval()

    else:
        raise RuntimeError("Unsupported mode.")

    # move model to device
    model = model.to(device)

    # reset model parameters' gradients with optimizer
    if mode == 'train':
        optimizer.zero_grad()

    total_loss: float = 0.0
    total_metric: float = 0.0
    samples_num: int = 0

    for i, (X, y_ref) in tqdm.tqdm(enumerate(data_generator), total = len(data_generator)):
        for _ in range(repeat):
            # convert tensors to device
            X = X.to(device)
            y_ref = y_ref.to(device)

            if mode == 'train':
                # process by network
                y_pred = model(X)
            else:
                with torch.no_grad():
                    y_pred = model(X)

            # calculate loss
            loss: torch.Tensor = criterion(y_pred, y_ref)

            if mode == 'train':
                # designate gradient based on loss
                loss.backward()

            if mode == 'train' and (i+1) % update_period == 0:
                # update parameters with optimizer
                optimizer.step()
                # gradient designation sums it's values from previous passes
                # there is needed zeroing stored values of gradient
                optimizer.zero_grad()

            # calculate metric
            metric_value = metric(y_pred, y_ref)

            total_loss += loss.item() * y_pred.shape[0]
            total_metric += metric_value.item() * y_pred.shape[0]
            samples_num += y_pred.shape[0]

    if samples_num == 0:
        return model, 0.0, 0.0

    return model, total_loss / samples_num, total_metric / samples_num


def training(model,
             train_loader,
             test_loader,
             loss_fcn,
             metric,
             optimizer,
             update_period,
             epoch_max,
             device) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    _summary_
    :param model: network
    :param data_generator: data loader
    :return: model, (loss_value, metric_value)

    :param model: network model
    :param train_loader: data loader for training
    :param test_loader: data loader for validation
    :param loss_fcn: criterion / loss two arg function
    :param metric: metric object - two arg function
    :param optimizer: optimizer object
    :param update_period: number of batches of processing to update parameters
    :param epoch_max: number of training epochs
    :param device: device to execute on.
    :return: model, dictionary with keys 'loss_train', 'loss_test', 'metric_train', 'metric_test'
    - each entry contains list of loss / metric value for given epoch
    """
    loss_train = []
    loss_test = []
    metric_train = []
    metric_test = []

    for e in range(epoch_max):
        epoch = e+1
        print(f'Epoch {epoch} / {epoch_max}: STARTED')
        print('TRAINING')
        net, loss, metric_value = train_test_pass(model,
                                         train_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='train',
                                         device=device)
        loss_train.append(loss)
        metric_train.append(metric_value)

        print('VALIDATION')
        net, loss, metric_value = train_test_pass(model,
                                         test_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='test',
                                         device=device)
        loss_test.append(loss)
        metric_test.append(metric_value)

        print(f'\rAfter epoch {epoch}: loss={loss_train[-1]:.4f} metric={metric_train[-1]:.4f} val_loss={loss_test[-1]:.4f} val_metric={metric_test[-1]:.4f}')
        print(f'Epoch {epoch} / {epoch_max}: FINISHED\n')

    return model, {'loss_train': loss_train,
                   'metric_train': metric_train,
                   'loss_test': loss_test,
                   'metric_test': metric_test}


def plot_history(history):
    plt.plot(history['loss_train'], label='train')
    plt.plot(history['loss_test'], label='test')
    plt.legend()
    plt.title("History of loss")
    plt.show()

    plt.plot(history['metric_train'], label='train')
    plt.plot(history['metric_test'], label='test')
    plt.legend()
    plt.title("History of metric")
    plt.show()


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 intermediate_channels: int,
                 kernel_size: Union[int, Tuple[int,int]],
                 ) -> None:
        super().__init__()
        self.L1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,
                                          out_channels=intermediate_channels,
                                          kernel_size=kernel_size,
                                          bias=False,
                                          padding=kernel_size//2),
                                nn.BatchNorm2d(intermediate_channels),
                                nn.ReLU()
                                )
        self.L2 = nn.Sequential(nn.Conv2d(in_channels=intermediate_channels,
                                          out_channels=input_channels,
                                          kernel_size=kernel_size,
                                          bias=False,
                                          padding=kernel_size//2),
                                nn.BatchNorm2d(input_channels),
                                nn.ReLU()
                                )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, self.L2(self.L1(x)))


class SimpleSegmenter(nn.Module):
    def __init__(self,
                 input_shape,
                 num_of_classes=10,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(input_shape[0], 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),

                                nn.Conv2d(64, num_of_classes, 1),
                                nn.Sigmoid(),
                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class CustomDataLoader:
    def __init__(self, data, labels, batch_size=1):
        self.batch_size = batch_size
        self.data = data
        self.labels = labels

    def __getitem__(self,index):
        if index >= len(self):
            raise StopIteration()

        beg = index*self.batch_size
        end = beg+self.batch_size
        return self.data[beg:end, ...], self.labels[beg:end, ...]

    def __len__(self):
        return len(self.data) // self.batch_size


class BinaryCrossEntropyLoss:
    def __init__(self, mul = 1) -> None:
        self.bce = torch.nn.BCELoss(reduce=None)
        self.mul = mul

    def __call__(self,
                 y_pred: torch.Tensor,
                 y_ref: torch.Tensor) -> Any:
        loss = self.bce(y_pred, y_ref) * (1 + y_ref.sum(dim=1, keepdim=True) * self.mul)

        return loss.mean()


class BinaryAccuracy(BaseMetic):
    def __init__(self) -> None:
        pass

    def __call__(self,
                 y_pred: torch.Tensor,
                 y_ref: torch.Tensor) -> Any:
        y_pred = y_pred > 0.5
        y_ref = y_ref > 0.5

        ok = y_pred == y_ref
        return ok.sum() / ok.numel()



import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(192, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = torch.sigmoid(self.outc(x))

        return logits

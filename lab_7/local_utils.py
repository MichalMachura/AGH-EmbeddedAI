import torch
from abc import ABC, abstractmethod
from typing import Any
from typing import Tuple
import tqdm
import matplotlib.pyplot as plt


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


class AccuracyMetic(BaseMetic):

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, y_pred: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        :param y_pred: tensor of shape (batch_size, num_of_classes) type float
        :param y_ref: tensor with shape (batch_size,) and type Long
        :return: scalar tensor with accuracy metric for batch
        """
        y_pred = y_pred.argmax(1)
        cmp = y_pred == y_ref
        # scalar value
        score: torch.Tensor = cmp.sum() / cmp.shape[0]

        return score


def count_params(model: torch.nn.Module):
    num_of_params = 0
    for p in model.parameters():
        num_of_params += p.view(-1,1).shape[0]

    return num_of_params


def train_test_pass(model: torch.nn.Module,
                    data_generator,
                    criterion,
                    metric: BaseMetic,
                    optimizer: torch.optim.Optimizer = None,
                    update_period: int = None,
                    mode: str = 'test',
                    device = torch.device('cpu')) -> Tuple[torch.nn.Module, float, float]:

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
    total_accuracy: float = 0.0
    samples_num: int = 0

    for i, (X, y_ref) in tqdm.tqdm(enumerate(data_generator),):
        # convert tensors to device
        X = X.to(device)
        y_ref = y_ref.to(device)

        # process by network
        y_pred = model(X)

        # calculate loss
        loss: torch.Tensor = criterion(y_pred, y_ref)

        # designate gradient based on loss
        loss.backward()

        if mode == 'train' and (i+1) % update_period == 0:
            # update parameters with optimizer
            optimizer.step()
            # gradient designation sums it's values from previous passes
            # there is needed zeroing stored values of gradient
            optimizer.zero_grad()

        # calculate accuracy
        accuracy = metric(y_pred, y_ref)

        total_loss += loss.item() * y_pred.shape[0]
        total_accuracy += accuracy.item() * y_pred.shape[0]
        samples_num += y_pred.shape[0]

    if samples_num == 0:
        return model, 0.0, 0.0

    return model, total_loss / samples_num, total_accuracy / samples_num


def training(model,
             train_loader,
             test_loader,
             loss_fcn,
             metric,
             optimizer,
             update_period,
             epoch_max,
             device):
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    for e in range(epoch_max):
        epoch = e+1
        print(f'Epoch {epoch} / {epoch_max}: STARTED')
        print('TRAINING', end='')
        net, loss, acc = train_test_pass(model,
                                         train_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='train',
                                         device=device)
        loss_train.append(loss)
        acc_train.append(acc)

        print('\rVALIDATION', end='')
        net, loss, acc = train_test_pass(model,
                                         test_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='test',
                                         device=device)
        loss_test.append(loss)
        acc_test.append(acc)

        print(
            f'\rAfter epoch {epoch}: loss={loss_train[-1]:.4f} acc={acc_train[-1]:.4f} val_loss={loss_test[-1]:.4f} val_acc={acc_test[-1]:.4f}')
        print(f'Epoch {epoch} / {epoch_max}: FINISHED\n')

    return model, {'loss_train': loss_train,
                   'acc_train': acc_train,
                   'loss_test': loss_test,
                   'acc_test': acc_test}

def plot_history(history):
    plt.plot(history['loss_train'], label='train')
    plt.plot(history['loss_test'], label='test')
    plt.legend()
    plt.title("History of loss")
    plt.show()

    plt.plot(history['acc_train'], label='train')
    plt.plot(history['acc_test'], label='test')
    plt.legend()
    plt.title("History of accuracy")
    plt.show()


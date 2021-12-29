import torch


class Logger:

    def __init__(self, epochs, batches, path=None):
        self.path = path
        self.data = torch.zeros(epochs, batches, 5)
        self.epochs = epochs
        self.batches = batches
        self.e = 0
        self.b = 0

    def log(
        self,
        results,
        elapsed_time
    ):
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, 0:4].mean(dim=0)
        times = self.data[self.e, self.b, 4]
        self.b += 1
        msg = self.msg((*means, times))
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            if self.path is not None:
                torch.save(self.data, self.path)
        return msg

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, elapsed_time = data
        msg = (
            f'{"":10}{self.e:<8}{self.b + 1:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
            f'{elapsed_time:<10.4f}'
        )
        return msg

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
        return msg


class Logger:

    def __init__(self, epochs, batches, path=None):
        self.path = path
        self.data = torch.zeros(epochs, batches, 5)
        self.epochs = epochs
        self.batches = batches
        self.e = 0
        self.b = 0

    def log(
        self,
        results,
        elapsed_time
    ):
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, 0:4].mean(dim=0)
        times = self.data[self.e, self.b, 4]
        self.b += 1
        msg = self.msg((*means, times))
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            if self.path is not None:
                torch.save(self.data, self.path)
        return msg

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, elapsed_time = data
        msg = (
            f'{"":10}{self.e:<8}{self.b + 1:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
            f'{elapsed_time:<10.4f}'
        )
        return msg

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
        return msg

    def __init__(self, epochs, batches, path=None):
        self.path = path
        self.data = torch.zeros(epochs, batches, 5)
        self.epochs = epochs
        self.batches = batches
        self.e = 0
        self.b = 0

    def log(
        self,
        results,
        elapsed_time
    ):
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, 0:4].mean(dim=0)
        times = self.data[self.e, self.b, 4]
        self.b += 1
        msg = self.msg((*means, times))
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            if self.path is not None:
                torch.save(self.data, self.path)
        return msg

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, elapsed_time = data
        msg = (
            f'{"":10}{self.e:<8}{self.b + 1:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
            f'{elapsed_time:<10.4f}'
        )
        return msg

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
        return msg

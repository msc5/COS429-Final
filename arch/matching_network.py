
import torch
import torch.nn as nn

from torchinfo import summary


class Conv(nn.Module):

    def __init__(self, fi, fo):
        super(Conv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(fi, fo, 3, padding='same'),
            nn.BatchNorm2d(fo),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.seq(x)


class Embed(nn.Module):

    def __init__(self, fi, fo):
        super(Embed, self).__init__()
        self.fo = fo
        self.seq = nn.Sequential(
            Conv(fi, fo),
            Conv(fo, fo),
            Conv(fo, fo),
            Conv(fo, fo),
        )

    def forward(self, x):
        # x: (kq, nm, c, h, w)
        kq, nm, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.seq(x)
        x = x.view(-1, self.fo)
        # x: (kq * nm, fo)
        return x


class Classifier(nn.Module):

    def __init__(self, device):
        self.device = device
        super(Classifier, self).__init__()

    def forward(self, x, shapes):
        k, n, q, m = shapes
        # y = y.type(torch.int64).repeat_interleave(m).unsqueeze(1)
        # b = torch.zeros(k * n, k).to(self.device)
        # y = b.scatter(0, y, 1)
        y = torch.eye(k).repeat_interleave(n, dim=0).to(self.device)
        pred = torch.mm(x, y)
        return y, pred


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, s, t):
        # L2 distance
        n, q = s.shape[0], t.shape[0]
        dist = (
            - s.unsqueeze(1).expand(n, q, -1)
            + t.unsqueeze(0).expand(n, q, -1)
        ).pow(2).sum(dim=2).T
        return dist


class MatchingNets(nn.Module):

    def __init__(self, device, fi, fo):
        super(MatchingNets, self).__init__()
        self.device = device
        self.f = Embed(fi, fo)
        self.g = Embed(fi, fo)
        self.distance = Distance()
        self.classify = Classifier(self.device)
        self.__name__ = 'MatchingNets'

    def forward(self, s, t, y):
        k, n, _, _, _ = s.shape
        q, m, _, _, _ = t.shape
        s = self.f(s)
        t = self.g(t)
        dist = self.distance(s, t)
        lab, pred = self.classify(dist, (k, n, q, m))
        return lab, pred.log()


if __name__ == '__main__':
    n = 10
    k = 8
    m = 5
    q = 12
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MatchingNets(device, 1, 64).to(device)
    summary(model, input_size=[
        (k, n, 1, 28, 28),
        (q, m, 1, 28, 28),
        (q, 1)
    ], device=device)

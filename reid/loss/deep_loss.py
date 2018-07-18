from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from ..evaluation_metrics import accuracy


class DeepLoss(nn.Module):
    def __init__(self, margin=0):
        super(DeepLoss, self).__init__()
        self.triplet_criterion = nn.MarginRankingLoss(margin=margin)
        self.soft_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, epoch, add_soft=0):
        cnn, rnn, main = inputs
        if epoch < add_soft:
            loss, prec = self.tri_loss(main, targets)
        else:
            loss_main, prec_main = self.tri_loss(main, targets)
            loss_cnn, prec_cnn = self.softmax(cnn, targets)
            loss_rnn, prec_rnn = self.softmax(rnn, targets)
            loss = loss_main + loss_cnn + loss_rnn
            prec = max(prec_main, prec_cnn, prec_rnn)
        return loss, prec

    def tri_loss(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.triplet_criterion(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

    def softmax(self, inputs, targets):
        loss = self.soft_criterion(inputs, targets)
        prec, = accuracy(inputs.data, targets.data)
        prec = prec[0]
        return loss, prec

    def normalize(self, inputs, p=2):
        outputs = inputs.pow(p) / inputs.pow(p).sum(dim=1, keepdim=True).expand_as(inputs)
        return outputs

    def fusion(self, dist, targets):
        pass

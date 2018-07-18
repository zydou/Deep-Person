# -*- coding: utf-8 -*-

from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

from torch.autograd import Variable
import torch


class DeepPerson(nn.Module):

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=751):
        super(DeepPerson, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            self.hiddenDim = 256
            # self.conv_reduce = nn.Conv2d(2048, 256, (1,1), 1)
            self.lstm1 = nn.LSTM(2048, self.hiddenDim, 1, bidirectional=True)
            self.lstm1_forward = nn.Linear(self.hiddenDim, self.hiddenDim)
            self.lstm1_backward = nn.Linear(self.hiddenDim, self.hiddenDim)
            self.lstm2 = nn.LSTM(self.hiddenDim, self.hiddenDim, 1, bidirectional=True)
            self.lstm2_forward = nn.Linear(self.hiddenDim, self.hiddenDim)
            self.lstm2_backward = nn.Linear(self.hiddenDim, self.hiddenDim)

            # Append new layers
            if self.has_embedding:
                # first split
                self.feat = nn.Linear(2048, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)

                # second split
                self.feat2 = nn.Linear(2048, self.num_features)
                self.feat_bn2 = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat2.weight, mode='fan_out')
                init.constant(self.feat2.bias, 0)
                init.constant(self.feat_bn2.weight, 1)
                init.constant(self.feat_bn2.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
                self.drop2 = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

                self.classifier2 = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier2.weight, std=0.001)
                init.constant(self.classifier2.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        # first split (raw cnn)
        x1 = F.avg_pool2d(x, x.size()[2:])  # bx2048x1x1
        x1 = x1.squeeze()  # bx2048

        # second split (rnn)
        x2 = F.avg_pool2d(x, (1, x.size()[3]))  # bx2048x8x1
        x2 = x2.squeeze()  # bx2048x8

        batchSize, seq_len = x2.size(0), x2.size(2)

        h0 = Variable(torch.zeros(2, x2.size(0), self.hiddenDim)).cuda()
        c0 = Variable(torch.zeros(2, x2.size(0), self.hiddenDim)).cuda()

        x2 = x2.transpose(2, 1)  # bx8x2048
        x2 = x2.transpose(1, 0)  # 8xbx2048

        output1, hn1 = self.lstm1(x2, (h0, c0))

        output1_forward = output1[:, :, : self.hiddenDim]
        output1_forward = output1_forward.resize(output1_forward.size(0) * output1_forward.size(1), self.hiddenDim)

        output1_backward = output1[:, :, self.hiddenDim:]
        output1_backward = output1_backward.resize(output1_backward.size(0) * output1_backward.size(1), self.hiddenDim)

        x2 = self.lstm1_forward(output1_forward) + self.lstm1_backward(output1_backward)  # (8xb) x256
        x2 = x2.view(seq_len, batchSize, -1)  # 8xbx256

        output2, hn2 = self.lstm2(x2, (h0, c0))
        hn2 = hn2[0]

        output2_forward = output2[:, :, : self.hiddenDim]
        output2_forward = output2_forward.resize(output2_forward.size(0) * output2_forward.size(1), self.hiddenDim)

        output2_backward = output2[:, :, self.hiddenDim:]
        output2_backward = output2_backward.resize(output2_backward.size(0) * output2_backward.size(1), self.hiddenDim)

        x2 = self.lstm2_forward(output2_forward) + self.lstm2_backward(output2_backward)  # (8xb) x256
        x2 = x2.view(seq_len, batchSize, -1)  # 8xbx256
        x2 = x2.transpose(1, 0)  # bx8x256
        x2 = x2.resize(batchSize, seq_len * self.hiddenDim)  # bx2048

        # third branch(resnet cnn)
        x3 = x1.clone()

        if self.has_embedding:
            x1 = self.feat(x1)
            x1 = self.feat_bn(x1)

            x2 = self.feat2(x2)
            x2 = self.feat_bn2(x2)

        if self.norm:
            x1 = F.normalize(x1)
            x2 = F.normalize(x2)
        elif self.has_embedding:
            x1 = F.relu(x1)
            x2 = F.relu(x2)

        if self.dropout > 0:
            x1 = self.drop(x1)
            x2 = self.drop2(x2)
        if self.num_classes > 0:
            x1 = self.classifier(x1)
            x2 = self.classifier2(x2)

        # x1: cnn  x2:rnn x3: main branch
        return x1, x2, x3

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

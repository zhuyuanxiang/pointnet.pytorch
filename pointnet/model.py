from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from tools import log_debug
from tools import log_subtitle
from tools import log_title


class STN3d(nn.Module):
    """T-Net, Input Transformer, 生成 3x3 的转换矩阵"""

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        log_debug(f"x.size={x.size()}")
        x = F.relu(self.bn1(self.conv1(x)))
        log_debug(f"conv1.size={x.size()}")
        x = F.relu(self.bn2(self.conv2(x)))
        log_debug(f"conv2.size={x.size()}")
        x = F.relu(self.bn3(self.conv3(x)))
        log_debug(f"conv3.size={x.size()}")
        x = torch.max(x, 2, keepdim=True)[0]
        log_debug(f"max.size={x.size()}")
        x = x.view(-1, 1024)
        log_debug(f"view.size={x.size()}")

        x = F.relu(self.bn4(self.fc1(x)))
        log_debug(f"fc1.size={x.size()}")
        x = F.relu(self.bn5(self.fc2(x)))
        log_debug(f"fc2.size={x.size()}")
        x = self.fc3(x)
        log_debug(f"fc3.size={x.size()}")

        identify_matrix = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        iden = Variable(identify_matrix).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """T-Net, Feature Transformer, 生成 64x64 的转换矩阵"""

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        log_debug(f"x.size={x.size()}")
        x = F.relu(self.bn1(self.conv1(x)))
        log_debug(f"conv1.size={x.size()}")
        x = F.relu(self.bn2(self.conv2(x)))
        log_debug(f"conv2.size={x.size()}")
        x = F.relu(self.bn3(self.conv3(x)))
        log_debug(f"conv3.size={x.size()}")
        x = torch.max(x, 2, keepdim=True)[0]
        log_debug(f"max.size={x.size()}")
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        log_debug(f"fc1.size={x.size()}")
        x = F.relu(self.bn5(self.fc2(x)))
        log_debug(f"fc2.size={x.size()}")
        x = self.fc3(x)
        log_debug(f"fc3.size={x.size()}")

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
                batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """特征生成网络：global_feat：只生成全局特征用于分类；还是生成局部特征+全局特征进行拼接，用于分割 """

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        log_debug(f"x.size={x.size()}")
        n_pts = x.size()[2]
        log_subtitle("stn3d")
        trans = self.stn(x)
        log_debug(f"stn3d.size={trans.size()}")
        x = x.transpose(2, 1)
        log_debug(f"transpose.size={x.size()}")
        # torch.bmm(a,b):计算两个张量的矩阵乘法，a.size=(b,h,w),b.size=(b,w,h),两个张量的维度必须为3
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        log_debug(f"conv1.size={x.size()}")

        if self.feature_transform:
            log_subtitle("stnkd")
            log_debug(f"x.size={x.size()}")
            trans_feat = self.fstn(x)
            log_debug(f"stnkd.size={trans_feat.size()}")
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        log_subtitle("point_feat")
        pointfeat = x
        log_debug(f"x.size={x.size()}")
        x = F.relu(self.bn2(self.conv2(x)))
        log_debug(f"conv2.size={x.size()}")
        x = self.bn3(self.conv3(x))
        log_debug(f"conv3.size={x.size()}")
        x = torch.max(x, 2, keepdim=True)[0]
        log_debug(f"max.size={x.size()}")
        x = x.view(-1, 1024)
        log_debug(f"view.size={x.size()}")
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    """PointNet 分类器网络"""

    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    """PointNet 分割器网络"""

    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        log_subtitle("PointNetDenseCls")
        log_debug(f"x.size={x.size()}")
        x = F.relu(self.bn1(self.conv1(x)))
        log_debug(f"conv1.size={x.size()}")
        x = F.relu(self.bn2(self.conv2(x)))
        log_debug(f"conv2.size={x.size()}")
        x = F.relu(self.bn3(self.conv3(x)))
        log_debug(f"conv3.size={x.size()}")
        x = self.conv4(x)
        log_debug(f"conv4.size={x.size()}")
        x = x.transpose(2, 1).contiguous()
        log_debug(f"transpose.size={x.size()}")
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        log_debug(f"log_softmax.size={x.size()}#将32个样本2500个点混合在一起，切割成不同区域的点")
        x = x.view(batchsize, n_pts, self.k)
        log_debug(f"x.size={x.size()}")
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def test_stn3d():
    log_title("stn3d-model")
    trans = STN3d()
    out = trans(sim_data)
    log_subtitle("stn3d-data")
    log_debug(f"sim_data.size={sim_data.size()}")
    log_debug(f"stn.size={out.size()}")
    log_debug(f"loss={feature_transform_regularizer(out)}")


def test_stnkd():
    log_title("stnkd")
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    log_debug(f"sim_data_64d.size={sim_data_64d.size()}")
    log_debug(f"stn.size={out.size()}")
    log_debug(f"loss={feature_transform_regularizer(out)}")


def test_pointnet_feat():
    log_title("PointNetFeat(global_feat=" + str(global_feature))
    pointfeat = PointNetFeat(global_feat=global_feature)
    out, _, _ = pointfeat(sim_data)
    log_debug(f"global feat.size={out.size()}")


def test_pointnet_cls():
    log_title("PointNetCls")
    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    log_debug(f"class.size={out.size()}")


def test_pointnet_dense_cls():
    log_title("PointNetDenseCls")
    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    log_debug(f"seg.size={out.size()}")


if __name__ == '__main__':
    # 批处理的大小：32条数据；3个特征；每条数据的数据点个数：2500个点
    sim_data = Variable(torch.rand(32, 3, 2500))
    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    global_feature = True

    # test_stn3d()
    # test_stnkd()
    # test_pointnet_feat()
    #
    global_feature=False
    # test_pointnet_feat()
    #
    test_pointnet_cls()
    #
    test_pointnet_dense_cls()

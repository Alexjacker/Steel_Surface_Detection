# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 9:21
# @Author  : Leii
# @File    : models.py
# @Code instructions:
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import build_targets, to_cpu, non_max_suppression
import warnings

warnings.filterwarnings('ignore')


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class VBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1, down_stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=down_stride)
        self.outrelu = nn.Softplus()

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.relu(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y = self.bn3(self.conv4(Y))

        return self.outrelu(Y)


class FBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.relu = nn.LeakyReLU(inplace=True)
        self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, X):
        Y = self.relu(self.bn(self.conv(X)))
        Y = self.down_sample(Y)
        return self.relu(Y)


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, X):
        X = F.interpolate(X, scale_factor=self.scale_factor, mode=self.mode)
        return X


class VFTransision(nn.Module):
    def __init__(self):
        super().__init__()
        self.Vnet1 = VBLOCK(3, 64, use_1x1conv=True)
        self.Vnet2 = VBLOCK(64, 128, use_1x1conv=True)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)
        self.Fnet1 = FBLOCK(3, 64)
        self.Fnet2 = FBLOCK(64, 128)
        self.Fnet3 = FBLOCK(128, 256)
        self.Fnet4 = FBLOCK(256, 512)
        self.upsampling = Upsample(scale_factor=2)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, X):
        # 第一次VF交互
        Y_v_1 = self.Vnet1(X)  # input：3*224*224 output：64*112*112
        Y_f_1 = self.relu(self.conv1(X))  # input：3*224*224 output：3*112*112
        Y_f_2 = self.Fnet1(Y_f_1)  # input：3*112*112 output：64*56*56
        Y_f_2_us = self.upsampling(Y_f_2)  # 上采样， input：64*56*56 output：64*112*112

        Y_v_2 = Y_v_1 + Y_f_2_us  # 合并， 64*112*112

        Y_v_2 = self.relu(self.conv2(Y_v_2))  # 消除混叠， 64*112*112
        # 第二次VF交互
        Y_v_3 = self.Vnet2(Y_v_2)  # input：64*112*112 output：128*56*56
        Y_f_3 = self.Fnet2(Y_f_2)  # input：64*56*56 output：128*28*28
        # Y_f_3_us = self.upsampling(Y_f_3)  # 上采样， input：128*28*28 output：128*56*56
        Y_f_4 = self.Fnet3(Y_f_3)  # input：128*28*28 output：256*14*14
        Y_f_5 = self.Fnet4(Y_f_4)  # input：256*14*14 output：512*7*7

        # FPN
        Y_f_5_us = self.conv3(self.upsampling(Y_f_5))  # input：512*7*7 output：256*14*14

        Y_f_5_add = Y_f_4 + Y_f_5_us  # 合并，256*14*14

        Y_f_4_us = self.conv4(self.upsampling(Y_f_5_add))  # input：256*14*14 output：128*28*28

        Y_f_4_add = Y_f_3 + Y_f_4_us  # 合并，128*28*28

        Y_f_3_add = Y_f_3 + Y_f_4_add  # 合并，128*28*28

        Y_f_3_us = self.upsampling(Y_f_3_add)  # input：128*28*28 output：128*56*56
        Y_cat = torch.cat((Y_v_3, Y_f_3_us), axis=1)  # output：256*56*56
        return Y_cat


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):  #### 5,9,13
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class VFSNLayer(nn.Module):
    def __init__(self, anchor_idx=None, anchors=None, num_classes=None, img_dim=224):
        super(VFSNLayer, self).__init__()
        self.anchor_idxs = [int(x) for x in anchor_idx]  # 获取当前特征图所需的anchor索引
        self.anchors = [int(x) for x in anchors]  # 获取当前anchors列表里所有的anchors
        # 将anchors内容按tuple存储到anchors_slice
        self.anchors_slice = [(self.anchors[i], self.anchors[i + 1]) for i in range(0, len(self.anchors), 2)]
        self.target_anchors = [self.anchors_slice[i] for i in self.anchor_idxs]  # 获取当前特征图索引对应的anchor
        print(self.target_anchors)
        self.num_anchors = len(self.target_anchors)  # 当前anchor的数量，默认是3
        self.num_classes = num_classes  # 类别数
        self.ignore_thres = 0.5  # 置信度阈值
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1  #
        self.noobj_scale = 100  #
        self.metrics = {}  #
        self.img_dim = img_dim  # 图像尺寸
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size  # 获取当前grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = int(self.img_dim) / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride)
                                           for a_w, a_h in self.target_anchors])

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):
        # Tensors for cuda support
        # shape: batch_size, num_anchors*(num_classes + 5)， gride_size, gride_size
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        num_samples = x.size(0)  # batch_size
        grid_size = x.size(2)  # 网格大小
        # num_samples：batch_size
        # num_anchors：3
        # num_classes + 5：（x，y，w，h，置信度）+ 类别
        # grid_size：网格个数
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)  # x.view变换数据维度
                .permute(0, 1, 3, 4, 2)  # 输出的shape：batch_size * 3 * grid_size * grid_size * （5 + num_classes）
                .contiguous()  # 断开与前面tensor的联系，即重新分配一个tensor而不是改变元数据
        )
        # Get outputs
        # shape: (x, y, w, h, c, num_class)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            # 相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors #特征图中的实际位置
        # 这个操作很绝，x，y，w，h里都是存在相对于当前cell的相对值，而建立的grid里是每个cell的左上角坐标
        # 对应位置的相对值和cell坐标相加就能得到绝对位置
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x  # torch.data 新建一个和原tensor公用数据的tensor，并且不可求梯度
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # 还原到原始图中
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:  # 测试时的数据是没有标签的，直接返回结果
            return output, 0
        else:
            # iou_scores：真实值与最匹配的anchor的IOU得分值
            # class_mask：分类正确的索引
            # obj_mask：目标框所在位置的最好anchor置为1
            # noobj_mask obj_mask那里置0，还有计算的iou大于阈值的也置0，其他都为1
            # tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值
            # tconf 目标置信度
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,  # 预测的坐标值
                pred_cls=pred_cls,  # 预测的类别(one_hot)
                target=targets,  # gt标签
                anchors=self.scaled_anchors,  # 相对坐标的anchor
                ignore_thres=self.ignore_thres,  # IOU阈值
            )
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # 只计算有目标的
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj  # 有物体越接近1越好 没物体的越接近0越好

            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])  # 分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  # 总损失

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class VFSN(nn.Module):
    def __init__(self, img_size=224, anchors=None, num_classes=None, depth=4):
        super(VFSN, self).__init__()
        self.anchors = anchors  # anchor列表
        self.depth = depth  # 提取深度(VFTransision后接VBLOCK数量）
        self.num_classes = num_classes  # 类别
        self.VFTransision = VFTransision()
        self.VBLOCK = VBLOCK(256, 256, use_1x1conv=True, down_stride=1)
        self.VBLOCK_down = VBLOCK(256, 256, use_1x1conv=True)
        self.SPP = SPP(256, 33)
        self.VFSNLayer_14 = VFSNLayer(anchor_idx=[6, 7, 8], anchors=self.anchors, num_classes=self.num_classes,
                                      img_dim=img_size)
        self.VFSNLayer_28 = VFSNLayer(anchor_idx=[3, 4, 5], anchors=self.anchors, num_classes=self.num_classes,
                                      img_dim=img_size)
        self.VFSNLayer_56 = VFSNLayer(anchor_idx=[0, 1, 2], anchors=self.anchors, num_classes=self.num_classes,
                                      img_dim=img_size)
        self.VFSNLayers = [self.VFSNLayer_14, self.VFSNLayer_28, self.VFSNLayer_56]

    def forward(self, X, target=None):
        img_dim = X.shape[2]
        VFSN_outputs = []
        X = self.VFTransision(X)
        for i in range(self.depth):
            X = self.VBLOCK(X)
        Y_28 = self.VBLOCK_down(X)
        Y_14 = self.VBLOCK_down(Y_28)
        Y_56 = self.SPP(X)  # 小尺寸检测
        Y_28 = self.SPP(Y_28)  # 中尺寸检测
        Y_14 = self.SPP(Y_14)  # 大尺寸检测
        output_14, layer_loss_14 = self.VFSNLayer_14(Y_14, target)
        VFSN_outputs.append(output_14)
        output_28, layer_loss_28 = self.VFSNLayer_28(Y_28, target)
        VFSN_outputs.append(output_28)
        output_56, layer_loss_56 = self.VFSNLayer_56(Y_56, target)
        VFSN_outputs.append(output_56)
        loss = layer_loss_14 + layer_loss_28 + layer_loss_56
        return VFSN_outputs if target is None else (loss, VFSN_outputs)


if __name__ == '__main__':
    X_14 = torch.rand(64, 33, 14, 14)
    X_28 = torch.rand(64, 33, 28, 28)
    X_56 = torch.rand(64, 33, 56, 56)
    anchors_new = []
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    for i in anchors:
        anchors_new.append(i / 416 * 224)
    VFSNLayer_14 = VFSNLayer(anchor_idx=[6, 7, 8], anchors=anchors_new, num_classes=6,
                             img_dim=224)
    VFSNLayer_28 = VFSNLayer(anchor_idx=[3, 4, 5], anchors=anchors_new, num_classes=6,
                                  img_dim=224)
    VFSNLayer_56 = VFSNLayer(anchor_idx=[0, 1, 2], anchors=anchors_new, num_classes=6,
                                  img_dim=224)
    Y_14 = VFSNLayer_14(X_14)
    Y_28 = VFSNLayer_14(X_28)
    Y_56 = VFSNLayer_14(X_56)
    print(Y_14[0][0][0])
    print(Y_28[0][0][0])
    print(Y_56[0][0][0])
    # spp = SPP(256, 33)
    # Y = spp(Y)
    # print(Y.shape)
    # img_size = 224
    # anchors = [(116, 90), (156, 198), (373, 326)]
    # anchors = [(anchor_w / 416 * img_size, anchor_h / 416 * img_size) for (anchor_w, anchor_h) in anchors]
    # g = 14
    # stride = img_size / g
    # FloatTensor = torch.FloatTensor
    # grid_size_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(torch.FloatTensor)
    # grid_size_y = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(torch.FloatTensor)
    # scaled_anchors = FloatTensor([(a_w / stride, a_h / stride)
    #                               for a_w, a_h in anchors])
    #
    # anchor_w = scaled_anchors[:, 0:1].view((1, 3, 1, 1))
    # anchor_h = scaled_anchors[:, 1:2].view((1, 3, 1, 1))
    # print(grid_size_x)
    # # print(grid_size_y)
    # # print(anchor_w)
    # # print(anchor_h)
    # prediction = torch.rand(4, 3, 14, 14, 85)
    # x = torch.sigmoid(prediction[..., 0])  # Center x
    # y = torch.sigmoid(prediction[..., 1])  # Center y
    # w = prediction[..., 2]  # Width
    # h = prediction[..., 3]  # Height
    # print(x.shape, y.shape, w.shape, h.shape)
    # pred_boxes = FloatTensor(prediction[..., :4].shape)
    # pred_boxes[..., 0] = x.detach() + grid_size_x
    # pred_boxes[..., 1] = y.detach() + grid_size_y
    # pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    # pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    # print(pred_boxes[..., 0][0].shape, '\n占位\n',
    #       pred_boxes[..., 1][0].shape, '\n占位\n',
    #       pred_boxes[..., 2][0].shape, '\n占位\n',
    #       pred_boxes[..., 3][0].shape, '\n占位\n',
    #       pred_boxes[0].shape)

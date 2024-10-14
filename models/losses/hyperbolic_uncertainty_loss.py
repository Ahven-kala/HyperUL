"""
    Function: considering hyperbolic uncertainty in the conventional cross entropy loss.

    Date: October 22, 2021.
    Updated: November 22, 2021. Adding more variants of HyperbolicUncertaintyLoss.
    Updated: July 12, 2022. Replacing Geoopt library with Hyptorch library.
"""

import torch
import torch.nn as nn
import libs.hyptorch.pmath as pmath


class HyperbolicUncertaintyLoss(nn.Module):
    """The implementation of HyperUL-CE."""
    def __init__(self, class_weights=None, num_classes=19, ignore_index=255, reduction="sum", c=0.1, t=2.718, hr=1.0):
        """
        :param class_weights: weight for each class.
        :param ignore_index: ignored classes.
        :param reduction: "none", "mean" or "sum".
        :param c: a constant negative curvature, like 0.1, 0.3, 0.5, 0.7, 0.9, or 1.0.
        :param hr: 0.3, 0.5, 0.7, 0.9, or 1.0 in [0, 1], used to choose top x% uncertain pixels.
        """
        super(HyperbolicUncertaintyLoss, self).__init__()
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.c = c
        self.t = t
        self.hr = hr
        if class_weights is None:
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
            self.class_weights = torch.ones(self.num_classes).cuda()
        else:
            self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction='none')

    def forward(self, predict, gt):
        r"""
        compute the hyperbolic uncertainty loss.
        weights of uncertain pixels.

            hyper\_weights = \frac{1}{log_e(t + p_{uncertain\_pixels})}

            if  hyper_dist <= threshold_hyper_dist, weights for uncertain pixels are hyper\_weights.
            weights for the rest pixels are 1.0.

        :param predict: tensor, B x C x H x W
        :param gt: tensor, B x H x W
        :return: scalar.
        """
        # (1/4) compute the hyperbolic distances from all points to origin in an image.
        hyper_dist = self._compute_hyper_dist_to_ori(x=predict)  # B x H x W.

        B, H, W = hyper_dist.shape
        hyper_dist = hyper_dist.view(B, -1)  # B x H*W.

        min_hyper_dist, _ = torch.min(hyper_dist, dim=-1, keepdim=True)  # B x 1
        max_hyper_dist, _ = torch.max(hyper_dist, dim=-1, keepdim=True)  # B x 1
        threshold_hyper_dist = min_hyper_dist + (max_hyper_dist - min_hyper_dist) * self.hr  # B x 1

        hyper_dist = hyper_dist.view(B, H, W)  # B x H x W
        threshold_hyper_dist = threshold_hyper_dist.view(B, 1, 1)  # B x 1 x 1
        hyper_mask = hyper_dist <= threshold_hyper_dist  # B x H x W

        max_hyper_dist = max_hyper_dist.view(B, 1, 1)
        # hyper_weights = ((1.0 /
        #                   torch.log(self.t + (hyper_dist / max_hyper_dist)))
        #                  + 1.0) * hyper_mask + (~hyper_mask)  # B x H x W

        hyper_weights = (1.0 / torch.log(self.t + (hyper_dist / max_hyper_dist))) * hyper_mask   # B x H x W

        # (2/4) compute the cross entropy loss.
        cross_entropy_losses = self.cross_entropy(predict, gt)  # B x H x W.

        # (3/4) compute the hyperbolic uncertainty loss.
        hyper_uncertainty_loss = hyper_weights * cross_entropy_losses

        # (4/4) when reduction is "mean" or "sum"
        if self.reduction == "sum":
            hyper_uncertainty_loss = hyper_uncertainty_loss.sum()
        elif self.reduction == "mean":
            num_indices = torch.histc(gt.float(), bins=len(self.class_weights), min=0, max=len(self.class_weights)-1)
            scalar = 1.0 / (num_indices * self.class_weights).sum()
            hyper_uncertainty_loss = hyper_uncertainty_loss.sum() * scalar
        return hyper_uncertainty_loss

    def _compute_hyper_dist_to_ori(self, x):
        """
        compute the hyperbolic distance to the origin for all points on an entire image.
        :param x: tensor. B x C x H x W.
        :return: tensor. B x H x W.
        """
        x = x.softmax(dim=1)  # x has been in Hyperbolic space
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # B x C x H x W -> B x H x W x C
        x = x.view(b*h*w, c)  # B*H*W x C
        dist = pmath.dist0(x, c=self.c, keepdim=True)  # B*H*W x 1
        dist = dist.view(b, h, w)  # B*H*W x 1 -> B x H x W
        return dist


if __name__ == "__main__":
    loss1 = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 2.0, 3.0]), reduction="none")
    loss2 = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 2.0, 3.0]), reduction="sum")
    loss3 = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 2.0, 3.0]), reduction="mean")

    input = torch.randn(2, 3, 5, 5, requires_grad=True)  # feature dims = 3.
    target = torch.empty(2, 5, 5, dtype=torch.long).random_(3)

    output_loss1 = loss1(input, target).sum()
    # print(output_loss1)

    output_loss2 = loss2(input, target)
    print("output_loss1 = {0}, output_loss2 = {1}".format(output_loss1, output_loss2))

    output_loss3 = loss3(input, target)
    num_indices = torch.histc(target.float(), bins=3, min=0, max=3-1)
    class_weights = torch.Tensor([1.0, 2.0, 3.0])
    scalar = 1.0 / (num_indices * class_weights).sum()
    output_loss33 = output_loss2 * scalar
    print("output_loss3 = {0}, output_loss33 = {1}".format(output_loss3, output_loss33))

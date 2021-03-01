from torch.nn.functional import l1_loss, cross_entropy
import torch
import numpy as np


class BaseLoss:
    def __init__(self, hm_ratio=1, wh_ratio=0.1, offset_ratio=1):
        self.hm_ratio = hm_ratio
        self.wh_ratio = wh_ratio
        self.offset_ratio = offset_ratio

    def __call__(self, pred, true):
        gthm, gthw, gtos, gtmask = true["hm"], true["hw"], true["offset"], true["mask"]
        phm, phw, pos = pred["hm"], pred["hw"], pred["offset"]

        positive_mask = gthm > 0
        num_total = np.prod(gthm.shape) + 1e-5
        negative_weights = (positive_mask.sum() + 1e-5) / num_total
        positive_weights = 1 - negative_weights

        positive_loss = - gthm[positive_mask] * torch.log(phm[positive_mask]) * positive_weights
        negative_loss = - gthm[~positive_mask] * torch.log(phm[~positive_mask]) * negative_weights

        hm_loss = (positive_loss + negative_loss) * self.hm_ratio
        wh_loss = l1_loss(phw[gtmask], gthw[gtmask]) * self.wh_ratio
        offset_loss = l1_loss(pos[gtmask], gtos[gtmask]) * self.offset_ratio

        total_loss = hm_loss + wh_loss + offset_loss
        info = {
            "hm": hm_loss.item(),
            "wh": wh_loss.item(),
            "offset": offset_loss.item()
        }
        return total_loss, info

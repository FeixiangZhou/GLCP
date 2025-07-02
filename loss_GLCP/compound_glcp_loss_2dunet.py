import torch
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.cldice_loss import SoftclDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize
from nnunetv2.training.loss.cbdice_loss import SoftcbDiceLoss, SoftclMDiceLoss


import torch.nn.functional as F
import numpy as np
criterion_BCE_withlogits = nn.BCEWithLogitsLoss(reduction="mean")
criterion_BCE = nn.BCELoss(reduction="mean")
kl_loss = nn.KLDivLoss(reduction='mean')


def consistency_constraint_loss_mse(outputs_skeleton, prob_skeleton):
    outputs_skeleton = F.softmax(outputs_skeleton, dim=1)
    # outputs_skeleton = outputs_skeleton[:,1]
    truncated_outputs_skeleton = outputs_skeleton.detach() 
    truncated_prob_skeleton = prob_skeleton.detach()   

    mse_loss_1 = F.mse_loss(outputs_skeleton, truncated_prob_skeleton)
    mse_loss_2 = F.mse_loss(prob_skeleton, truncated_outputs_skeleton)
    Lcon = mse_loss_1 + mse_loss_2
    return Lcon


def consistency_constraint_loss_kl(outputs_skeleton, prob_skeleton):
    outputs_skeleton = F.softmax(outputs_skeleton, dim=1)
    outputs_skeleton = torch.clamp(outputs_skeleton, min=1e-6, max=1) 
    prob_skeleton = torch.clamp(prob_skeleton, min=1e-6, max=1) 

    truncated_outputs_skeleton = outputs_skeleton.detach()
    truncated_prob_skeleton = prob_skeleton.detach() 

    # outputs_skeleton is prediction/prob_skeleton is GT
    kl_loss_1 = F.kl_div(torch.log(outputs_skeleton), truncated_prob_skeleton, reduction='mean')

    # prob_skeleton is prediction/outputs_skeleton is GT
    kl_loss_2 = F.kl_div(torch.log(prob_skeleton), truncated_outputs_skeleton, reduction='mean')
    Lcon = kl_loss_1 + kl_loss_2
    return Lcon



def weighted_ce_Loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):

    weight_ce =weight.clone()
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(logits, target)
    probs_weight = torch.softmax(weight_ce, dim=1)
    probs_weight = probs_weight[:,1]
    probs_weight_np = probs_weight.detach().cpu().numpy()
    probs_weight_np[probs_weight_np >= 0.5] *= 2
    probs_weight_np[probs_weight_np < 0.5] = 1
    probs_weight_tensor  = torch.tensor(probs_weight_np).to(logits.device)
    loss = probs_weight_tensor * loss
    return loss.mean()


def weighted_bce_Loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    weight_ce =weight.clone()
    pos_probs = torch.softmax(logits, 1)

    pos_probs = pos_probs[:, 1]
    with torch.cuda.amp.autocast(enabled=False):
        loss = F.binary_cross_entropy(pos_probs,  target, reduction="mean")
    return loss.mean()

class weighted_soft_iou_loss(nn.Module):
    def __init__(self):
        super(weighted_soft_iou_loss, self).__init__()
    def forward(self, pred, label, weight):
        # pred = torch.sigmoid(pred)
        pred = torch.softmax(pred, 1)
        pred = pred[:, 1]
        weight_ce =weight.clone()

        probs_weight = torch.softmax(weight_ce, dim=1)
        probs_weight = probs_weight[:,1]
        probs_weight_np = probs_weight.detach().cpu().numpy()
        probs_weight_np[probs_weight_np >= 0.5] *= 2
        probs_weight_np[probs_weight_np < 0.5] = 1
        weit  = torch.tensor(probs_weight_np).to(pred.device)

        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        weit = weit.view(b, -1)
        inter_ = torch.mul(pred, label)
        inter = torch.sum(torch.mul(inter_, weit), dim=-1, keepdim=False)
        union_ = torch.mul(torch.mul(pred, pred) + label, weit)
        unit = torch.sum(union_, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / (unit + 1e-10))


class DC_and_CE_and_GLCP_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldc_kwargs, weight_ce=1, weight_dice=1, weight_cldice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_GLCP_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(**cldc_kwargs)
        self.m_skeletonize = SoftSkeletonize(num_iter=10)
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        self.cbdice = SoftcbDiceLoss(**cldc_kwargs)


        self.iou = weighted_soft_iou_loss()
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, output_ske=None, localwindow_fea=None, skeleton_gt=None, prob_ske=None, output_criticalregion=None, mask_criticalregion_eachbatch=None, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) + 1 \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long())\
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cbdice_loss = self.cbdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) + 1 if self.weight_cldice != 0 else 0


        #refine loss
        if localwindow_fea is not None:
            if (len(localwindow_fea) > 0):
                ce_loss_refine = self.ce(localwindow_fea, target[:, 0].long())  #refine CE
                dc_loss_refine = self.dc(localwindow_fea, target, loss_mask=None)  + 1
                # ce_loss_refine = ce_loss_refine + dc_loss_refine
            else:
                ce_loss_refine = 0
        else:
            ce_loss_refine = 0

        # skeleton loss
        if output_ske is not None:
            # with torch.no_grad():
            #     y_true = torch.where(target > 0, 1, 0).squeeze(1).float()  # ground truth of foreground #【2，32，32，32】
            #     skel_true = self.m_skeletonize(y_true.unsqueeze(1)) #[bs, 1, 96, 192, 128]]
            skel_true = skeleton_gt.unsqueeze(1)
            ce_loss_ske = self.ce(output_ske, skel_true[:, 0].long())
            dc_loss_ske = self.dc(output_ske, skel_true)
            ce_loss_ske = ce_loss_ske + 0.5 * (dc_loss_ske + 1)
        else:
            ce_loss_ske=0

        # criticalregion prediction loss
        if output_criticalregion is not None:
            bce_loss_criticalregion = self.ce(output_criticalregion, mask_criticalregion_eachbatch[:, 0].long())  #pre-2 channels with all 1
            dc_loss_criticalregion = self.dc(output_criticalregion, mask_criticalregion_eachbatch, loss_mask=None)
            bce_loss_criticalregion = bce_loss_criticalregion + 0.5 * (dc_loss_criticalregion + 1)
        else:
            bce_loss_criticalregion = 0

        if prob_ske is not None:
            # con_loss = consistency_constraint_loss_mse(output_ske, prob_ske)
            con_loss = consistency_constraint_loss_kl(output_ske, prob_ske)
            print("****consis_loss:", con_loss.item())
        else:
            con_loss = 0
        print("****ce_loss:", ce_loss.item() + dc_loss.item())
        print("****ce_loss_ske:", ce_loss_ske)
        print("****bce_loss_criticalregion:", bce_loss_criticalregion)
        print("****ce_loss_refine:", ce_loss_refine)
        result = ce_loss + self.weight_dice * dc_loss  + bce_loss_criticalregion + ce_loss_ske + 0.5* ce_loss_refine 
        return result


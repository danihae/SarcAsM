import torch
import torch.nn.functional as F
from torch import nn


class MaskBoundaryLoss(nn.Module):
    """
    Loss for :class:`~contraction_net.contraction_net.ContractionNet`.

    Combines a per-sample BCE + Dice term on the contraction mask with a positive-weighted
    BCE on the onset and offset channels.

    Two details matter:

    - The Dice term is computed **per sample** and then averaged. Pooling intersections and
      cardinalities across the batch instead would let whichever traces contain the most
      positive frames dominate the gradient -- that is, the high-duty traces, which are
      precisely the ones the model must not develop a bias about.
    - Boundaries carry a positive weight, because onsets and offsets occupy a handful of
      frames in a trace of hundreds and would otherwise be driven to zero.

    Parameters
    ----------
    dice_weight : float, optional
        Weight of the Dice term relative to mask BCE. Default is 1.0.
    boundary_weight : float, optional
        Weight of the boundary loss relative to the mask loss. Default is 0.5.
    boundary_pos_weight : float, optional
        Positive-class weight inside the boundary BCE. Default is 20.0.
    """

    def __init__(self, dice_weight=1.0, boundary_weight=0.5, boundary_pos_weight=20.0):
        super(MaskBoundaryLoss, self).__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.register_buffer('boundary_pos_weight', torch.tensor(float(boundary_pos_weight)))

    @staticmethod
    def per_sample_dice(logits, targets, epsilon=1e-6):
        """Soft Dice averaged over samples rather than pooled across the batch."""
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.dim()))
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
        return (1.0 - dice).mean()

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw output of shape ``(batch, 3, length)``: mask, onset, offset.
        targets : torch.Tensor
            Targets of the same shape.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        dict
            The individual terms, for logging.
        """
        n_out = logits.shape[1]
        mask_logits, mask_target = logits[:, :1], targets[:, :1]

        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
        dice = self.per_sample_dice(mask_logits, mask_target)
        total = bce + self.dice_weight * dice
        terms = {'bce': bce.item(), 'dice': dice.item(), 'boundary': 0.0}

        if n_out > 1:
            boundary_logits = logits[:, 1:]
            if n_out == 2:
                # a single head for both transitions: target either one
                boundary_target = targets[:, 1:3].max(dim=1, keepdim=True).values
            else:
                boundary_target = targets[:, 1:n_out]
            boundary = F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_target,
                pos_weight=self.boundary_pos_weight.to(boundary_logits.device))
            total = total + self.boundary_weight * boundary
            terms['boundary'] = boundary.item()
        return total, terms


def iou_score(logits, targets, threshold=0.0):
    """Mean per-sample IoU of the mask channel. ``threshold`` is on the logits."""
    pred = (logits[:, 0] > threshold)
    true = targets[:, 0] > 0.5
    dims = tuple(range(1, pred.dim()))
    inter = (pred & true).sum(dim=dims).float()
    union = (pred | true).sum(dim=dims).float()
    # a correctly empty prediction on a quiescent trace scores 1, not 0/0
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean()


def iou_at_thresholds(logits, targets, thresholds):
    """Per-sample IoU of the mask channel at several **probability** thresholds.

    Returned per sample rather than averaged so a caller can accumulate over an epoch and
    weight the pools itself. Scoring at one fixed threshold would penalise a model that is
    better but calibrated differently.

    Returns
    -------
    torch.Tensor
        Shape ``(len(thresholds), batch)``.
    """
    probs = torch.sigmoid(logits[:, 0])
    true = targets[:, 0] > 0.5
    dims = tuple(range(1, true.dim()))
    rows = []
    for threshold in thresholds:
        pred = probs > float(threshold)
        inter = (pred & true).sum(dim=dims).float()
        union = (pred | true).sum(dim=dims).float()
        rows.append(torch.where(union > 0, inter / union.clamp(min=1),
                                torch.ones_like(union)))
    return torch.stack(rows)

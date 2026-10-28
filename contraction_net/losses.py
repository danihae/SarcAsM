import torch
import torch.nn.functional as F
from torch import nn


def _mean(values, weight=None):
    """Mean over every element, or the mean weighted by a per-frame ``weight``.

    ``weight`` has shape ``(batch, 1, length)`` and is broadcast over the channels.
    """
    if weight is None:
        return values.mean()
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


class MaskBoundaryLoss(nn.Module):
    """
    Loss for :class:`~contraction_net.contraction_net.ContractionNet`.

    Combines a per-sample BCE + Dice term on the contraction mask with a positive-weighted
    BCE on the onset and offset channels.

    Three details matter:

    - The Dice term is computed **per sample** and then averaged. Pooling intersections and
      cardinalities across the batch instead would let whichever traces contain the most
      positive frames dominate the gradient -- that is, the high-duty traces, which are
      precisely the ones the model must not develop a bias about.
    - Boundaries carry a positive weight, because onsets and offsets occupy a handful of
      frames in a trace of hundreds and would otherwise be driven to zero.
    - A per-frame ``weight`` turns every term into a weighted mean, so the frames a short
      trace is padded with to fill its crop teach nothing.

    Parameters
    ----------
    dice_weight : float, optional
        Weight of the Dice term relative to mask BCE. Default is 1.0. Dice is constant on a
        crop without a contraction, so it gives such a crop almost no gradient; 0 leaves
        the mask to BCE alone.
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
    def per_sample_dice(logits, targets, epsilon=1e-6, weight=None):
        """Soft Dice averaged over samples rather than pooled across the batch."""
        probs = torch.sigmoid(logits)
        if weight is not None:
            probs, targets = probs * weight, targets * weight
        dims = tuple(range(1, probs.dim()))
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
        return (1.0 - dice).mean()

    def forward(self, logits, targets, weight=None):
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw output of shape ``(batch, 3, length)``: mask, onset, offset.
        targets : torch.Tensor
            Targets of the same shape.
        weight : torch.Tensor or None, optional
            Per-frame weight of shape ``(batch, length)``, 1 on real frames and 0 on
            padding. None weights every frame equally.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        dict
            The individual terms, for logging.
        """
        n_out = logits.shape[1]
        mask_logits, mask_target = logits[:, :1], targets[:, :1]
        if weight is not None:
            weight = weight.reshape(weight.shape[0], 1, -1).to(logits.dtype)

        bce = _mean(F.binary_cross_entropy_with_logits(mask_logits, mask_target,
                                                       reduction='none'), weight)
        dice = self.per_sample_dice(mask_logits, mask_target, weight=weight)
        total = bce + self.dice_weight * dice
        terms = {'bce': bce.item(), 'dice': dice.item(), 'boundary': 0.0}

        if n_out > 1:
            boundary_logits = logits[:, 1:]
            if n_out == 2:
                # a single head for both transitions: target either one
                boundary_target = targets[:, 1:3].max(dim=1, keepdim=True).values
            else:
                boundary_target = targets[:, 1:n_out]
            boundary = _mean(F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_target, reduction='none',
                pos_weight=self.boundary_pos_weight.to(boundary_logits.device)), weight)
            total = total + self.boundary_weight * boundary
            terms['boundary'] = boundary.item()
        return total, terms


def iou_score(logits, targets, threshold=0.0, weight=None):
    """Mean per-sample IoU of the mask channel. ``threshold`` is on the logits.

    Frames with zero ``weight`` (padding) count on neither side.
    """
    pred = (logits[:, 0] > threshold)
    true = targets[:, 0] > 0.5
    if weight is not None:
        pred, true = pred & (weight > 0), true & (weight > 0)
    dims = tuple(range(1, pred.dim()))
    inter = (pred & true).sum(dim=dims).float()
    union = (pred | true).sum(dim=dims).float()
    # a correctly empty prediction on a quiescent trace scores 1, not 0/0
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean()


def iou_at_thresholds(logits, targets, thresholds, weight=None):
    """Per-sample IoU of the mask channel at several **probability** thresholds.

    Returned per sample rather than averaged so a caller can accumulate over an epoch and
    weight the pools itself. Scoring at one fixed threshold would penalise a model that is
    better but calibrated differently. Frames with zero ``weight`` (padding) count on
    neither side.

    Returns
    -------
    torch.Tensor
        Shape ``(len(thresholds), batch)``.
    """
    probs = torch.sigmoid(logits[:, 0])
    true = targets[:, 0] > 0.5
    real = None if weight is None else weight > 0
    if real is not None:
        true = true & real
    dims = tuple(range(1, true.dim()))
    rows = []
    for threshold in thresholds:
        pred = probs > float(threshold)
        if real is not None:
            pred = pred & real
        inter = (pred & true).sum(dim=dims).float()
        union = (pred | true).sum(dim=dims).float()
        rows.append(torch.where(union > 0, inter / union.clamp(min=1),
                                torch.ones_like(union)))
    return torch.stack(rows)

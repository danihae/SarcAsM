import torch
import torch.nn.functional as F
from torch import nn


class BCEDiceLoss(nn.Module):
    """
    A combination of Binary Cross Entropy (BCE) and Dice Loss for binary segmentation tasks.

    Parameters
    ----------
    loss_params : tuple, optional
        A tuple containing the weights for BCE and Dice losses respectively. Default is (1, 1).

    Methods
    -------
    dice_loss(inputs, targets, epsilon=1e-6)
        Computes the Dice loss.

    forward(inputs, targets)
        Computes the combined BCE and Dice loss.
    """

    def __init__(self, loss_params=(1, 1)):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.loss_params = loss_params

    def dice_loss(self, inputs, targets, epsilon=1e-6):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + epsilon) / (inputs.sum() + targets.sum() + epsilon)
        return 1 - dice_coeff

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.loss_params[0] * bce + self.loss_params[1] * dice


class SmoothnessLoss(nn.Module):
    """
    Computes the smoothness loss for a sequence of predictions.

    Parameters
    ----------
    alpha : float, optional
        Weight of the smoothness loss component. Default is 10.

    Methods
    -------
    forward(predictions)
        Computes the smoothness loss for a sequence of predictions.
    """

    def __init__(self, alpha=10):
        super(SmoothnessLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions):
        if predictions.dim() < 3:
            raise ValueError("The input tensor must be 3-dimensional.")
        diffs = predictions[:, :, 1:] - predictions[:, :, :-1]
        loss = torch.sum(diffs ** 2) / predictions.size(0)
        return self.alpha * loss


def f1_score(logits, true_labels, threshold=0.5, epsilon=1e-7):
    """
    Computes the F1 score for binary classification.

    Parameters
    ----------
    logits : torch.Tensor
        The raw output from the model (before applying sigmoid).
    true_labels : torch.Tensor
        The ground truth binary labels.
    threshold : float, optional
        The threshold to convert probabilities to binary predictions. Default is 0.5.
    epsilon : float, optional
        A small value to avoid division by zero. Default is 1e-7.

    Returns
    -------
    float
        The computed F1 score.
    """
    probabilities = torch.sigmoid(logits)
    predictions = probabilities > threshold
    predictions = predictions.float()
    true_labels = true_labels.float()
    tp = (predictions * true_labels).sum().item()
    fp = ((1 - true_labels) * predictions).sum().item()
    fn = (true_labels * (1 - predictions)).sum().item()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score


class MaskBoundaryLoss(nn.Module):
    """
    Loss for :class:`~contraction_net.contraction_net.ContractionNetV2`.

    Combines a per-sample BCE + Dice term on the contraction mask with a positive-weighted
    BCE on the onset and offset channels.

    Two differences from :class:`BCEDiceLoss`, both of which were distorting training:

    - The Dice term is computed **per sample** and then averaged. The original summed
      intersections and cardinalities over the whole batch, so a batch's loss was dominated
      by whichever traces happened to contain the most positive frames -- exactly the
      high-duty traces whose behaviour is the problem being fixed.
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
        mask_logits, boundary_logits = logits[:, :1], logits[:, 1:]
        mask_target, boundary_target = targets[:, :1], targets[:, 1:]

        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
        dice = self.per_sample_dice(mask_logits, mask_target)
        boundary = F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_target,
            pos_weight=self.boundary_pos_weight.to(boundary_logits.device))

        total = bce + self.dice_weight * dice + self.boundary_weight * boundary
        return total, {'bce': bce.item(), 'dice': dice.item(), 'boundary': boundary.item()}


def iou_score(logits, targets, threshold=0.0):
    """Mean per-sample IoU of the mask channel. ``threshold`` is on the logits."""
    pred = (logits[:, 0] > threshold)
    true = targets[:, 0] > 0.5
    dims = tuple(range(1, pred.dim()))
    inter = (pred & true).sum(dim=dims).float()
    union = (pred | true).sum(dim=dims).float()
    # a correctly empty prediction on a quiescent trace scores 1, not 0/0
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean()

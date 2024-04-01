import torch
from torch import nn as nn, flatten


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean' if size_average else 'sum')

    def forward(self, logits, targets):
        return self.bce_loss(logits, targets)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = 2. * (intersection + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1 - score.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)


class logcoshDiceLoss(nn.Module):
    def __init__(self):
        super(logcoshDiceLoss, self).__init__()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        x = self.dice(logits, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class logcoshTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(logcoshTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.log(torch.cosh(1 - Tversky))


def dice_coefficient(logits, ground_truth, smooth=1.0):
    """
    Calculate the Dice coefficient between logits and ground truth masks.

    Args:
    - logits (torch.Tensor): The raw output from the model (before sigmoid).
    - ground_truth (torch.Tensor): The ground truth binary masks.
    - smooth (float): A smoothing factor to avoid division by zero.

    Returns:
    - dice (float): The Dice coefficient.
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Threshold probabilities to create binary prediction map
    preds = (probs > 0.5).float()

    # Flatten the predictions and ground truth
    preds_flat = preds.view(-1)
    ground_truth_flat = ground_truth.view(-1)

    # Calculate intersection and union
    intersection = (preds_flat * ground_truth_flat).sum()
    union = preds_flat.sum() + ground_truth_flat.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice
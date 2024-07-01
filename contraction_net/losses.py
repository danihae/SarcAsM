import torch
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

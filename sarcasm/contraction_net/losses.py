import torch
from torch import nn


class BCEDiceLoss(nn.Module):
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
    def __init__(self, alpha=10):
        """
        Initializes the SmoothnessLoss module.

        Parameters:
        - alpha: Weight of the smoothness loss component.
        """
        super(SmoothnessLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions):
        """
        Computes the smoothness loss for a sequence of predictions.

        Parameters:
        - predictions: Tensor containing the model's predictions. Expected shape is
                       (batch_size, 1, len_time_series) for univariate time-series.

        Returns:
        - loss: The computed smoothness loss.
        """
        # Ensure predictions is at least 2D
        if predictions.dim() < 3:
            raise ValueError("The input tensor must be 3-dimensional.")

        # Calculate differences between consecutive predictions along the time series dimension
        diffs = predictions[:, :, 1:] - predictions[:, :, :-1]
        # Compute the squared differences and sum over the time series dimension
        loss = torch.sum(diffs ** 2) / predictions.size(0)  # Normalize by batch size
        return self.alpha * loss


def f1_score(logits, true_labels, threshold=0.5, epsilon=1e-7):
    # Step 1: Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Step 2: Threshold probabilities to get binary predictions
    predictions = probabilities > threshold

    # Convert to float for calculation
    predictions = predictions.float()
    true_labels = true_labels.float()

    # True positives, false positives, false negatives
    tp = (predictions * true_labels).sum().item()
    fp = ((1 - true_labels) * predictions).sum().item()
    fn = (true_labels * (1 - predictions)).sum().item()

    # Step 3: Calculate precision and recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Step 4: Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score

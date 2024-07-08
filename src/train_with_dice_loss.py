# We implement a loss function based on the dice coefficient

def dice_score_new(input_, target):
    # Check if multi-class or binary segmentation
    num_classes = input_.shape[1]

    if num_classes == 1:
        # Binary segmentation
        input_sigmoid = torch.sigmoid(input_)
        intersection = (input_sigmoid * target).sum(dim=(2, 3))
        union = input_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    else:
        # Multi-class segmentation
        # One-hot encode target to match input shape
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)  # Rearrange dimensions

        # Apply softmax to predictions
        input_softmax = F.softmax(input_, dim=1)

        # Calculate intersection and union for each class
        intersection = (input_softmax * target_one_hot).sum(dim=(2, 3))
        union = input_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    # Compute Dice score
    dice = (2. * intersection + 1e-6) / (union + 1e-6)

    # Compute average over channels
    dice_score_average = dice.mean(dim=1)
    return dice_score_average.mean()

def dice_loss(pred, target):
    return 1 - dice_score_new(pred, target)

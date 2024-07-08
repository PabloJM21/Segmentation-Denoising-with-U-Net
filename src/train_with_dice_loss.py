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


# We implement and compare with different loss function variations
model = UNet(out_channels=3)
model.to(device)
loss = dice_loss
n_epochs = 10
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
metric = None
train_losses, _, val_losses, _ = run_training(model, train_loader, val_loader, loss, metric, optimizer, n_epochs)
plot('Loss vs. Epoch', 'Loss', train_losses, val_losses)

f1s = []
model.eval()
with torch.no_grad():
    for im, mask in tqdm.tqdm(zip(test_images, test_masks), total=len(test_images)):
        # 
        if any(sh % 16 != 0 for sh in im.shape):
            crop = tuple(
                slice(0, -(sh%16)) for sh in im.shape
            )
            im = im[crop]
            mask = mask[crop]
        
        input_ = torch.from_numpy(im[None, None]).to(device)
        pred = model(input_)
        pred = torch.softmax(pred, dim=1).cpu().numpy().squeeze()
        assert pred.shape[0] == 3
        nuclei = instance_segmentation(pred[1], pred[2])
        f1s.append(f1_score(nuclei, mask, best_threshold))
print()
print(np.mean(f1s))


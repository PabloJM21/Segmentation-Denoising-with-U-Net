# We implement a transform that outputs an image with 3 values:
# 0 for backgorund, 1 for foreground (= nucleus) and 2 for boundary (boundary pixel between nucleus and background or between 2 nuclei)
# HINT: you can use skimage.segmentation.find_boundaries (already imported) to determine the boundary pixels
def label_transform(mask):
    # Create a transformed mask with the same shape as the original mask
    transformed_mask = np.zeros(mask.shape, dtype=np.int64)
    
    # Foreground: Assign 1 to nucleus areas
    transformed_mask[mask > 0] = 1
    
    # Boundary: Use find_boundaries to detect boundaries and assign 2
    boundaries = find_boundaries(mask)
    transformed_mask[boundaries] = 2  # Boundary label

    # Convert the transformed_mask to a tensor
    return torch.from_numpy(transformed_mask)

# We instantiate the training and validation datasets with the new label transform
train_dataset = NucleiDataset(
    train_images, train_masks, mask_transform=label_transform,image_transform=image_transform
)
val_dataset = NucleiDataset(
    val_images, val_masks, mask_transform=label_transform, image_transform=image_transform
)

# We visualize the new label transform and make sure it's correct
counter = 0
for im, target in train_dataset:
    if counter > 3:
        break
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].axis("off")
    ax[0].imshow(im[0], cmap="gray")
    ax[1].axis("off")
    ax[1].imshow(target)
    plt.show()
    counter += 1

# We instantiate the new unet and loss function
model = UNet(out_channels=3)
model.to(device)
loss = nn.CrossEntropyLoss()

# We train the new U-Net for 10 epochs
# (we don't use a metric here, since the target (with class labels 0, 1, 2) and prediction (one-hot encoding) have different representations
n_epochs = 10
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
metric = None
train_losses, _, val_losses, _ = run_training(model, train_loader, val_loader, loss, metric, optimizer, n_epochs)
plot('Loss vs. Epoch', 'Loss', train_losses, val_losses)

# We implement the watershed approach described above.
def instance_segmentation(foreground_prediction, boundary_prediction, threshold=0.5):
    # Subtract the boundary prediction from the foreground prediction
    separated_nuclei = foreground_prediction - boundary_prediction
    # Threshold the separated nuclei to create a binary image
    binary_nuclei = (separated_nuclei > threshold).astype(np.int64)
    # Label connected components as markers for watershed
    markers = label(binary_nuclei)
    
    # Use the boundary prediction as a heightmap for watershed
    heightmap = boundary_prediction  
    
    # Use the thresholded foreground predictions as a mask to restrict watershed
    mask = (foreground_prediction > threshold).astype(np.int64)
    
    # Apply the watershed algorithm
    ws_labels = watershed(heightmap, markers, mask=mask)
    
    return ws_labels

# We check the prediction results and instance segmentation for a few images
# make sure your instance segmentation implementation is correct
# HINT: you need to apply a softmax to the network predictions 
counter = 0
with torch.no_grad():
    for im, mask in val_loader:
        if counter > 3:
            break
        # predict with the model and apply sigmoid to map the prediction to the range [0, 1]
        pred = model(im.to(device))
        pred = torch.softmax(pred, dim=1).cpu().numpy().squeeze()
        # get tbe nucleus instance segmentation by applying connected components to the binarized prediction
        nuclei = instance_segmentation(pred[1], pred[2], threshold=0.5)
        fig, ax = plt.subplots(1, 2, figsize=(16, 16))
        ax[0].axis("off")
        ax[0].imshow(im.squeeze().squeeze(), cmap="gray")
       
        ax[1].axis("off")
        ax[1].imshow(nuclei, cmap=get_random_colors(nuclei), interpolation="nearest")
        
    
        plt.show()
        counter += 1


# We use the validation set to find a good value for the 'threshold' parameter in the instance_segmentation function
# Define a range of possible thresholds to test
thresholds = np.linspace(0.1, 0.9, 9)

# Initialize a dictionary to hold the average F1 scores for each threshold
threshold_scores = {}

with torch.no_grad():
    for threshold in thresholds:
        f1_scores = []
        for im, mask in val_loader:
        # predict with the model and apply sigmoid to map the prediction to the range [0, 1]
        # Iterate over all threshold values
            pred = model(im.to(device))
            pred = torch.softmax(pred, dim=1).cpu().numpy().squeeze()
            # get tbe nucleus instance segmentation by applying connected components to the binarized prediction
            nuclei = instance_segmentation(pred[1], pred[2], threshold)
            mask = mask.numpy().squeeze()
            f1_sc = f1_score(nuclei, mask, threshold)
            f1_scores.append(f1_sc)
    
        # Compute the average F1 score across the validation set for this threshold
        threshold_scores[threshold] = np.mean(f1_scores)
        
# Find the threshold with the best average F1 score
best_threshold = max(threshold_scores, key=threshold_scores.get)
# print(threshold_scores)
print(f"Best threshold: {best_threshold} with average F1 score: {threshold_scores[best_threshold]}")


# We compute the average f1 score of all the test images
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

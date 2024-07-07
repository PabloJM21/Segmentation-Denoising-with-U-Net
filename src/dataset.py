# now we load the images and masks into memory, and normalize the images so that they have zero mean and unit variance
images = [imageio.imread(img) for img in image_paths]
masks = [imageio.imread(mask_path) for mask_path in mask_paths]

ims_flat = np.concatenate([im.ravel() for im in images])
mean, std = np.mean(ims_flat), np.std(ims_flat)
images = [(im.astype("float32") - mean) / std for im in images]

print(f"Training Images: {len(train_images)}, Training Masks: {len(train_masks)}, Validation Images: {len(val_images)}, Validation Masks: {len(val_masks)}")

# finally, let's choose the appropriate torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#As a next step, we implement a torch.utils.data.Dataset to access the data for training

class NucleiDataset(Dataset):
    def __init__(self, images, masks, image_transform=None, mask_transform=None, transform=None):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.transform = transform

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]

        # crop the images to have the shape 256 x 256, so that we can feed them into memory
        # despite them having different sizes
        crop_shape = (256, 256)
        shape = image.shape
        if shape != crop_shape:
            assert image.ndim == mask.ndim == 2
            crop_start = [np.random.randint(0, sh - csh) if sh != csh else 0 for sh, csh in zip(shape, crop_shape)]
            crop = tuple(slice(cs, cs + csh) for cs, csh in zip(crop_start, crop_shape))
            image, mask = image[crop], mask[crop]
              
                
      # make sure we have numpy arrays and add a channel dimension for the image data
#         image, mask = np.array(image), np.array(mask)
#         if image.ndim == 2:
#             image = image[None]
        
        # apply the transforms if given
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return image, mask
        
    def __len__(self):
        return len(self.images)



# we implement a transform that outputs the binary target 
# and instantiate the training dataset and validation dataset with it
# HINT: the transform can be a function that takes the mask array as input and returns the binarized version
# HINT: you will also need to add a channel dimension to the target
# HINT: the targets need to be "float32" values (for the BCE Loss coming ahead)
def mask_transform(mask):
    bin_img = (mask>0).astype(np.float32)[None]
    return torch.from_numpy(bin_img)
# Image transform
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert numpy array to torch.Tensor
])

train_dataset = NucleiDataset(train_images, train_masks, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = NucleiDataset(val_images, val_masks, image_transform=image_transform, mask_transform=mask_transform)


# we sample a few images from the dataset and verify that they are correct
import random
rand_idx = random.randint(0, len(train_dataset.images))
img, mask = train_dataset[rand_idx]

fig, ax = plt.subplots(1, 2)
ax[0].axis("off")
ax[0].imshow(np.squeeze(img), cmap='gray')
# visualize the masks with random colors
ax[1].axis("off")
ax[1].imshow(np.squeeze(mask))
plt.show()



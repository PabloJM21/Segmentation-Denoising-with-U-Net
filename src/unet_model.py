# We implement the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        super(UNet, self).__init__()

        # Encoder blocks
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        ])

        # Bottleneck block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU()
        )

        # Decoder blocks
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.ReLU(),
                          nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(),
                          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
                          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
                          nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        ])

        # Final 1x1 conv to get the segmentation map
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, dec_block in enumerate(self.dec_blocks):
            x = dec_block[0](x)  # Apply the transposed convolution
            skip_connection = skip_connections[-(i+1)]
            x = torch.cat((x, skip_connection), dim=1)  # Skip connection
            x = dec_block[1:](x)  # Apply the rest of the sequential block
        
        # Final segmentation map
        x = self.final_conv(x)
        return x


!pip install torchsummary
from torchsummary import summary

# We instantiate a unet and check that our architecture is correct by applying it to
# an input from the train loader
model = UNet()
model.to(device)
summary(model, (1,256,256))

# We display the prediction
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
ex_img, ex_mask = next(iter(train_loader))

model.eval()
with torch.no_grad():
    ex_prediction = model(ex_img.to(device))
    
# Squeeze to remove the batch dimension and channel dimension for visualization
ex_predicted_mask = ex_prediction.squeeze(0).squeeze(0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(ex_predicted_mask.cpu(), cmap='gray')
plt.title("Predicted Mask")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ex_mask.squeeze(0).squeeze(0).cpu(), cmap='gray')
plt.title("Original Mask")
plt.axis('off');

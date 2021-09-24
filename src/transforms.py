from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((100, 100)), # Hint: this might not be the best way to resize images
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Hint: this might not be the best normalization
     ]
)

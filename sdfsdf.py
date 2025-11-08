train_transform = A.Compose([
    A.Resize(224, 224),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.07),
        hole_width_range=(0.03, 0.07),
        fill=(0.54398818*255, 0.52586353*255, 0.52226893*255),
        p=0.3
    ),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.54398818, 0.52586353, 0.52226893], std=[0.328297, 0.31159157, 0.30748192]),
    ToTensorV2()
], seed=seed)

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.54398818, 0.52586353, 0.52226893], std=[0.328297, 0.31159157, 0.30748192]),
    ToTensorV2()
], seed=seed)
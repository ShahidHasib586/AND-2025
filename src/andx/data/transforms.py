from torchvision import transforms

def build_transforms(img_size, mean, std, aug):
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(aug.get("scale_min",0.2), 1.0)),
        transforms.ColorJitter(*aug.get("color_jitter",[0.4,0.4,0.4,0.4])),
        transforms.RandomGrayscale(p=aug.get("random_gray",0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    t_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return t_train, t_test

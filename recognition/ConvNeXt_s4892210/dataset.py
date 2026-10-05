"""

"""

from torchvision import transforms

# Dataset configuration and normalisation settings for the ADNI images.
ADNI_MEAN = 0.115540   # Normalisation statistics computed from training data in util.py
ADNI_STD = 0.225449
IMAGE_SIZE = 224


def build_transform(is_train):
    """

    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandAugment(5, 5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_MEAN] * 3,
                std=[ADNI_STD] * 3
            ),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[ADNI_MEAN] * 3,
                std=[ADNI_STD] * 3
            )
        ])
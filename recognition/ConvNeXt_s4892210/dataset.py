"""

"""

import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Dataset configuration and normalisation settings for the ADNI images.
DEFAULT_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
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
            transforms.RandomRotation(15),
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


def build_dataloader(
    is_train,
    root_dir = DEFAULT_ROOT,
    batch_size = 128,
    num_workers = 2,
    validation_ratio = 0.1
):
    """

    """
    if is_train:
        data_dir = os.path.join(root_dir, "train")
    else:
        data_dir = os.path.join(root_dir, "test")

    base_dataset = datasets.ImageFolder(root=data_dir)

    if is_train:
        all_paths = []
        patient_ids = []

        for sample in base_dataset.samples:
            path = sample[0]
            all_paths.append(path)

        for path in all_paths:
            patient_id = os.path.basename(path).split("_")[0]
            patient_ids.append(patient_id)

        id_to_indices = {}
        for index, pid in enumerate(patient_ids):
            id_to_indices.setdefault(pid, []).append(index)

        unique_patients = list(id_to_indices.keys())

        train_patients, val_patients = train_test_split(
            unique_patients,
            test_size=validation_ratio,
            shuffle=True,
            random_state=42
        )

        train_indices = []
        validation_indices = []

        for pid in train_patients:
            for i in id_to_indices[pid]:
                train_indices.append(i)

        for pid in val_patients:
            for i in id_to_indices[pid]:
                validation_indices.append(i)

        train_dataset = Subset(
            type(base_dataset)(
                root=base_dataset.root,
                transform=build_transform(is_train=True)
            ),
            train_indices
        )
        val_dataset = Subset(
            type(base_dataset)(
                root=base_dataset.root,
                transform=build_transform(is_train=False)
            ),
            validation_indices
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader

    else:
        transform = build_transform(is_train=False)
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return test_loader


def get_class_distribution(dataset):
    """

    """

    image_counts = {"AD": 0, "NC": 0}
    patients_by_class = {"AD": set(), "NC": set()}

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(base_dataset.samples))

    for index in indices:
        image_path, label = base_dataset.samples[index]
        class_name = base_dataset.classes[label]
        patient_id = os.path.basename(image_path).split("_")[0]
        image_counts[class_name] += 1
        patients_by_class[class_name].add(patient_id)

    return image_counts, patients_by_class


def print_dataset_summary(train_loader, val_loader, test_loader):
    """

    """

    train_images, train_patients = get_class_distribution(train_loader.dataset)
    val_images, val_patients = get_class_distribution(val_loader.dataset)
    test_images, test_patients = get_class_distribution(test_loader.dataset)

    train_base_dataset = train_loader.dataset.dataset
    print("\n=== ADNI dataset ===")
    print(f"Train subjects      : {len(train_patients['AD'] | train_patients['NC']):,}")
    print(f"Validation subjects : {len(val_patients['AD'] | val_patients['NC']):,}")
    print(f"Test subjects       : {len(test_patients['AD'] | test_patients['NC']):,}")
    print(f"Train images        : {sum(train_images.values()):,}")
    print(f"Validation images   : {sum(val_images.values()):,}")
    print(f"Test images         : {sum(test_images.values()):,}")
    print(f"Classes             : {train_base_dataset.class_to_idx}")

    print("\n=== Training Class Distribution ===")
    print(f"AD: {train_images['AD']:,} images, {len(train_patients['AD']):,} subjects")
    print(f"NC: {train_images['NC']:,} images, {len(train_patients['NC']):,} subjects")

    print("\n=== Validation Class Distribution ===")
    print(f"AD: {val_images['AD']:,} images, {len(val_patients['AD']):,} subjects")
    print(f"NC: {val_images['NC']:,} images, {len(val_patients['NC']):,} subjects")

    print("\n=== Test Class Distribution ===")
    print(f"AD: {test_images['AD']:,} images, {len(test_patients['AD']):,} subjects")
    print(f"NC: {test_images['NC']:,} images, {len(test_patients['NC']):,} subjects")
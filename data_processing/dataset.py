import torch
import os
import random
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import re


def is_image_file(file_path):
    # Common image file extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    # Get file extension
    ext = os.path.splitext(file_path)[-1].lower()
    return ext in image_extensions


def is_valid_part_format(s):
    # Define the pattern
    pattern = r"^part([1-9]|1[0-4])$"
    # Match the string against the pattern
    match = re.match(pattern, s)
    return bool(match)


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        sample_negatives="epoch",
        limit=5000,
        neg_only_reviews=True,
    ):
        """
        root_dir: Path to dataset (folders as classes)
        transform: Image transformations (e.g., augmentation, normalization)
        sample_negatives:
            - "batch" → Selects a random negative for each sample dynamically.
            - "epoch" → Assigns a negative at the start of each epoch.
            - "fixed" → Precomputed negative samples from a CSV file.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_negatives = sample_negatives
        self.neg_only_reviews = neg_only_reviews

        self.class_to_images = defaultdict(list)  # { class: [image1, image2, ...] }
        self.samples = []  # [(anchor_path, positive_path, class)]

        # Read dataset structure
        for part_folder in os.listdir(root_dir):
            if self.__len__() >= limit:
                break
            part_path = os.path.join(root_dir, part_folder)
            if not os.path.isdir(part_path) or not is_valid_part_format(part_folder):
                continue
            for product_folder in os.listdir(part_path):
                product_path = os.path.join(part_path, product_folder)
                if os.path.isdir(product_path):
                    product_and_review = [
                        os.path.join(product_path, img)
                        for img in os.listdir(product_path)
                    ]
                    if (
                        len(product_and_review) >= 2
                    ):  # Ensure at least an anchor-positive pair
                        for i in product_and_review:
                            positive = None
                            anchor = None
                            if os.path.isdir(i):  # review
                                reviews = [
                                    os.path.join(i, review_img)
                                    for review_img in os.listdir(i)
                                ]
                                if len(reviews) == 0:
                                    continue
                                # Get only first review image if it have multiple reivews
                                positive = reviews[0]
                            elif is_image_file(i) and anchor is not None:
                                anchor = i

                            if positive is not None and anchor is not None:
                                self.class_to_images[product_folder] = [
                                    anchor,
                                    positive,
                                ]
                                self.samples.append(
                                    (anchor, positive, product_folder)
                                )  # Anchor & positive

        # Precompute negatives if needed
        if self.sample_negatives == "epoch":
            self.negative_map = self.assign_negatives()

    def assign_negatives(self):
        """Assigns a random negative from a different class at the start of each epoch."""
        negative_map = {}
        product_list = list(self.class_to_images.keys())

        for product_label in self.class_to_images:
            neg_reviews = [cls for cls in product_list if cls != product_label]
            neg_review = random.choice(neg_reviews)
            if not self.neg_only_reviews:
                negative_map[product_label] = random.choice(
                    self.class_to_images[neg_review]
                )
            else:
                negative_map[product_label] = self.class_to_images[neg_review][1]

        return negative_map

    def __getitem__(self, index):
        anchor_path, positive_path, product_label = self.samples[index]

        # Choose negative based on sampling strategy
        if self.sample_negatives == "batch":
            neg_reviews = [
                cls for cls in self.class_to_images.keys() if cls != product_label
            ]
            neg_review = random.choice(neg_reviews)
            negative_path = random.choice(self.class_to_images[neg_review])
        elif self.sample_negatives == "epoch":
            negative_path = self.negative_map[product_label]
        else:
            raise ValueError("Unsupported sampling strategy. Use 'batch' or 'epoch'.")

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.samples)

    def update_negatives(self):
        """Call this at the start of each epoch if using 'epoch' sampling."""
        if self.sample_negatives == "epoch":
            self.negative_map = self.assign_negatives()

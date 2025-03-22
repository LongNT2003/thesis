import os

dir = r"D:\20241\thesis\data"


def is_image_file(file_path):
    # Common image file extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    # Get file extension
    ext = os.path.splitext(file_path)[-1].lower()
    return ext in image_extensions


# List all files and directories in the current directory
entries = os.listdir(dir)
print("Current directory contents:")
print(entries)
for i in entries:
    print(i, is_image_file(os.path.join(dir, i)))

import re


def is_valid_format(s):
    # Define the pattern
    pattern = r"^part([1-9]|1[0-4])$"
    # Match the string against the pattern
    match = re.match(pattern, s)
    return bool(match)


# Test cases
print(is_valid_format("part1"))  # True
print(is_valid_format("part0"))  # False
print(is_valid_format("part15"))  # False
print(is_valid_format("part14"))  # True

import os
import pandas as pd

dir_path = "data/part7"
label_file = "review_labels_part7.csv"
df = pd.read_csv(label_file)
products = os.listdir(dir_path)


def unique_elements(list1, list2):
    return list(set(list1) ^ set(list2))  # Symmetric difference


# print(unique_elements(products, df["product"].to_list()))
print(f"Length of dir: {len(os.listdir(dir_path))}")


def unique_elements(list1, list2):
    only_in_list1 = list(set(list1) - set(list2))  # Elements in list1 but not in list2
    only_in_list2 = list(set(list2) - set(list1))  # Elements in list2 but not in list1
    return only_in_list1, only_in_list2


only_a, only_b = unique_elements(products, df["product"].to_list())
print("Only in list1:", only_a)  # Output: [1, 2]
print("Only in list2:", only_b)  # Output: [6, 7]

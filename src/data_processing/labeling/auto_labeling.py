import os
import random
import pandas as pd


def generate_review_table(root_dir, incorrect_ratio=0.2, existing_file=None):
    data = []

    # Nếu có file dữ liệu trước đó, load vào dataframe
    if existing_file and os.path.exists(existing_file):
        df_existing = pd.read_csv(existing_file)
        data.extend(df_existing.values.tolist())

    product_folders = sorted(os.listdir(root_dir))
    random.shuffle(product_folders)  # Shuffle để randomize dữ liệu

    for product_id in product_folders:
        product_path = os.path.join(root_dir, product_id)
        if not os.path.isdir(product_path):
            continue

        # Tìm ảnh sản phẩm (ảnh nằm trong folder chính)
        product_images = [
            f for f in os.listdir(product_path) if f.endswith((".jpg", ".png"))
        ]
        if not product_images:
            continue
        product_image = product_images[0]  # Lấy 1 ảnh sản phẩm

        # Tìm folder chứa ảnh review
        review_folders = [
            f
            for f in os.listdir(product_path)
            if os.path.isdir(os.path.join(product_path, f))
        ]
        if not review_folders:
            continue
        review_folder = os.path.join(
            product_path, review_folders[0]
        )  # Lấy folder đầu tiên

        # Tìm ảnh review
        review_images = [
            f for f in os.listdir(review_folder) if f.endswith((".jpg", ".png"))
        ]
        if not review_images:
            continue
        review_image = review_images[0]  # Lấy 1 ảnh review

        # Xác định nhãn "Correct" hoặc "Incorrect"
        label = "Correct"
        data.append([product_id, os.path.join(review_folder, review_image), label])

    # Chèn incorrect ở giữa danh sách
    num_incorrect = int(len(data) * incorrect_ratio)
    incorrect_indices = list(range(len(data) // 3, 2 * len(data) // 3))
    incorrect_indices = random.sample(
        incorrect_indices, min(num_incorrect, len(incorrect_indices))
    )

    for i in incorrect_indices:
        data[i][2] = "Incorrect"

    # Lưu vào DataFrame
    df_result = pd.DataFrame(data, columns=["product", "review", "label_correct"])
    df_result = df_result.drop_duplicates(subset=["product", "review"])

    return df_result


# Sử dụng hàm
root_directory = "data/part7"  # Đường dẫn đến thư mục chứa dữ liệu
output_file = "review_labels_part7.csv"
df_result = generate_review_table(
    root_directory, incorrect_ratio=0.0, existing_file=output_file
)
df_result.to_csv(output_file, index=False)

print(f"Saved result to {output_file}")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def test_image_embedding_model(
    image1_path, image2_path, model, threshold, metric="l2", show_images=True
):
    """
    Kiểm tra mức độ tương đồng của hai ảnh dựa trên model embedding.

    Args:
        image1_path (str): Đường dẫn ảnh thứ nhất.
        image2_path (str): Đường dẫn ảnh thứ hai.
        model (torch.nn.Module): Mô hình trích xuất embedding.
        threshold (float): Ngưỡng phân biệt giữa ảnh tương đồng và không tương đồng.
        metric (str): Phương pháp đo lường, chọn 'cosine' hoặc 'l2'.
        show_images (bool): Nếu True, hiển thị hai ảnh để so sánh trực quan.

    Returns:
        int: 1 nếu hai ảnh tương đồng, 0 nếu không tương đồng.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tiền xử lý ảnh
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Thêm batch dimension

    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)
    img1, img2 = img1.to(device), img2.to(device)
    # Chuyển ảnh sang tensor và đưa vào model để lấy embedding
    model.eval()
    with torch.no_grad():
        emb1 = model(img1).squeeze(0)  # (N, D) → (D,)
        emb2 = model(img2).squeeze(0)  # (N, D) → (D,)

    # Tính toán khoảng cách hoặc độ tương đồng
    if metric == "cosine":
        similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
        score_text = f"Similarity: {similarity:.4f}"
        result = 1 if similarity >= threshold else 0
    elif metric == "l2":
        distance = torch.norm(emb1 - emb2, p=2).item()
        score_text = f"Distance: {distance:.4f}"
        result = 1 if distance <= threshold else 0
    else:
        raise ValueError("metric phải là 'cosine' hoặc 'l2'")

    # Hiển thị ảnh nếu được yêu cầu
    if show_images:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(Image.open(image1_path))
        axes[0].set_title("Image 1")
        axes[0].axis("off")

        axes[1].imshow(Image.open(image2_path))
        axes[1].set_title("Image 2")
        axes[1].axis("off")

        plt.suptitle(score_text)
        plt.show()

    print(score_text)

    return result


def evaluate_batch(anchor, pos, neg, threshold=0.5, metric="l2"):
    """
    Đánh giá theo batch với metric là cosine similarity hoặc L2 distance.

    Args:
        anchor (torch.Tensor): Batch embedding của anchor, shape (batch_size, embedding_dim)
        pos (torch.Tensor): Batch embedding của positive, shape (batch_size, embedding_dim)
        neg (torch.Tensor): Batch embedding của negative, shape (batch_size, embedding_dim)
        threshold (float): Ngưỡng quyết định mẫu có giống nhau không.
        metric (str): 'cosine' hoặc 'l2' để chọn phương pháp đo khoảng cách.

    Returns:
        tuple: (TP, TN, FP, FN)
    """

    if metric == "cosine":
        # Tính cosine similarity
        sim_pos = F.cosine_similarity(
            anchor, pos, dim=-1
        )  # Cosine similarity giữa anchor và positive
        sim_neg = F.cosine_similarity(
            anchor, neg, dim=-1
        )  # Cosine similarity giữa anchor và negative

        # Xác định TP, FP, TN, FN
        tp = (sim_pos >= threshold).sum().item()  # Dự đoán đúng positive
        fn = (
            (sim_pos < threshold).sum().item()
        )  # Dự đoán sai positive (đáng lẽ giống nhưng bị xem là khác)
        tn = (sim_neg < threshold).sum().item()  # Dự đoán đúng negative
        fp = (
            (sim_neg >= threshold).sum().item()
        )  # Dự đoán sai negative (đáng lẽ khác nhưng bị xem là giống)

    elif metric == "l2":
        # Tính L2 distance
        dist_pos = torch.norm(
            anchor - pos, p=2, dim=-1
        )  # Khoảng cách L2 giữa anchor và positive
        dist_neg = torch.norm(
            anchor - neg, p=2, dim=-1
        )  # Khoảng cách L2 giữa anchor và negative

        # Xác định TP, FP, TN, FN
        tp = (dist_pos <= threshold).sum().item()  # Dự đoán đúng positive
        fn = (dist_pos > threshold).sum().item()  # Dự đoán sai positive
        tn = (dist_neg > threshold).sum().item()  # Dự đoán đúng negative
        fp = (dist_neg <= threshold).sum().item()  # Dự đoán sai negative

    else:
        raise ValueError("Metric must be 'cosine' or 'l2'")

    return tp, tn, fp, fn


def evaluate_metrics(tp, tn, fp, fn):
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # F1-Score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    # False Negative Rate (FNR)
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "False Positive Rate (FPR)": fpr,
        "False Negative Rate (FNR)": fnr,
    }

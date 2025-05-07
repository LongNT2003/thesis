import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Load ảnh gốc
image = Image.open(
    r"data\test_label\P6000077221\38532930251205576_1612400445.png"
).convert(
    "RGB"
)  # Dùng PIL để mở ảnh

# Định nghĩa phép biến đổi
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Chuyển ảnh thành tensor (giá trị trong khoảng [0,1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Áp dụng phép biến đổi
normalized_image = transform(image)

# Chuyển đổi tensor về ảnh để hiển thị (giải chuẩn hóa)
unnormalized_image = normalized_image.clone()
unnormalized_image = unnormalized_image * torch.tensor([0.229, 0.224, 0.225]).view(
    3, 1, 1
) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
unnormalized_image = torch.clamp(
    unnormalized_image, 0, 1
)  # Giới hạn giá trị pixel trong khoảng [0,1]

# Hiển thị ảnh trước & sau khi chuẩn hóa
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Ảnh Gốc")
ax[1].imshow(unnormalized_image.permute(1, 2, 0))  # Đưa tensor về dạng HWC để hiển thị
ax[1].set_title("Ảnh Sau Khi Chuẩn Hóa")
plt.show()

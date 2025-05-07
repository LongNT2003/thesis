from loss import NTXentLoss
from head import EmbeddingHead
from train import SpamDetectorTrainer
from dataset import TripletDataset, EvalTripletDataset
import torchvision.transforms as transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch
import wandb

wandb.login(key="c436a4917c43e09e30b67b919bd06e7bf7b0c10d")
wandb.init(project="review thesis project", name="experiment_1", resume="allow")

data_path = "data"
# Hyperparameters
base_lr = 1e-3  # Learning rate ban đầu
num_epochs = 30
dataset_size = 5000
val_dataset_size = 1000
batch_size = 32
warmup_ratio = 0.1  # 10% epochs đầu là warmup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = convnext_tiny(weight=ConvNeXt_Tiny_Weights.DEFAULT)
# model.requires_grad_(False)
model.classifier = EmbeddingHead(128, 768)
model = torch.nn.DataParallel(model)
model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize tất cả ảnh về 224x224
        transforms.ToTensor(),  # Chuyển ảnh thành tensor
        transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang ảnh với xác suất 50%
        transforms.ColorJitter(brightness=0.3),  # Điều chỉnh độ sáng (±30%)
        transforms.RandomPerspective(
            distortion_scale=0.5, p=0.5
        ),  # Biến dạng phối cảnh
        transforms.RandomRotation(degrees=30),  # Xoay ảnh trong khoảng ±30 độ
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = TripletDataset(data_path, transform=transform, limit=dataset_size)
val_dataset = EvalTripletDataset(data_path, transform=transform, limit=val_dataset_size)


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Khởi tạo Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=base_lr)
criterion = torch.nn.TripletMarginLoss()
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=base_lr,
    epochs=num_epochs,
    steps_per_epoch=int(dataset_size / batch_size),
    pct_start=warmup_ratio,
)

# Lặp qua dataloader để lấy batch
for images, labels, negative in train_loader:
    print(
        f"Batch size: {images.shape}, Labels: {labels.shape}, Negative: {negative.shape}"
    )
    break  # Dừng sau batch đầu tiên


trainer = SpamDetectorTrainer(
    model=model,
    criterion=criterion,
    train_loader=train_loader,
    valid_loader=val_loader,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    device=device,
    epochs=num_epochs,
    log_writer=wandb,
)

trainer.train()

wandb.finish()

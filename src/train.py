import torch
import wandb
from typing import Literal
from eval import evaluate_batch, evaluate_metrics


class SpamDetectorTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        lr_types: Literal["step", "epoch"] = "step",
        device: torch.device = "cuda",
        epochs: int = 10,
        max_norm: float = 0,
        log_writer: wandb = None,
        patience=3,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_types = lr_types
        self.device = device
        self.epochs = epochs
        self.max_norm = max_norm
        self.log_writer = log_writer
        self.patience = patience

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        global_step = epoch * len(self.train_loader)  # Global step tracking🔥

        for step, (anchor, positive, negative) in enumerate(self.train_loader):
            anchor, positive, negative = (
                anchor.to(self.device),
                positive.to(self.device),
                negative.to(self.device),
            )

            self.optimizer.zero_grad()
            anchor_embeded = self.model(anchor)
            positive_embeded = self.model(positive)
            negative_embeded = self.model(negative)
            loss = self.criterion(anchor_embeded, positive_embeded, negative_embeded)

            loss.backward()

            # Tính toán Gradient Norm
            total_norm = 0
            param_count = 0

            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm().item()
                    total_norm += param_norm
                    param_count += 1

            mean_grad_norm = total_norm / param_count if param_count > 0 else 0

            # Áp dụng Gradient Clipping nếu cần
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()

            # Step scheduler per step
            if self.scheduler and self.lr_types == "step":
                self.scheduler.step()

            # Log to W&B
            if self.log_writer:
                self.log_writer.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm": mean_grad_norm,
                        "train/global_step": global_step + step,
                        "train/epoch": epoch + step / len(self.train_loader),
                    }
                )

            total_loss += loss.item()

        epoch_loss = total_loss / len(self.train_loader)
        if self.log_writer:
            self.log_writer.log({"train/mean_loss": epoch_loss})

        # Step scheduler per epoch
        if self.scheduler and self.lr_types == "epoch":
            self.scheduler.step()  # Step based on epoch

        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for step, (anchor, positive, negative) in enumerate(self.valid_loader):
                anchor, positive, negative = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    negative.to(self.device),
                )
                anchor_embeded = self.model(anchor)
                positive_embeded = self.model(positive)
                negative_embeded = self.model(negative)
                loss = self.criterion(
                    anchor_embeded, positive_embeded, negative_embeded
                )
                running_loss += loss.item()

                eval_step_result = evaluate_batch(
                    anchor_embeded, positive_embeded, negative_embeded
                )
                tp += eval_step_result[0]
                tn += eval_step_result[1]
                fp += eval_step_result[2]
                fn += eval_step_result[3]

        metrics = evaluate_metrics(tp, tn, fp, fn)

        # Calculate average loss
        avg_loss = running_loss / len(self.valid_loader)

        # Log metrics and loss to W&B
        if self.log_writer:
            # Log evaluation metrics
            self.log_writer.log(
                {
                    "val/accuracy": metrics["Accuracy"],
                    "val/precision": metrics["Precision"],
                    "val/recall": metrics["Recall"],
                    "val/f1_score": metrics["F1-Score"],
                    "val/False Positive Rate": metrics["False Positive Rate (FPR)"],
                    "val/False Negative Rate": metrics["False Negative Rate (FNR)"],
                    "val/val_loss": avg_loss,
                }
            )
        return avg_loss

    def train(self, resume_from_checkpoint=None):
        start_epoch = 0
        best_val_loss = float("inf")
        epochs_without_improvement = 0  # Track epochs without improvement

        # Load checkpoint if provided
        if resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Start training!")
        for epoch in range(start_epoch, start_epoch + self.epochs):
            epoch_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()

            print(
                f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset counter
                print(
                    f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint."
                )

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": (
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    "best_val_loss": best_val_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                }
                torch.save(checkpoint, f"checkpoint_{epoch}.pth")
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement}/{self.patience} epochs."
                )

            # Stop training if no improvement for `self.patience` epochs
            if epochs_without_improvement >= self.patience:
                print(
                    f"Validation loss hasn't improved for {self.patience} epochs. Stopping training early."
                )
                break

        print("Training complete.")

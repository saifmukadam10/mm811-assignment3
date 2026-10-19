import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder_model import AutoEncoder
from train_autoencoder import train_epoch, evaluate


def get_dataloaders(batch_size: int, num_workers: int = 2):
    tfm = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = train_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Train without binarization for first N epochs",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1e-3,
        help="Weight for latent push-away regularizer",
    )
    parser.add_argument(
        "--use_l1", action="store_true", help="Use L1 loss (default off if flag absent)"
    )
    parser.add_argument(
        "--no_l1", action="store_true", help="Force MSE even if --use_l1 set elsewhere"
    )
    args = parser.parse_args()

    # Resolve L1 flag
    use_l1 = True
    if args.no_l1:
        use_l1 = False
    elif args.use_l1:
        use_l1 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(args.batch_size)

    model = AutoEncoder(use_binary=False)  # start with warm-up by default
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        # Toggle binarization after warmup period
        if epoch == args.warmup_epochs + 1:
            model.use_binary = True

        train_loss, train_psnr = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            use_l1=use_l1,
            reg_lambda=args.reg_lambda,
        )
        val_psnr = evaluate(model, val_loader, device)
        scheduler.step(val_psnr)

        print(
            f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | TrainPSNR: {train_psnr:.2f} | ValPSNR: {val_psnr:.2f}"
        )

        # save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "autoencoder_best.pth")
            print(f" -> Saved new best (Val PSNR: {best_psnr:.2f})")

        # Early exit if goal reached
        if val_psnr > 25.0:
            print(f"\nGoal achieved! PSNR ≥ 25.0 (Current: {val_psnr:.2f}) ***")
            break

    print(f"\nTraining complete. Best Validation PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()

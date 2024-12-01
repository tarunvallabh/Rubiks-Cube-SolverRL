# train.py
import matplotlib.pyplot as plt
from network import CubeNet
from trainer import ADITrainer


def train(epochs=1000):
    model = CubeNet()
    trainer = ADITrainer(model)

    # Training history
    history = {"total_loss": [], "value_loss": [], "policy_loss": []}

    # Training loop
    for epoch in range(epochs):
        # Training step
        losses = trainer.train_step(batch_size=128, scramble_depth=20)

        # Record losses
        for key, value in losses.items():
            history[key].append(value)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"Total Loss: {losses['total_loss']:.4f}")
            print(f"Value Loss: {losses['value_loss']:.4f}")
            print(f"Policy Loss: {losses['policy_loss']:.4f}\n")

            # Save checkpoint
            trainer.save_checkpoint(f"checkpoint_{epoch}.pt")

    # Plot training history
    plt.figure(figsize=(10, 6))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, trainer


if __name__ == "__main__":
    model, trainer = train()

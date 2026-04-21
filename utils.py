import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Loads and splits dataset from a flat class-folder directory.
    Applies 80/20 train/validation split automatically.
    """
    img_shape = (image_size, image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_shape,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_shape,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def get_latest_checkpoint(checkpoint_dir: str):
    pattern = os.path.join(checkpoint_dir, "model_epoch_*.keras")
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def get_initial_epoch(checkpoint_path) -> int:
    if checkpoint_path is None:
        return 0
    basename = os.path.basename(checkpoint_path)
    epoch_str = basename.replace("model_epoch_", "").replace(".keras", "")
    try:
        return int(epoch_str)
    except ValueError:
        return 0


def plot_training_history(history, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history["accuracy"],     label="Train Accuracy",      color="#2ecc71", linewidth=2)
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy", color="#e74c3c", linewidth=2, linestyle="--")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history["loss"],     label="Train Loss",      color="#3498db", linewidth=2)
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss", color="#e67e22", linewidth=2, linestyle="--")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=150)
    plt.close()

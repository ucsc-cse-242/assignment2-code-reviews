"""Code answers for CSE 242 Assignment 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "Advertising.csv"


def fit_advertising_models(data):
    """Fit the requested three-feature and two-feature regression models."""
    y = data["Sales"]
    X = data[["TV", "Radio", "Newspaper"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    X_two = data[["TV", "Radio"]]
    X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(
        X_two, y, test_size=0.2, random_state=42
    )
    model_two = LinearRegression()
    model_two.fit(X_train_two, y_train_two)
    y_pred_two = model_two.predict(X_test_two)
    rmse_two = mean_squared_error(y_test_two, y_pred_two) ** 0.5

    r2 = model.score(X_test, y_test)
    return model, rmse, r2, model_two, rmse_two


def display_digits(images, labels, count=15):
    """Show up to 15 digit images with their matching labels."""
    count = min(count, len(images))
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < count:
            image = images[idx]
            if image.size == 64:
                image = image.reshape(8, 8)
            ax.imshow(image, cmap="gray")
            ax.set_title(f"label {labels[idx]}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_digit_pca(digits, projected, variance_ratios, components):
    """Plot the 2D PCA projection, explained variance, and eigen-digits."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    scatter = axes[0].scatter(
        projected[:, 0], projected[:, 1], c=digits.target, cmap="Paired", s=12
    )
    axes[0].set(xlabel="First component", ylabel="Second component",
                title="Digits projected onto two components")
    fig.colorbar(scatter, ax=axes[0], label="Digit label")

    axes[1].plot(np.arange(1, len(variance_ratios) + 1), variance_ratios,
                 marker=".", markersize=3)
    axes[1].set(xlabel="Component", ylabel="Explained variance ratio",
                title="Explained variance by component")

    axes[2].imshow(components[0].reshape(8, 8), cmap="gray")
    axes[2].set(title="First principal component", xticks=[], yticks=[])
    return fig


def plot_eigen_digits(components):
    """Show all 64 PCA eigenvectors as 8 by 8 pixel patterns."""
    fig, axes = plt.subplots(6, 12, figsize=(12, 7))
    for idx, ax in enumerate(axes.flat):
        if idx < len(components):
            ax.imshow(components[idx].reshape(8, 8), cmap="gray")
            ax.set_title(f"{idx + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Eigenvectors of the digit data")
    fig.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH, index_col=0)
    model, rmse, r2, model_two, rmse_two = fit_advertising_models(data)

    digits = load_digits()
    pca_2 = PCA(n_components=2)
    projected = pca_2.fit_transform(digits.data)
    pca_full = PCA().fit(digits.data)
    variance_ratios = pca_full.explained_variance_ratio_
    print("Digit data shape:", digits.data.shape)
    print("Two-component projection shape:", projected.shape)
    print("Explained variance ratios:", variance_ratios)

    # Reconstruct clean and noisy digits while retaining 80% of the variance.
    pca_08 = PCA(n_components=0.8)
    clean_reconstruction = pca_08.inverse_transform(
        pca_08.fit_transform(digits.data)
    )
    rng = np.random.default_rng(0)
    noisy_images = rng.normal(digits.data, 5)
    pca_noise = PCA(n_components=0.8)
    noisy_reconstruction = pca_noise.inverse_transform(
        pca_noise.fit_transform(noisy_images)
    )

    summary = [
        "Advertising data",
        f"Rows and columns {data.shape}",
        f"Training rows 160 and test rows 40 with random_state 42",
        f"Intercept {model.intercept_:.4f}",
        "Coefficients for TV, Radio, Newspaper "
        + ", ".join(f"{value:.4f}" for value in model.coef_),
        f"Test R2 with three features {r2:.4f}",
        f"Test RMSE with TV, Radio, Newspaper {rmse:.4f}",
        f"Test RMSE with TV and Radio {rmse_two:.4f}",
        "",
        "Digits PCA",
        f"Input shape {digits.data.shape}",
        f"Two component output shape {projected.shape}",
        f"Components needed to retain 80 percent variance {pca_08.n_components_}",
        "First ten explained variance ratios "
        + ", ".join(f"{value:.4f}" for value in variance_ratios[:10]),
        f"Components needed for noisy images at 80 percent variance {pca_noise.n_components_}",
        "Noise generation seed 0",
    ]
    (output_dir / "results.txt").write_text("\n".join(summary) + "\n")
    for filename, images in [
        ("sample_digits.png", digits.images),
        ("clean_reconstruction.png", clean_reconstruction),
        ("noisy_digits.png", noisy_images),
        ("noisy_reconstruction.png", noisy_reconstruction),
    ]:
        fig = display_digits(images, digits.target)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig = plot_digit_pca(
        digits, projected, variance_ratios, pca_full.components_
    )
    fig.savefig(output_dir / "pca_summary.png", dpi=160)
    plt.close(fig)
    fig = plot_eigen_digits(pca_full.components_)
    fig.savefig(output_dir / "eigen_digits.png", dpi=160)
    plt.close(fig)

    print("Results written to", output_dir)
    print("\n".join(summary))


if __name__ == "__main__":
    main()

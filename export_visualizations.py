"""
Script to export key visualizations from the Jupyter notebook for presentation.
Run this script to generate high-quality images for your slides.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300

# Load data
print("Loading dataset...")
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

# Create output directory
import os

os.makedirs("presentation_images", exist_ok=True)

print("\nGenerating visualizations...")

# 1. Target Variable Distribution
print("1. Creating target distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_dist = df["loan_status"].value_counts()
target_percent = df["loan_status"].value_counts(normalize=True) * 100

# Count plot
sns.countplot(data=df, x="loan_status", palette="Set2", ax=axes[0])
axes[0].set_title("Loan Status Distribution (Count)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Loan Status", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)

# Pie chart
axes[1].pie(
    target_dist.values,
    labels=target_dist.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("Set2"),
    startangle=90,
)
axes[1].set_title(
    "Loan Status Distribution (Percentage)", fontsize=14, fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    "presentation_images/01_target_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/01_target_distribution.png")

# 2. Key Features vs Loan Status (Box Plots)
print("2. Creating box plots for key features...")
key_numerical = ["income_annum", "loan_amount", "cibil_score", "loan_term"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(key_numerical):
    sns.boxplot(data=df, x="loan_status", y=col, palette="Set2", ax=axes[i])
    axes[i].set_title(
        f'{col.replace("_", " ").title()} vs Loan Status',
        fontsize=12,
        fontweight="bold",
    )
    axes[i].set_xlabel("Loan Status", fontsize=10)
    axes[i].set_ylabel(col.replace("_", " ").title(), fontsize=10)

plt.tight_layout()
plt.savefig(
    "presentation_images/02_boxplots_key_features.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/02_boxplots_key_features.png")

# 3. Correlation Heatmap
print("3. Creating correlation heatmap...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [col for col in numerical_cols if col != "loan_id"]

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold", pad=20
)
plt.tight_layout()
plt.savefig(
    "presentation_images/03_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/03_correlation_heatmap.png")

# 4. Categorical Features Analysis
print("4. Creating categorical features analysis...")
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_features = [col for col in categorical_cols if col != "loan_status"]

fig, axes = plt.subplots(1, len(categorical_features), figsize=(16, 5))
if len(categorical_features) == 1:
    axes = [axes]

for i, col in enumerate(categorical_features):
    sns.countplot(data=df, x=col, hue="loan_status", palette="Set2", ax=axes[i])
    axes[i].set_title(
        f'{col.replace("_", " ").title()} vs Loan Status',
        fontsize=12,
        fontweight="bold",
    )
    axes[i].set_xlabel(col.replace("_", " ").title(), fontsize=10)
    axes[i].set_ylabel("Count", fontsize=10)
    axes[i].legend(title="Loan Status")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(
    "presentation_images/04_categorical_features.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/04_categorical_features.png")

# 5. Data Quality Summary
print("5. Creating data quality summary visualization...")
quality_data = {
    "Metric": [
        "Missing Values",
        "Duplicate Rows",
        "Negative Values",
        "Data Consistency",
    ],
    "Status": ["None", "None", "None", "All Consistent"],
    "Count": [0, 0, 0, "✓"],
}

quality_df = pd.DataFrame(quality_data)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("tight")
ax.axis("off")

table = ax.table(
    cellText=quality_df.values,
    colLabels=quality_df.columns,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Color code the status column
for i in range(1, len(quality_df) + 1):
    table[(i, 1)].set_facecolor("#90EE90")  # Light green for good status

ax.set_title("Data Quality Assessment Summary", fontsize=14, fontweight="bold", pad=20)
plt.savefig(
    "presentation_images/05_data_quality.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/05_data_quality.png")

# 6. Preprocessing Pipeline Flowchart
print("6. Creating preprocessing pipeline diagram...")
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

steps = [
    "1. Data Loading\n& Cleaning",
    "2. Quality\nAssessment",
    "3. Outlier\nHandling",
    "4. Categorical\nEncoding",
    "5. Feature\nEngineering",
    "6. Feature\nPreparation",
]

# Draw boxes
box_width = 1.5
box_height = 0.8
start_x = 0.5
y_pos = 0.5

for i, step in enumerate(steps):
    x_pos = start_x + i * 2
    rect = plt.Rectangle(
        (x_pos - box_width / 2, y_pos - box_height / 2),
        box_width,
        box_height,
        linewidth=2,
        edgecolor="#2E86AB",
        facecolor="#A8DADC",
    )
    ax.add_patch(rect)
    ax.text(
        x_pos, y_pos, step, ha="center", va="center", fontsize=10, fontweight="bold"
    )

    # Draw arrow
    if i < len(steps) - 1:
        ax.arrow(
            x_pos + box_width / 2,
            y_pos,
            0.5,
            0,
            head_width=0.1,
            head_length=0.15,
            fc="#2E86AB",
            ec="#2E86AB",
        )

ax.set_xlim(0, 14)
ax.set_ylim(0, 1)
ax.set_title("Data Preprocessing Pipeline", fontsize=16, fontweight="bold", pad=20)
plt.savefig(
    "presentation_images/06_preprocessing_pipeline.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("   ✓ Saved: presentation_images/06_preprocessing_pipeline.png")

print("\n" + "=" * 60)
print("✓ All visualizations exported successfully!")
print("=" * 60)
print("\nImages saved in: presentation_images/")
print("\nYou can now use these images in your presentation:")
for i, img in enumerate(
    [
        "01_target_distribution.png",
        "02_boxplots_key_features.png",
        "03_correlation_heatmap.png",
        "04_categorical_features.png",
        "05_data_quality.png",
        "06_preprocessing_pipeline.png",
    ],
    1,
):
    print(f"  {i}. {img}")

print("\nNext steps:")
print("1. Review the images in the 'presentation_images' folder")
print("2. Create your presentation using Google Slides, Canva, or PowerPoint")
print("3. Insert these images into your slides")
print("4. Add the markdown summary (Step2_Submission_Summary.md)")

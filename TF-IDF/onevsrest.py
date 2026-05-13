import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


labels = [
    "O",
    "API_KEY",
    "CREDIT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "IBAN",
    "PASSWORD",
    "SSN",
    "FULL_NAME",
    "FIRST_NAME",
    "LAST_NAME",
    "EMAIL",
    "PHONE_NUMBER",
]


def add_box(ax, x, y, text, width=2.7, height=0.65):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )


fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# Input box
add_box(ax, 5.4, 7.0, "Combined TF-IDF feature vector", width=3.4, height=0.7)

# Arrow down
ax.annotate(
    "",
    xy=(7.1, 6.45),
    xytext=(7.1, 7.0),
    arrowprops=dict(arrowstyle="->", linewidth=1.5),
)

# One-vs-Rest title
ax.text(
    7.1,
    6.15,
    "One-vs-Rest classification",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
)

# Classifier boxes
start_x = 0.6
start_y = 4.8
x_gap = 3.3
y_gap = 1.25

for i, label in enumerate(labels):
    row = i // 4
    col = i % 4
    x = start_x + col * x_gap
    y = start_y - row * y_gap
    add_box(ax, x, y, f"Binary SVM\n{label}", width=2.7, height=0.75)

    # arrows from input area to classifier
    ax.annotate(
        "",
        xy=(x + 1.35, y + 0.75),
        xytext=(7.1, 6.0),
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
    )

# Output
add_box(ax, 4.7, 0.35, "Final multilabel prediction", width=4.8, height=0.7)

# Arrows from classifiers to output
for i in range(4):
    x = start_x + i * x_gap + 1.35
    ax.annotate(
        "",
        xy=(7.1, 1.05),
        xytext=(x, 2.3),
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
    )

ax.text(
    7.1,
    0.05,
    "Each classifier independently predicts whether one PII label is present or absent.",
    ha="center",
    va="center",
    fontsize=10,
)

plt.tight_layout()
plt.savefig("one_vs_rest_visualization.png", dpi=300, bbox_inches="tight")
plt.show()
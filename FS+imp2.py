import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# ------------------------------
# Load dataset
# ------------------------------
#GSE118553 GSE132903 GSE48350 GSE36980

dbname = "GSE48350"
ds = pd.read_excel(dbname+"_e.xlsx")
ds_t = ds.transpose()
X = ds_t.iloc[1:, :-1].values
y = ds_t.iloc[1:, -1].astype(int).values
genes = ds_t.iloc[0, :-1].values

n_features = X.shape[1]
n_folds = 10
alpha_value = 0.01  # ثابت
all_coefs = np.zeros((n_folds, n_features))

# ------------------------------
# Stratified KFold
# ------------------------------
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(i)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Lasso with fixed alpha
    lasso = Lasso(alpha=alpha_value, max_iter=30000)
    lasso.fit(X_train_scaled, y_train)

    # Store absolute coefficients
    all_coefs[i, :] = np.abs(lasso.coef_)

# ------------------------------
# Compute mean + STD across folds
# ------------------------------
coef_mean = np.mean(all_coefs, axis=0)
coef_std = np.std(all_coefs, axis=0)

# ------------------------------
# Sort genes by importance
# ------------------------------
sorted_idx = np.argsort(coef_mean)[::-1]
sorted_genes = genes[sorted_idx]
sorted_mean = coef_mean[sorted_idx]
sorted_std = coef_std[sorted_idx]

# ------------------------------
# Shorten long gene names
# ------------------------------
def shorten_name(g, max_len=12):
    g = str(g)
    return g[:max_len] + "…" if len(g) > max_len else g

short_genes = [shorten_name(g) for g in sorted_genes]

# ------------------------------
# Plot Top 20 Genes
# ------------------------------
top_n = 20
dataset_name = dbname
plt.figure(figsize=(13, 7), dpi=400)
plt.bar(range(top_n), sorted_mean[:top_n], yerr=sorted_std[:top_n],
        capsize=4, color="#87CEFA", edgecolor="black", linewidth=0.7)
plt.xticks(range(top_n), short_genes[:top_n], rotation=75, fontsize=8)
plt.title(f"Top 20 Important Genes ({dataset_name}) with LASSO", fontsize=13)
plt.xlabel("Genes", fontsize=10)
plt.ylabel("Mean Importance ± STD", fontsize=10)
plt.tight_layout()

# Add STD value above each bar
for i in range(top_n):
    plt.text(i, sorted_mean[i] + sorted_std[i] + (0.01 * max(sorted_mean)),
             f"{sorted_std[i]:.3f}", ha='center', va='bottom', fontsize=7, rotation=45)

plt.savefig(f"Top20_LASSO_{dataset_name}_With_STD.png", dpi=400)
plt.show()

# ------------------------------
# Save table of top genes
# ------------------------------
mapping_table = pd.DataFrame({
    "Original_Gene_Name": sorted_genes[:top_n],
    "Short_Name": short_genes[:top_n],
    "Mean_Importance": sorted_mean[:top_n],
    "Std_Importance": sorted_std[:top_n]
})
mapping_table.to_excel(f"Top20_Genes_Lasso_{dataset_name}_With_STD.xlsx", index=False)
print(f"Saved: Top20_Genes_Lasso_{dataset_name}_With_STD.xlsx")

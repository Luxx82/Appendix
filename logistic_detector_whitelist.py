# logistic_detector_whitelist.py
# ------------------------------------------------------------
# Detect AI-generated TEC sheets with logistic regression
# (human vs GPT-4o + Claude 3·7 only)
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# -----------------------------------------------------------------
# 1. LOAD AND FILTER  ------------------------------------------------
# -----------------------------------------------------------------
FILE  = Path("/mnt/data/Allresults4.xlsx")   # <- adjust if needed
SHEET = "AllResults"

df   = pd.read_excel(FILE, sheet_name=SHEET)
mask = pd.to_numeric(df["Unnamed: 0"], errors="coerce").notnull()
qdf  = df[mask]                                # 96 valid questions

header    = df.iloc[0]                         # row 0 = model label
temp_row  = df.iloc[1]                         # row 1 = temperature
cols      = [c for c in df.columns[2:] if c]   # answer columns

# keep only humans + whitelisted AI models
AI_WHITELIST = {"gpt-4o", "Claude 3.7 Sonnet"}
keep_cols    = [
    c for c in cols
    if header[c] == "Human" or header[c] in AI_WHITELIST
]

# -----------------------------------------------------------------
# 2. ENCODE  --------------------------------------------------------
# -----------------------------------------------------------------
mapping = {'A':0,'B':1,'C':2,'D':3,'E':4}      # blanks/Erro → -1
X = (
    qdf[keep_cols]
    .applymap(lambda x: mapping.get(str(x).strip(), -1))
    .T
    .values
)                                               # shape (992, 96)

y = np.array([0 if header[c]=="Human" else 1 for c in keep_cols])
print(f"Matrix X: {X.shape} | humans={sum(y==0)}   AI={sum(y==1)}")

# -----------------------------------------------------------------
# 3. CLASSIFIER  ----------------------------------------------------
# -----------------------------------------------------------------
clf = LogisticRegression(max_iter=1000, solver="lbfgs")
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(clf, X, y, cv=cv)
y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:,1]

# -----------------------------------------------------------------
# 4. METRICS  -------------------------------------------------------
# -----------------------------------------------------------------
cm             = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
auc_val        = roc_auc_score(y, y_prob)

print("\nConfusion matrix (rows = actual, cols = predicted):\n", cm)
print(f"\nAccuracy      : {(tp+tn)/len(y):.4f}")
print(f"Sensitivity AI: {tp/(tp+fn):.4f}")
print(f"Specificity    : {tn/(tn+fp):.4f}")
print(f"ROC-AUC       : {auc_val:.4f}")

# -----------------------------------------------------------------
# 5. ROC CURVE  -----------------------------------------------------
# -----------------------------------------------------------------
fpr, tpr, _ = roc_curve(y, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc_val:.3f})")
plt.plot([0,1], [0,1], "--", lw=1)
plt.xlabel("False Positive Rate (1 − Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC curve – Human vs AI (GPT-4o & Claude 3·7)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("logreg_whitelist_ROC.png", dpi=300)
plt.show()

# -----------------------------------------------------------------
# 6. OPTIONAL: most influential questions --------------------------
# -----------------------------------------------------------------
clf.fit(X, y)                     # final fit on all data
coef = clf.coef_[0]
top  = np.argsort(np.abs(coef))[::-1][:10]
print("\nTop-10 weighted questions:")
for rank, idx in enumerate(top, 1):
    sign = "➚" if coef[idx] > 0 else "➘"
    print(f"{rank:2}. Q{idx+1:>2}  {sign}  weight = {coef[idx]:+.3f}")

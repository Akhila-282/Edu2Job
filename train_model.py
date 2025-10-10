import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("Dataset.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# =========================
# 2. Preprocess Skills & Certifications
# =========================
df['skills'] = df['skills'].apply(lambda x: [s.strip().lower() for s in x.split(',')] if pd.notna(x) else [])
df['certifications'] = df['certifications'].apply(lambda x: [c.strip().lower() for c in x.split(',')] if pd.notna(x) else [])

# =========================
# 3. Encode Categorical Columns
# =========================
label_encoders = {}
for col in ["degree", "major", "industry_preference", "employed"]:
    df[col] = df[col].astype(str).str.strip().str.title()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 4. Multi-Label Encode Skills & Certifications
# =========================
skills_encoder = MultiLabelBinarizer()
skills_encoded = skills_encoder.fit_transform(df['skills'])

certs_encoder = MultiLabelBinarizer()
certs_encoded = certs_encoder.fit_transform(df['certifications'])

# =========================
# 5. Combine All Features
# =========================
X_numeric = df[["degree", "major", "cgpa", "experience", "industry_preference", "employed"]].values
X = np.hstack([X_numeric, skills_encoded, certs_encoded])

# =========================
# 6. Encode Target (Job Role)
# =========================
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["job_role"].astype(str))

# =========================
# 7. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

# =========================
# 8. Train Model (RandomForest)
# =========================
model = RandomForestClassifier(n_estimators=10000, random_state=42)
model.fit(X_train, y_train)

# =========================
# 9. Evaluate Model
# =========================
y_pred = model.predict(X_test)
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# 10. Save Model & Encoders
# =========================
with open("jobrole_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

with open("feature_encoders.pkl", "wb") as f:
    pickle.dump({
        "label_encoders": label_encoders,
        "skills_encoder": skills_encoder,
        "certs_encoder": certs_encoder
    }, f)

print("🎉 Model trained & saved successfully!")
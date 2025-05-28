# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ( confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv('C:/bengkod/ObesityDataSet.csv')

# ===============================
# 1. Tangani Missing Values, Error, Duplikasi
# ===============================
print("===== Missing Values =====")
print(df.isnull().sum())  # Tampilkan jumlah missing values

print("\n===== Duplikasi =====")
print(f"Jumlah duplikat: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# (Opsional) Drop missing values jika ada
df.dropna(inplace=True)

# ===============================
# 2. Tangani Outlier
# ===============================
# Identifikasi kolom numerik (agar tidak error pada kolom string)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Terapkan untuk setiap kolom numerik
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# ===============================
# 3. Encoding Data Kategori
# ===============================
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('NObeyesdad')  # Target label diproses terpisah

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Encode target label
target_encoder = LabelEncoder()
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

# ===============================
# 4. Seleksi Fitur (Opsional)
# ===============================
# Untuk saat ini semua fitur digunakan, seleksi bisa dilakukan di tahap modeling

# ===============================
# 5. Tangani Ketidakseimbangan Kelas
# ===============================
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

print("\nDistribusi Kelas Sebelum SMOTE:")
print(y.value_counts())

# SMOTE untuk oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nDistribusi Kelas Setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())

# ===============================
# 6. Normalisasi / Standarisasi
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# ===============================
# 7. Final Output dan Ringkasan
# ===============================
# Konversi kembali ke DataFrame untuk keperluan lanjutan
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
y_resampled_df = pd.DataFrame(y_resampled, columns=['NObeyesdad'])

# Gabungkan menjadi satu DataFrame
processed_df = pd.concat([X_scaled_df, y_resampled_df], axis=1)

# (Opsional) Simpan dataset hasil preprocessing
# processed_df.to_csv('C:/bengkod/ObesityDataSet_preprocessed.csv', index=False)

print("\n===== PREPROCESSING SELESAI =====")
print(f"Shape data akhir (setelah SMOTE dan scaling): {processed_df.shape}")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_resampled_df, test_size=0.2, random_state=42, stratify=y_resampled_df
)

# Definisikan model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# Simpan hasil evaluasi
metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Melatih dan evaluasi tiap model
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics["Model"].append(name)
    metrics["Accuracy"].append(acc)
    metrics["Precision"].append(prec)
    metrics["Recall"].append(rec)
    metrics["F1 Score"].append(f1)

    # Tampilkan confusion matrix
    print(f"\n===== Confusion Matrix: {name} =====")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

# ===============================
# 9. Visualisasi Perbandingan Model
# ===============================
metrics_df = pd.DataFrame(metrics)

# Bar plot performa model
plt.figure(figsize=(12, 6))
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_melted)
plt.title("Perbandingan Performa Model")
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.show()

# ===============================
# 10. Kesimpulan
# ===============================
print("\n===== KESIMPULAN =====")
best_model = metrics_df.loc[metrics_df['F1 Score'].idxmax()]
print(f"Model dengan performa terbaik berdasarkan F1 Score:")
print(best_model)

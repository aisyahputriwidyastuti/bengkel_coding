# Import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('C:/bengkod/ObesityDataSet.csv')

# -------------------------------
# 1. Tampilkan beberapa baris pertama dan informasi umum
print("===== 5 Baris Pertama =====")
print(df.head())

print("\n===== Info Dataset =====")
print(df.info())

print("\n===== Deskripsi Statistik =====")
print(df.describe(include='all'))

# -------------------------------
# 2. Cek missing values, unique values, data duplikat
print("\n===== Cek Missing Values =====")
print(df.isnull().sum())

print("\n===== Jumlah Unique Values Tiap Kolom =====")
print(df.nunique())

print("\n===== Cek Data Duplikat =====")
print(df.duplicated().sum())

# -------------------------------
# 3. Analisis keseimbangan data (Target class balance)
print("\n===== Distribusi Target (NObeyesdad) =====")
print(df['NObeyesdad'].value_counts())

# Visualisasi distribusi target
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='NObeyesdad', hue='NObeyesdad', order=df['NObeyesdad'].value_counts().index, palette='Set2', legend=False)
plt.xticks(rotation=45)
plt.title('Distribusi Kelas Target (NObeyesdad)')
plt.tight_layout()
plt.show()


# -------------------------------
# 4. Deteksi outlier menggunakan boxplot
num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Visualisasi distribusi fitur numerik
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, color='salmon')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()

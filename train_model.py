import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Sedang memproses data...")

# 1. meLoad Data
# memaastikan file csv ada di folder yang sama dengan script ini
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')

# 2. Bersih-bersih Data error
# File Training.csv dari Kaggle sering punya kolom error kosong di akhir, dibuang sek
if 'Unnamed: 133' in train_df.columns:
    train_df = train_df.drop(columns=['Unnamed: 133'])

# 3. Pisahkan Gejala (X) dan Penyakit (Y)
X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']

X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

# 4. Melatih model
print("Sedang melatih AI...")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 5. Cek Kepintaran Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}% (Sangat bagus!)")

# 6. mentimpan Model dan Daftar Gejala
# Kita butuh daftar gejala agar nanti Backend tahu urutan inputnya
with open('model_penyakit.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('daftar_gejala.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Selesai! File 'model_penyakit.pkl' dan 'daftar_gejala.pkl' sudah berhasil dibuat.")


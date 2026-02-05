import pandas as pd
import glob
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_and_convert_data(folder_path):
    combined_data = []
    
    # Cari semua file .csv di dalam folder dataset
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"📂 Ditemukan {len(files)} file dataset: {[os.path.basename(f) for f in files]}")

    for file in files:
        try:
            df = pd.read_csv(file)
            
            # KASUS 1: Dataset Cerita (Symptom2Disease.csv)
            # Cirinya: Punya kolom 'text' dan 'label'
            if 'text' in df.columns and 'label' in df.columns:
                print(f"   -> Memproses '{os.path.basename(file)}' sebagai Format NLP (Cerita)...")
                # Ambil hanya kolom yang dibutuhkan
                df_clean = df[['text', 'label']]
                combined_data.append(df_clean)

            # KASUS 2: Dataset Angka/Checklist (Training.csv / Testing.csv)
            # Cirinya: Punya kolom 'prognosis' (target penyakit)
            elif 'prognosis' in df.columns:
                print(f"   -> Memproses '{os.path.basename(file)}' sebagai Format Checklist (Angka)...")
                
                # Kita harus ubah angka 0/1 menjadi kalimat agar bisa digabung dengan data cerita
                def row_to_sentence(row):
                    # Ambil nama gejala yang nilainya 1 (ada)
                    symptoms = [col.replace('_', ' ') for col in df.columns[:-1] if row[col] == 1]
                    # Gabungkan jadi kalimat
                    return "I have " + ", ".join(symptoms)

                # Buat DataFrame baru dengan format text & label
                df_converted = pd.DataFrame()
                df_converted['text'] = df.apply(row_to_sentence, axis=1)
                df_converted['label'] = df['prognosis']
                
                combined_data.append(df_converted)
            
            else:
                print(f"   ⚠️ PERINGATAN: File '{os.path.basename(file)}' formatnya tidak dikenali. Dilewati.")

        except Exception as e:
            print(f"   ❌ Error membaca file {file}: {e}")

    # Gabungkan semua data jadi satu
    if not combined_data:
        print("❌ Tidak ada data yang berhasil dimuat!")
        return None

    final_df = pd.concat(combined_data, ignore_index=True)
    print(f"✅ Total Data Digabungkan: {len(final_df)} baris data.")
    return final_df

# ==========================================
# EKSEKUSI UTAMA
# ==========================================
print("1. Memulai Smart Data Loading...")
df_total = load_and_convert_data('dataset') # Folder tempat nyimpen csv

if df_total is not None:
    # Pisahkan Data Latih & Uji
    X = df_total['text']
    y = df_total['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n2. Melatih Model AI (TF-IDF + SVM)...")
    # Pipeline: Ubah teks ke angka -> Masukkan ke algoritma SVM
    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(X_train, y_train)

    # Evaluasi
    print("3. Menguji Akurasi...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"   🎯 Akurasi Model Gabungan: {acc * 100:.2f}%")

    # Simpan
    print("4. Menyimpan Model...")
    with open('model_nlp.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("🎉 Selesai! Model baru sudah tersimpan.")

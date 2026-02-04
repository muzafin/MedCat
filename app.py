from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# =================================================================================
# 1. LOAD OTAK AI (MODEL & DATA)
# =================================================================================
print("Sedang memuat model AI...")
try:
    with open('model_penyakit.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('daftar_gejala.pkl', 'rb') as f:
        col_names = pickle.load(f)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan kamu sudah menjalankan train_model.py sebelumnya.")
    exit()

# =================================================================================
# 2. KAMUS BAHASA (Indonesia -> Dataset Kaggle)
# =================================================================================
# Ini trik supaya AI mengerti bahasa Indonesia. 
# Kiri: Kata yang mungkin diketik user. Kanan: Nama kolom di CSV Kaggle.
kamus_gejala = {
    # --- DEMAM & FLU ---
    'demam': 'high_fever',
    'badan panas': 'high_fever',
    'menggigil': 'chills',
    'kedinginan': 'chills',
    'bersin': 'continuous_sneezing',
    'pilek': 'runny_nose',
    'hidung meler': 'runny_nose',
    'hidung tersumbat': 'congestion',
    'batuk': 'cough',
    'sakit tenggorokan': 'throat_irritation',
    'tenggorokan gatal': 'throat_irritation',
    'sesak': 'breathlessness',
    'susah napas': 'breathlessness',

    # --- KEPALA & SARAF ---
    'pusing': 'headache',
    'sakit kepala': 'headache',
    'kepala sakit': 'headache', # Variasi baru
    'migrain': 'headache',
    'kunang-kunang': 'dizziness',
    'kliyengan': 'dizziness',
    'leher kaku': 'stiff_neck',
    'susah konsentrasi': 'lack_of_concentration',

    # --- PERUT & PENCERNAAN ---
    'sakit perut': 'stomach_pain',
    'perut sakit': 'stomach_pain',  # <--- INI SOLUSI MASALAHMU
    'nyeri perut': 'abdominal_pain',
    'perut nyeri': 'abdominal_pain',
    'kembung': 'distention_of_abdomen',
    'mual': 'nausea',
    'pengen muntah': 'nausea',
    'muntah': 'vomiting',
    'diare': 'diarrhoea',
    'mencret': 'diarrhoea',
    'bab cair': 'diarrhoea',
    'sembelit': 'constipation',
    'susah bab': 'constipation',
    'nafsu makan hilang': 'loss_of_appetite',
    'tidak nafsu makan': 'loss_of_appetite',
    
    # --- KULIT ---
    'gatal': 'itching',
    'ruam': 'skin_rash',
    'bintik merah': 'skin_rash',
    'kulit merah': 'skin_rash',
    'jerawat': 'pus_filled_pimples',
    'kulit kuning': 'yellowish_skin',

    # --- MATA ---
    'mata kuning': 'yellowing_of_eyes',
    'mata merah': 'redness_of_eyes',
    'penglihatan kabur': 'blurred_and_distorted_vision',

    # --- OTOT & SENDI ---
    'nyeri sendi': 'joint_pain',
    'sendi sakit': 'joint_pain',
    'nyeri otot': 'muscle_pain',
    'otot sakit': 'muscle_pain',
    'badan pegal': 'muscle_pain',
    'sakit punggung': 'back_pain',
    'pinggang sakit': 'back_pain',
    
    # --- JANTUNG & DADA ---
    'nyeri dada': 'chest_pain',
    'dada sakit': 'chest_pain',
    'jantung berdebar': 'fast_heart_rate',
    
    # --- UMUM ---
    'lelah': 'fatigue',
    'capek': 'fatigue',
    'lemas': 'fatigue',
    'berat badan turun': 'weight_loss',
    'gelisah': 'restlessness'
}

# =================================================================================
# 3. FUNGSI EKSTRAKSI (Simulasi MedCAT)
# =================================================================================
def extract_symptoms(text):
    """
    Fungsi ini membaca cerita user dan mencari kata kunci gejala.
    Di project besar, bagian ini digantikan oleh MedCAT.
    """
    text = text.lower() # Ubah ke huruf kecil semua
    found_symptoms = []
    
    # Cek Kamus Indonesia
    for kata_indo, kode_dataset in kamus_gejala.items():
        if kata_indo in text:
            if kode_dataset in col_names: # Pastikan gejalanya dikenal model
                found_symptoms.append(kode_dataset)
    
    # (Opsional) Cek juga bahasa Inggris langsung, jaga-jaga user ngetik 'headache'
    for col in col_names:
        readable_col = col.replace('_', ' ') # ubah 'high_fever' jadi 'high fever'
        if readable_col in text:
            found_symptoms.append(col)
            
    # Hapus duplikat (set) lalu kembalikan jadi list
    return list(set(found_symptoms))

# =================================================================================
# 4. API ROUTE (Pintu Masuk Website)
# =================================================================================
@app.route('/', methods=['GET'])
def home():
    return "Halo! Backend AI Penyakit sudah aktif. Kirim data ke /predict ya."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari user
        data = request.json
        cerita_user = data.get('cerita', '')
        
        if not cerita_user:
            return jsonify({'status': 'error', 'pesan': 'Cerita tidak boleh kosong'}), 400

        # STEP 1: Ekstrak Gejala
        gejala_terdeteksi = extract_symptoms(cerita_user)
        
        if not gejala_terdeteksi:
            return jsonify({
                'status': 'sukses',
                'pesan': 'Maaf, AI belum mengenali gejala spesifik dari ceritamu. Coba gunakan kata baku seperti "demam", "sakit perut", atau "batuk".',
                'hasil_diagnosa': 'Tidak diketahui',
                'gejala_ditemukan': []
            })

        # STEP 2: Siapkan Input untuk Model (Vector 0 dan 1)
        # Buat baris data kosong berisi 0 semua
        input_data = pd.DataFrame(0, index=[0], columns=col_names)
        
        # Isi 1 untuk gejala yang ditemukan
        for g in gejala_terdeteksi:
            input_data[g] = 1
            
        # STEP 3: Prediksi Penyakit
        prediksi_penyakit = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'sukses',
            'hasil_diagnosa': prediksi_penyakit,
            'gejala_ditemukan': gejala_terdeteksi,
            'cerita_original': cerita_user
        })

    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
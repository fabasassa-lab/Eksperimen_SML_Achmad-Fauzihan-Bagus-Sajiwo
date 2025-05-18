import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_air_quality(file_path, output_path='preprocessing/air_quality_preprocessed.csv'):
    # Baca data
    df = pd.read_csv(file_path)

    # Hapus kolom yang tidak diperlukan
    drop_cols = ['Date', 'Max', 'NO2', 'Critical Component']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Tentukan fitur numerik dan kategorikal
    numeric_cols = ['PM10', 'SO2', 'CO', 'O3']
    categorical_col = 'Category'

    # Scaling numerik dengan MinMaxScaler
    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])
    numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_cols)

    # Label encoding untuk kolom kategori
    le = LabelEncoder()
    category_encoded = le.fit_transform(df[categorical_col])
    category_df = pd.DataFrame({categorical_col: category_encoded})

    # Gabungkan data akhir
    final_df = pd.concat([numeric_df, category_df], axis=1)

    # Pastikan direktori output tersedia
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan hasil preprocessing
    final_df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai. File disimpan di: {output_path}")


if __name__ == "__main__":
    preprocess_air_quality("ispu_raw/pollutant-standards-index-jogja-2020.csv")

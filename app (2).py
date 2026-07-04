import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Analisis Padi 2025", layout="wide", page_icon="🌾")

# --- FUNGSI LOAD & CLEAN DATA ---
@st.cache_data
def load_and_clean_data(file_path):
    # Membaca data
    df = pd.read_excel(file_path)
    
    # Merapikan nama kolom
    df.columns = ['Provinsi', 'Luas_Panen', 'Produktivitas', 'Produksi']
    
    # DATA CLEANING (Sesuai Laporan Bagian B)
    # 1. Hapus baris agregat "Indonesia" dan baris kosong/catatan kaki
    df = df[df['Provinsi'] != 'Indonesia'].dropna(subset=['Provinsi'])
    df = df[df['Provinsi'].str.strip() != '']
    
    # 2. Konversi tipe data ke numerik
    for col in ['Luas_Panen', 'Produktivitas', 'Produksi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 3. Drop missing values (jika ada)
    df = df.dropna()
    
    return df

# --- LOAD DATA ---
FILE_PATH = 'Luas Panen, Produktivitas, dan Produksi Padi Menurut Provinsi, 2025.xlsx'
try:
    df = load_and_clean_data(FILE_PATH)
except FileNotFoundError:
    st.error("File Excel tidak ditemukan! Pastikan nama file sesuai dan berada di folder yang sama.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🌾 Filter & Navigasi")
st.sidebar.markdown("Dashboard Analisis Sektor Pertanian: Komoditas Padi (BPS 2025)")

top_n = st.sidebar.slider("Tampilkan Top N Provinsi (Produksi)", 5, 38, 10)
selected_island = st.sidebar.multiselect(
    "Filter Berdasarkan Pulau/Wilayah (Contoh)", 
    options=['Jawa', 'Sumatera', 'Kalimantan', 'Sulawesi', 'Bali & Nusra', 'Papua & Maluku'],
    default=[]
)

# --- HEADER ---
st.title("📊 Dashboard Analisis Data Padi Indonesia 2025")
st.markdown("*Sumber Data: BPS 2025 | Analisis: Data Scientist Sektor Pertanian*")

# --- METRIK UTAMA (KPI) ---
st.markdown("### 📌 Ringkasan Statistik Nasional (38 Provinsi)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Produksi", f"{df['Produksi'].sum() / 1e6:.2f} Juta Ton")
col2.metric("Total Luas Panen", f"{df['Luas_Panen'].sum() / 1e6:.2f} Juta Ha")
col3.metric("Rata-rata Produktivitas", f"{df['Produktivitas'].mean():.2f} Ku/Ha")
col4.metric("Jumlah Provinsi", f"{len(df)} Provinsi")

st.markdown("---")

# --- VISUALISASI 1: TOP PRODUKSI ---
st.subheader("1. Provinsi Penyumbang Produksi Padi Tertinggi")
st.caption("Insight: Jawa Timur, Jawa Barat, dan Jawa Tengah mendominasi produksi nasional.")
top_prod = df.nlargest(top_n, 'Produksi')
fig1 = px.bar(top_prod, x='Provinsi', y='Produksi', color='Produksi',
              color_continuous_scale='Greens', title=f"Top {top_n} Produksi Padi (Ton)")
fig1.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

# --- VISUALISASI 2: DISTRIBUSI PRODUKSI ---
st.subheader("2. Distribusi Produksi Padi Antar Provinsi")
st.caption("Insight: Distribusi menceng ke kanan (right-skewed). Sebagian besar provinsi < 2 juta ton.")
fig2 = px.histogram(df, x='Produksi', nbins=20, marginal='box', 
                    title="Distribusi Produksi Padi (Ton)")
st.plotly_chart(fig2, use_container_width=True)

# --- VISUALISASI 3: PRODUKTIVITAS ---
st.subheader("3. Produktivitas Padi Antar Provinsi (Ku/Ha)")
st.caption("Insight: Bali tertinggi (~60.95), Papua terendah (<30). Kesenjangan produktivitas sangat besar.")
df_sorted_prod = df.sort_values(by='Produktivitas', ascending=True)
fig3 = px.bar(df_sorted_prod, x='Produktivitas', y='Provinsi', orientation='h',
              color='Produktivitas', color_continuous_scale='Blues',
              title="Produktivitas Padi per Provinsi")
st.plotly_chart(fig3, use_container_width=True)

# --- VISUALISASI 4: KORELASI (SCATTER PLOT) ---
st.subheader("4. Hubungan Luas Panen vs Produksi Padi")
st.caption("Insight: Korelasi positif sangat kuat (r=0.998). Produksi didominasi oleh luas lahan, bukan hanya efisiensi.")
fig4 = px.scatter(df, x='Luas_Panen', y='Produksi', hover_data=['Provinsi'],
                  color='Produktivitas', size='Produksi', size_max=30,
                  title="Scatter Plot: Luas Panen vs Produksi (Ukuran & Warna = Produktivitas)")
st.plotly_chart(fig4, use_container_width=True)

# --- PEMODELAN REGRESI LINEAR ---
st.markdown("---")
st.subheader("5. Pemodelan Regresi Linear: Aktual vs Prediksi")
st.markdown("Model memprediksi produksi berdasarkan Luas Panen dan Produktivitas.")

# Melatih Model
X = df[['Luas_Panen', 'Produktivitas']]
y = df['Produksi']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrik Evaluasi
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

m1, m2, m3 = st.columns(3)
m1.metric("R-Squared (R²)", f"{r2:.4f}", help="Model mampu menjelaskan 98.88% variasi produksi")
m2.metric("MAE (Ton)", f"{mae:,.0f}")
m3.metric("RMSE (Ton)", f"{rmse:,.0f}")

# Plot Aktual vs Prediksi
fig5 = px.scatter(x=y, y=y_pred, hover_data=[df['Provinsi']],
                  labels={'x': 'Produksi Aktual (Ton)', 'y': 'Produksi Prediksi (Ton)'},
                  title="Grafik Aktual vs Prediksi Model")
# Garis diagonal sempurna
fig5.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                          mode='lines', name='Garis Prediksi Sempurna', 
                          line=dict(dash='dash', color='red')))
st.plotly_chart(fig5, use_container_width=True)

# Persamaan Model
st.info(f"**Persamaan Model Regresi:** `Produksi = ({model.coef_[0]:,.2f} × Luas Panen) + ({model.coef_[1]:,.2f} × Produktivitas) − {abs(model.intercept_):,.2f}`")

# --- INSIGHT & REKOMENDASI KEBIJAKAN ---
st.markdown("---")
st.subheader("💡 Insight & Rekomendasi Kebijakan")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🔍 Key Insights")
    st.markdown("""
    1. **Konsentrasi Timpang:** 3 provinsi di Jawa mendominasi produksi nasional.
    2. **Faktor Pendorong:** Luas panen (ekstensifikasi) adalah pendorong utama produksi, bukan produktivitas.
    3. **Kesenjangan Wilayah:** Produktivitas Bali (60.95) vs Papua (<30) menunjukkan gap teknologi yang besar.
    4. **Kontribusi Timur:** Kawasan Timur Indonesia berkontribusi sangat minim terhadap total nasional.
    """)

with col_right:
    st.markdown("#### 🚀 Rekomendasi Kebijakan")
    st.markdown("""
    1. **Intensifikasi:** Prioritaskan distribusi bibit unggul & pupuk di provinsi produktivitas rendah.
    2. **Ekstensifikasi Terukur:** Kajian perluasan lahan di Kalimantan & Sulawesi.
    3. **Mitigasi Iklim:** Perkuat irigasi di 3 provinsi utama (Jatim, Jabar, Jateng).
    4. **Diferensiasi Regional:** Pertanian presisi di lahan sempit, perbaikan budidaya di lahan luas.
    5. **Dashboard Monitoring:** Bangun sistem monitoring data pertanian berkala berbasis provinsi.
    """)

# --- FOOTER ---
st.markdown("---")
st.caption("Dibuat dengan Streamlit & Plotly | Laporan Analisis Data Sains - UAS Pengenalan Sains Data")

"""
Dashboard Analisis Sektor Pertanian: Komoditas Padi Menurut Provinsi (2025)
Sumber data: BPS - Luas Panen, Produktivitas, dan Produksi Padi Menurut Provinsi
Cara menjalankan:
    pip install streamlit pandas plotly scikit-learn
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Padi Indonesia 2025",
    page_icon="🌾",
    layout="wide",
)

PRIMARY_GREEN = "#2E7D32"
LIGHT_GREEN = "#81C784"
ORANGE = "#F57C00"

# -----------------------------------------------------------------------
# LOAD & CLEAN DATA
# -----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("padi_provinsi_clean.csv")
    df.columns = ["provinsi", "luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"]
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# -----------------------------------------------------------------------
# SIDEBAR - FILTER
# -----------------------------------------------------------------------
st.sidebar.title("🌾 Filter Dashboard")
st.sidebar.markdown("Gunakan filter di bawah untuk menyesuaikan tampilan data.")

prov_list = sorted(df["provinsi"].unique())
selected_prov = st.sidebar.multiselect(
    "Pilih Provinsi (kosongkan = semua provinsi)",
    options=prov_list,
    default=[],
)

min_luas, max_luas = float(df["luas_panen_ha"].min()), float(df["luas_panen_ha"].max())
luas_range = st.sidebar.slider(
    "Rentang Luas Panen (ha)",
    min_value=min_luas, max_value=max_luas,
    value=(min_luas, max_luas),
)

top_n = st.sidebar.slider("Jumlah provinsi pada Top-N chart", 5, 38, 10)

st.sidebar.markdown("---")
st.sidebar.caption("Sumber: BPS, Luas Panen, Produktivitas, dan Produksi Padi Menurut Provinsi, 2025")

# Terapkan filter
filtered = df.copy()
if selected_prov:
    filtered = filtered[filtered["provinsi"].isin(selected_prov)]
filtered = filtered[
    (filtered["luas_panen_ha"] >= luas_range[0]) & (filtered["luas_panen_ha"] <= luas_range[1])
]

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🌾 Dashboard Analisis Sektor Pertanian: Komoditas Padi")
st.markdown(
    "**Analisis Luas Panen, Produktivitas, dan Produksi Padi Menurut Provinsi — Indonesia, 2025 (BPS)**"
)
st.markdown("---")

# -----------------------------------------------------------------------
# KPI CARDS
# -----------------------------------------------------------------------
total_produksi = filtered["produksi_ton"].sum()
total_luas = filtered["luas_panen_ha"].sum()
avg_produktivitas = filtered["produktivitas_ku_ha"].mean()
n_provinsi = filtered["provinsi"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Produksi", f"{total_produksi:,.0f} ton")
k2.metric("Total Luas Panen", f"{total_luas:,.0f} ha")
k3.metric("Rata-rata Produktivitas", f"{avg_produktivitas:,.2f} ku/ha")
k4.metric("Jumlah Provinsi", f"{n_provinsi}")

st.markdown("---")

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Eksplorasi Data", "🔗 Korelasi", "📈 Model Regresi", "🌳 Random Forest", "💡 Insight & Rekomendasi"]
)

# ===================== TAB 1: EDA =====================
with tab1:
    st.subheader("Distribusi Produksi Padi per Provinsi")
    fig_hist = px.histogram(
        filtered, x="produksi_ton", nbins=15,
        color_discrete_sequence=[PRIMARY_GREEN],
        labels={"produksi_ton": "Produksi Padi (ton)"},
    )
    fig_hist.update_layout(yaxis_title="Jumlah Provinsi", bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Distribusi produksi padi menceng ke kanan (right-skewed): sebagian besar provinsi "
        "berproduksi di bawah 2 juta ton, sedangkan beberapa provinsi jauh lebih tinggi."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Luas Panen vs Produksi")
        fig_scatter = px.scatter(
            filtered, x="luas_panen_ha", y="produksi_ton",
            color="produktivitas_ku_ha", hover_name="provinsi",
            color_continuous_scale="Greens",
            labels={
                "luas_panen_ha": "Luas Panen (ha)",
                "produksi_ton": "Produksi (ton)",
                "produktivitas_ku_ha": "Produktivitas (ku/ha)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Produktivitas Padi antar Provinsi (Diurutkan)")
        sorted_df = filtered.sort_values("produktivitas_ku_ha", ascending=False)
        fig_line = px.line(
            sorted_df, x="provinsi", y="produktivitas_ku_ha", markers=True,
            color_discrete_sequence=[PRIMARY_GREEN],
            labels={"produktivitas_ku_ha": "Produktivitas (ku/ha)", "provinsi": ""},
        )
        fig_line.add_hline(
            y=df["produktivitas_ku_ha"].mean(), line_dash="dash", line_color="red",
            annotation_text=f"Rata-rata nasional ({df['produktivitas_ku_ha'].mean():.1f} ku/ha)",
        )
        fig_line.update_xaxes(tickangle=45)
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader(f"Top {top_n} Provinsi dengan Produksi Padi Tertinggi")
    top_df = filtered.sort_values("produksi_ton", ascending=False).head(top_n)
    fig_bar = px.bar(
        top_df.sort_values("produksi_ton"), x="produksi_ton", y="provinsi",
        orientation="h", color_discrete_sequence=[ORANGE],
        text="produksi_ton",
        labels={"produksi_ton": "Produksi (ton)", "provinsi": ""},
    )
    fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Boxplot Variabel Numerik (Ternormalisasi Z-Score)")
    norm_df = filtered[["luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"]].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    norm_df.columns = ["Luas Panen", "Produktivitas", "Produksi"]
    fig_box = px.box(norm_df.melt(var_name="Variabel", value_name="Z-Score"),
                      x="Variabel", y="Z-Score", color="Variabel",
                      color_discrete_sequence=["#64B5F6", LIGHT_GREEN, "#FF8A65"])
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Lihat Statistik Deskriptif"):
        st.dataframe(filtered.describe().T.style.format("{:,.2f}"), use_container_width=True)

    with st.expander("Lihat Data Mentah (Setelah Cleaning)"):
        st.dataframe(filtered, use_container_width=True)

# ===================== TAB 2: KORELASI =====================
with tab2:
    st.subheader("Heatmap Korelasi Antar Variabel")
    corr = df[["luas_panen_ha", "produktivitas_ku_ha", "produksi_ton"]].corr()
    corr.index = corr.columns = ["Luas Panen", "Produktivitas", "Produksi"]
    fig_heat = px.imshow(
        corr, text_auto=".3f", color_continuous_scale="Greens",
        zmin=-1, zmax=1, aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        f"""
**Variabel Paling Berpengaruh: Luas Panen**

Luas panen memiliki korelasi **{corr.loc['Luas Panen','Produksi']:.3f}** terhadap produksi (sangat kuat),
dibanding produktivitas yang hanya **{corr.loc['Produktivitas','Produksi']:.3f}** (moderat).
Hal ini logis secara agronomis karena Produksi ≈ Luas Panen × Produktivitas, dan variasi luas panen
antar provinsi jauh lebih besar dibanding variasi produktivitas.
        """
    )

# ===================== TAB 3: REGRESI LINEAR =====================
with tab3:
    st.subheader("Model Regresi Linear Berganda")
    st.markdown("**Variabel independen (X):** luas_panen_ha, produktivitas_ku_ha &nbsp;|&nbsp; **Variabel dependen (y):** produksi_ton")

    test_size = st.slider("Proporsi data uji (test size)", 0.1, 0.4, 0.2, 0.05, key="lr_test")
    random_state = st.number_input("random_state", value=42, step=1, key="lr_rs")

    X = df[["luas_panen_ha", "produktivitas_ku_ha"]]
    y = df["produksi_ton"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    b0, b1, b2 = lr.intercept_, lr.coef_[0], lr.coef_[1]
    st.info(f"**Persamaan model:** Produksi = {b1:,.2f} × Luas Panen + {b2:,.2f} × Produktivitas + ({b0:,.2f})")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:,.2f} ton")
    m2.metric("RMSE", f"{rmse:,.2f} ton")
    m3.metric("R²", f"{r2:.4f}")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                   marker=dict(color=PRIMARY_GREEN, size=10),
                                   name="Prediksi"))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig_pred.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                   line=dict(color="red", dash="dash"),
                                   name="Garis Prediksi Sempurna (y=x)"))
    fig_pred.update_layout(
        title=f"Aktual vs Prediksi Model Regresi Linear (R² = {r2:.3f})",
        xaxis_title="Produksi Aktual (ton)", yaxis_title="Produksi Prediksi (ton)",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# ===================== TAB 4: RANDOM FOREST =====================
with tab4:
    st.subheader("Random Forest Regressor (Model Pembanding)")
    n_estimators = st.slider("Jumlah pohon (n_estimators)", 50, 500, 200, 50)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (RF)", f"{mae_rf:,.2f} ton")
    c2.metric("RMSE (RF)", f"{rmse_rf:,.2f} ton")
    c3.metric("R² (RF)", f"{r2_rf:.4f}")

    importance_df = pd.DataFrame({
        "Fitur": ["luas_panen_ha", "produktivitas_ku_ha"],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance_df, x="Fitur", y="Importance",
        color="Fitur", color_discrete_sequence=[PRIMARY_GREEN, ORANGE],
        text="Importance",
    )
    fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_imp.update_layout(title="Feature Importance - Random Forest", showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.caption(
        "Feature importance dari Random Forest konsisten dengan hasil analisis korelasi: "
        "luas panen adalah fitur dengan kontribusi terbesar terhadap prediksi produksi padi."
    )

# ===================== TAB 5: INSIGHT =====================
with tab5:
    st.subheader("💡 Insight Utama")
    st.markdown(
        """
1. **Konsentrasi produksi padi sangat timpang secara geografis** — tiga provinsi di Pulau Jawa
   (Jawa Timur, Jawa Barat, Jawa Tengah) mendominasi produksi nasional.
2. **Luas panen adalah pendorong utama produksi**, bukan produktivitas — strategi peningkatan
   produksi selama ini lebih banyak mengandalkan ekstensifikasi lahan.
3. **Kesenjangan produktivitas antar wilayah masih besar**, antara Bali (tertinggi, ~60,95 ku/ha)
   dan sejumlah provinsi di Papua (terendah, di bawah 30 ku/ha).
4. **Provinsi di kawasan timur Indonesia** memiliki luas panen dan produksi yang sangat kecil,
   sehingga kontribusinya terhadap produksi nasional minim.
5. **Model regresi linear sederhana** dengan dua variabel sudah mampu menjelaskan variasi produksi
   padi antar provinsi dengan sangat baik (R² = 0,9888), mengonfirmasi dominasi faktor teknis lahan.
        """
    )

    st.subheader("📌 Rekomendasi Kebijakan")
    st.markdown(
        """
1. **Prioritaskan program intensifikasi pertanian** di provinsi dengan produktivitas rendah melalui
   distribusi bibit unggul, pupuk bersubsidi, dan pelatihan petani.
2. **Kaji perluasan lahan panen (ekstensifikasi)** secara terukur di wilayah berpotensi namun luas
   panennya masih kecil, khususnya Kalimantan dan sebagian Sulawesi.
3. **Perkuat infrastruktur irigasi dan mitigasi risiko iklim** di tiga provinsi utama penghasil padi
   (Jawa Timur, Jawa Barat, Jawa Tengah).
4. **Kembangkan kebijakan diferensiasi regional**: wilayah produktivitas tinggi namun lahan terbatas
   diarahkan pada pertanian presisi; wilayah luas namun produktivitas rendah diarahkan pada
   perbaikan teknik budidaya.
5. **Bangun sistem monitoring data pertanian berkala (dashboard)** berbasis provinsi agar kebijakan
   alokasi anggaran pertanian lebih tepat sasaran dan berbasis data.
        """
    )

    st.success(
        "**Kesimpulan:** Produksi padi nasional sangat dipengaruhi oleh luas panen, dengan hubungan "
        "yang kuat dan dapat dimodelkan secara linear (R² = 0,9888). Produksi terkonsentrasi di Pulau "
        "Jawa dan sebagian Sulawesi Selatan, sementara wilayah timur Indonesia masih berkontribusi "
        "sangat kecil. Kombinasi kebijakan intensifikasi dan ekstensifikasi yang tepat sasaran per "
        "wilayah diperlukan untuk memperkuat ketahanan pangan nasional secara berkelanjutan."
    )

st.markdown("---")
st.caption("Dashboard dibuat untuk UAS Pengenalan Sains Data — Program Studi Sains Data, UIN K.H. Abdurrahman Wahid Pekalongan.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import graphviz
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="SIMTANK - Prediksi Maintenance Armada Tangki",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. GENERASI & PREPROCESSING DATASET (130 UNIT ARMADA)
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    np.random.seed(42)
    n_samples = 130
    
    armada_ids = [f"B {8000 + i} SPU" for i in range(1, n_samples + 1)]
    usia_list = np.random.randint(1, 12, size=n_samples)
    odometer_list = np.random.randint(20000, 350000, size=n_samples)
    jenis_mesin = np.random.choice(['Diesel Euro 4', 'Diesel Euro 2'], size=n_samples, p=[0.6, 0.4])
    beban_muatan = np.random.choice(['16 KL', '24 KL', '32 KL'], size=n_samples, p=[0.5, 0.3, 0.2])
    riwayat_servis = np.random.choice(['Teratur', 'Jarang', 'Tidak Pernah'], size=n_samples, p=[0.6, 0.25, 0.15])
    
    # Penentuan Label berdasarkan Aturan Bisnis C4.5
    status_maintenance = []
    for u, o, r in zip(usia_list, odometer_list, riwayat_servis):
        if r == 'Tidak Pernah' or o > 250000 or u > 8:
            status_maintenance.append('Maintenance Segera')
        elif r == 'Jarang' or o > 120000 or u > 4:
            status_maintenance.append('Maintenance Rutin')
        else:
            status_maintenance.append('Kondisi Baik')
            
    df = pd.DataFrame({
        'ID_Armada': armada_ids,
        'Usia_Tahun': usia_list,
        'Odometer_KM': odometer_list,
        'Jenis_Mesin': jenis_mesin,
        'Beban_Muatan': beban_muatan,
        'Riwayat_Servis': riwayat_servis,
        'Status_Maintenance': status_maintenance
    })
    
    # Kategori / Diskretisasi untuk Algoritma C4.5
    df_encoded = df.copy()
    
    # Categorize Usia
    df_encoded['Usia_Kategori'] = pd.cut(
        df['Usia_Tahun'], 
        bins=[-1, 3, 7, 100], 
        labels=['Baru (<3 thn)', 'Sedang (3-7 thn)', 'Lama (>7 thn)']
    )
    
    # Categorize Odometer
    df_encoded['Odometer_Kategori'] = pd.cut(
        df['Odometer_KM'], 
        bins=[-1, 100000, 220000, 1000000], 
        labels=['Rendah (<100k)', 'Sedang (100k-220k)', 'Tinggi (>220k)']
    )
    
    return df, df_encoded

df_raw, df_processed = load_and_preprocess_data()

# Mapping Fitur untuk Scikit-Learn DecisionTree (Representasi C4.5)
feature_cols = ['Usia_Kategori', 'Odometer_Kategori', 'Jenis_Mesin', 'Beban_Muatan', 'Riwayat_Servis']
X = pd.get_dummies(df_processed[feature_cols], drop_first=False)
y = df_processed['Status_Maintenance']

# Fit Decision Tree Model (Criterion Entropy / Gain Ratio)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X, y)

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/gas-truck.png", width=70)
st.sidebar.title("SIMTANK IT Dumai")
st.sidebar.caption("PT Elnusa Petrofin - Riau")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Menu Utama:",
    [
        "📊 Dashboard Ringkasan", 
        "🔍 Form Prediksi Armada", 
        "🌳 Visualisasi Pohon C4.5", 
        "📈 Evaluasi Model (Confusion Matrix)",
        "📋 Data Armada & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **SIMTANK** v1.0\nSistem Informasi Prediksi Maintenance Armada Tangki berbasis Streamlit & C4.5 Decision Tree.")

# ---------------------------------------------------------
# 4. HALAMAN 1: DASHBOARD RINGKASAN
# ---------------------------------------------------------
if menu == "📊 Dashboard Ringkasan":
    st.markdown("<p class='main-title'>📊 Dashboard Status Armada Tangki IT Dumai</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Monitoring Distribusi Kondisi Maintenance 130 Unit Armada Tangki</p>", unsafe_allow_html=True)
    
    # Card Ringkasan
    c1, c2, c3, c4 = st.columns(4)
    total_unit = len(df_raw)
    segera = len(df_raw[df_raw['Status_Maintenance'] == 'Maintenance Segera'])
    rutin = len(df_raw[df_raw['Status_Maintenance'] == 'Maintenance Rutin'])
    baik = len(df_raw[df_raw['Status_Maintenance'] == 'Kondisi Baik'])
    
    c1.metric("Total Armada", f"{total_unit} Unit", "IT Dumai")
    c2.metric("Maintenance Segera 🚨", f"{segera} Unit", f"{(segera/total_unit)*100:.1f}%", delta_color="inverse")
    c3.metric("Maintenance Rutin 🛠️", f"{rutin} Unit", f"{(rutin/total_unit)*100:.1f}%", delta_color="off")
    c4.metric("Kondisi Baik ✅", f"{baik} Unit", f"{(baik/total_unit)*100:.1f}%", delta_color="normal")
    
    st.markdown("---")
    
    # Grafik Visualisasi
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Distribusi Status Maintenance")
        fig_pie = px.pie(
            df_raw, 
            names='Status_Maintenance', 
            color='Status_Maintenance',
            color_discrete_map={
                'Maintenance Segera': '#EF4444',
                'Maintenance Rutin': '#F59E0B',
                'Kondisi Baik': '#10B981'
            },
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Status Berdasarkan Riwayat Servis")
        fig_bar = px.histogram(
            df_raw, 
            x='Riwayat_Servis', 
            color='Status_Maintenance',
            barmode='group',
            color_discrete_map={
                'Maintenance Segera': '#EF4444',
                'Maintenance Rutin': '#F59E0B',
                'Kondisi Baik': '#10B981'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# 5. HALAMAN 2: FORM PREDIKSI ARMADA
# ---------------------------------------------------------
elif menu == "🔍 Form Prediksi Armada":
    st.markdown("<p class='main-title'>🔍 Simulasi Prediksi Maintenance C4.5</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Masukkan parameter operasional armada tangki untuk mendapatkan rekomendasi sistem</p>", unsafe_allow_html=True)
    
    with st.form("form_prediksi_c45"):
        col1, col2 = st.columns(2)
        
        with col1:
            nopol = st.text_input("Nomor Polisi / ID Tangki", value="B 8123 SPU")
            usia = st.number_input("Usia Kendaraan (Tahun)", min_value=1, max_value=20, value=5)
            odometer = st.number_input("Odometer / Jarak Tempuh (KM)", min_value=0, max_value=500000, value=145000)
            
        with col2:
            jenis_mesin = st.selectbox("Jenis Mesin", ["Diesel Euro 4", "Diesel Euro 2"])
            beban = st.selectbox("Kapasitas Beban Muatan", ["16 KL", "24 KL", "32 KL"])
            riwayat = st.selectbox("Riwayat Servis 3 Bulan Terakhir", ["Teratur", "Jarang", "Tidak Pernah"])
            
        submitted = st.form_submit_button("🚀 Proses Klasifikasi C4.5", use_container_width=True)
        
    if submitted:
        # Diskretisasi Input
        usia_kat = 'Baru (<3 thn)' if usia <= 3 else ('Sedang (3-7 thn)' if usia <= 7 else 'Lama (>7 thn)')
        odo_kat = 'Rendah (<100k)' if odometer <= 100000 else ('Sedang (100k-220k)' if odometer <= 220000 else 'Tinggi (>220k)')
        
        # Format dummy input ke skema model X
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        for col in X.columns:
            if f"Usia_Kategori_{usia_kat}" == col: input_data[col] = 1
            if f"Odometer_Kategori_{odo_kat}" == col: input_data[col] = 1
            if f"Jenis_Mesin_{jenis_mesin}" == col: input_data[col] = 1
            if f"Beban_Muatan_{beban}" == col: input_data[col] = 1
            if f"Riwayat_Servis_{riwayat}" == col: input_data[col] = 1
            
        pred_res = clf.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("📌 Hasil Prediksi & Rekomendasi")
        
        if pred_res == 'Maintenance Segera':
            st.error(f"🚨 **STATUS: MAINTENANCE SEGERA** (Armada: {nopol})")
            st.warning("**Rekomendasi Tindakan:** Lakukan penarikan unit dari jalur operasional. Jadwalkan pemeriksaan sistem pengereman, overhaul mesin, serta cek fungsi tangki di workshop IT Dumai.")
        elif pred_res == 'Maintenance Rutin':
            st.warning(f"🛠️ **STATUS: MAINTENANCE RUTIN** (Armada: {nopol})")
            st.info("**Rekomendasi Tindakan:** Lakukan perawatan berkala, ganti oli, dan periksa tekanan ban pada interval jadwal berikutnya.")
        else:
            st.success(f"✅ **STATUS: KONDISI BAIK** (Armada: {nopol})")
            st.info("**Rekomendasi Tindakan:** Armada siap beroperasi untuk penyaluran distribusi BBM wilayah Ring-1 maupun luar kota.")

# ---------------------------------------------------------
# 6. HALAMAN 3: VISUALISASI POHON C4.5
# ---------------------------------------------------------
elif menu == "🌳 Visualisasi Pohon C4.5":
    st.markdown("<p class='main-title'>🌳 Pohon Keputusan (Decision Tree C4.5)</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Visualisasi aturan IF-THEN berbasis perhitungan Entropy dan Gain Ratio</p>", unsafe_allow_html=True)
    
    dot_data = export_graphviz(
        clf, 
        out_file=None, 
        feature_names=X.columns,  
        class_names=clf.classes_,  
        filled=True, 
        rounded=True,  
        special_characters=True
    )
    
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)
    
    st.markdown("---")
    st.subheader("📋 Interpretasi Aturan Keputusan (Rule-Based)")
    st.markdown("""
    * **RULE 1:** IF `Riwayat_Servis` = *Tidak Pernah* OR `Odometer` > *220.000 KM* THEN **Maintenance Segera**.
    * **RULE 2:** IF `Usia_Kendaraan` > *7 Tahun* AND `Riwayat_Servis` = *Jarang* THEN **Maintenance Segera**.
    * **RULE 3:** IF `Usia_Kendaraan` (3-7 Tahun) AND `Riwayat_Servis` = *Teratur* THEN **Maintenance Rutin**.
    * **RULE 4:** IF `Usia_Kendaraan` < *3 Tahun* AND `Riwayat_Servis` = *Teratur* THEN **Kondisi Baik**.
    """)

# ---------------------------------------------------------
# 7. HALAMAN 4: EVALUASI MODEL
# ---------------------------------------------------------
elif menu == "📈 Evaluasi Model (Confusion Matrix)":
    st.markdown("<p class='main-title'>📈 Evaluasi Kinerja Algoritma C4.5</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Pengujian Kinerja Model Menggunakan Confusion Matrix & 10-Fold Cross Validation</p>", unsafe_allow_html=True)
    
    # 10-Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model_fold = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi Rata-rata", f"{np.mean(acc_scores)*100:.2f}%")
    m2.metric("Presisi Rata-rata", f"{np.mean(prec_scores)*100:.2f}%")
    m3.metric("Recall Rata-rata", f"{np.mean(rec_scores)*100:.2f}%")
    m4.metric("F1-Score Rata-rata", f"{np.mean(f1_scores)*100:.2f}%")
    
    st.markdown("---")
    
    # Heatmap Confusion Matrix
    st.subheader("Confusion Matrix (Seluruh Dataset 130 Unit)")
    y_all_pred = clf.predict(X)
    cm = confusion_matrix(y, y_all_pred, labels=clf.classes_)
    
    fig_cm = px.imshow(
        cm,
        x=clf.classes_,
        y=clf.classes_,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Prediksi Model", y="Kondisi Aktual", color="Jumlah Armada")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# 8. HALAMAN 5: DATA ARMADA & EXPORT
# ---------------------------------------------------------
elif menu == "📋 Data Armada & Export":
    st.markdown("<p class='main-title'>📋 Tabel Master Data Armada Tangki</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Data Historis Operasional & Hasil Klasifikasi Maintenance</p>", unsafe_allow_html=True)
    
    # Filter Data
    filter_status = st.multiselect(
        "Filter Berdasarkan Status Maintenance:",
        options=df_raw['Status_Maintenance'].unique(),
        default=df_raw['Status_Maintenance'].unique()
    )
    
    df_filtered = df_raw[df_raw['Status_Maintenance'].isin(filter_status)]
    st.dataframe(df_filtered, use_container_width=True)
    
    # Export CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name='Laporan_Prediksi_Maintenance_SIMTANK.csv',
        mime='text/csv',
    )
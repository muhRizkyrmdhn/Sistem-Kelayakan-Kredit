import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Penilaian Kelayakan Kredit", page_icon="💳", layout="wide")

NUM = ["usia", "penghasilan_juta", "tanggungan", "lama_kerja_tahun",
       "pinjaman_juta", "tenor_bulan", "rasio_cicilan"]
CAT = ["riwayat_kredit"]
MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Naive Bayes": lambda: GaussianNB(),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
}


@st.cache_data
def load_data():
    return pd.read_csv("dataset_kredit.csv")


@st.cache_resource
def train_all(df):
    X, y = df[NUM + CAT], df["layak"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    out = {}
    for name, make in MODELS.items():
        pipe = Pipeline([("pre", pre), ("clf", make())]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[name] = {
            "pipe": pipe,
            "acc": accuracy_score(y_te, pred),
            "prec": precision_score(y_te, pred),
            "rec": recall_score(y_te, pred),
            "auc": roc_auc_score(y_te, proba),
            "cm": confusion_matrix(y_te, pred),
        }
    return out


df = load_data()
models = train_all(df)

st.title("💳 Sistem Penilaian Kelayakan Kredit")
st.caption("Prediksi peminjaman berbasis probabilitas · Pengambilan keputusan algoritmik")

with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Model AI", list(MODELS), help="Algoritma machine learning konvensional")
    st.subheader("Kebijakan keputusan")
    t_setuju = st.slider("Ambang setuju (P ≥)", 0.50, 0.95, 0.70, 0.05)
    t_tolak = st.slider("Ambang tolak (P <)", 0.05, 0.50, 0.40, 0.05)
    st.subheader("Parameter ekonomi")
    bunga = st.number_input("Pendapatan bunga jika lancar (% pokok)", 1.0, 50.0, 12.0, 0.5)
    lgd = st.number_input("Kerugian jika gagal bayar (% pokok)", 10.0, 100.0, 60.0, 5.0)

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "🗂️ Dataset"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data pemohon")
        usia = st.number_input("Usia (tahun)", 21, 65, 30)
        penghasilan = st.number_input("Penghasilan bulanan (juta Rp)", 1.0, 100.0, 8.0, 0.5)
        tanggungan = st.number_input("Jumlah tanggungan", 0, 8, 1)
        lama_kerja = st.number_input("Lama bekerja (tahun)", 0, 40, 4)
        riwayat = st.selectbox("Riwayat kredit", ["Baik", "Cukup", "Buruk"])
    with c2:
        st.subheader("Pengajuan pinjaman")
        pinjaman = st.number_input("Jumlah pinjaman (juta Rp)", 1.0, 500.0, 40.0, 1.0)
        tenor = st.selectbox("Tenor (bulan)", [6, 12, 24, 36, 48], index=2)
        cicilan = pinjaman / tenor
        rasio = cicilan / penghasilan
        st.metric("Cicilan per bulan", f"Rp {cicilan:,.2f} juta")
        st.metric("Rasio cicilan terhadap penghasilan", f"{rasio:.1%}")

    if st.button("Nilai Kelayakan", type="primary", width='stretch'):
        row = pd.DataFrame([{
            "usia": usia, "penghasilan_juta": penghasilan, "tanggungan": tanggungan,
            "lama_kerja_tahun": lama_kerja, "pinjaman_juta": pinjaman, "tenor_bulan": tenor,
            "rasio_cicilan": round(rasio, 3), "riwayat_kredit": riwayat,
        }])
        p = float(models[model_name]["pipe"].predict_proba(row)[0, 1])
        ev = p * pinjaman * bunga / 100 - (1 - p) * pinjaman * lgd / 100

        if p >= t_setuju:
            st.success(f"✅ DISETUJUI · probabilitas layak {p:.1%}")
        elif p < t_tolak:
            st.error(f"❌ DITOLAK · probabilitas layak {p:.1%}")
        else:
            st.warning(f"🔎 TINJAU MANUAL · probabilitas layak {p:.1%}")

        m1, m2, m3 = st.columns(3)
        m1.metric("P(layak)", f"{p:.1%}")
        m2.metric("P(gagal bayar)", f"{1 - p:.1%}")
        m3.metric("Expected value", f"Rp {ev:,.2f} juta",
                  delta="menguntungkan" if ev > 0 else "merugikan",
                  delta_color="normal" if ev > 0 else "inverse")
        st.progress(p)
        st.caption(f"Expected value = P(layak) × bunga − P(gagal) × kerugian, dihitung dari "
                   f"pinjaman Rp {pinjaman:,.1f} juta.")

        if model_name == "Logistic Regression":
            pipe = models[model_name]["pipe"]
            names = NUM + [f"riwayat_{c}" for c in pipe.named_steps["pre"].named_transformers_["cat"].categories_[0]]
            contrib = pipe.named_steps["pre"].transform(row)[0] * pipe.named_steps["clf"].coef_[0]
            st.subheader("Faktor yang memengaruhi keputusan")
            st.bar_chart(pd.Series(contrib, index=names).sort_values())
            st.caption("Positif menaikkan peluang layak, negatif menurunkannya.")

with tab2:
    st.subheader("Perbandingan model (data uji 25%)")
    st.dataframe(pd.DataFrame({
        n: {"Akurasi": m["acc"], "Presisi": m["prec"], "Recall": m["rec"], "ROC-AUC": m["auc"]}
        for n, m in models.items()
    }).T.style.format("{:.3f}"), width='stretch')
    cm = models[model_name]["cm"]
    st.write(f"**Confusion matrix · {model_name}**")
    st.dataframe(pd.DataFrame(cm, index=["Aktual: tidak layak", "Aktual: layak"],
                              columns=["Prediksi: tidak layak", "Prediksi: layak"]))

with tab3:
    st.subheader("Dataset sintetis")
    st.write(f"{len(df)} data · proporsi layak {df.layak.mean():.1%}. "
             "Data dibuat otomatis untuk simulasi, bukan data nasabah nyata.")
    st.dataframe(df, width='stretch', height=350)
    st.bar_chart(df.groupby("riwayat_kredit")["layak"].mean().rename("proporsi layak"))

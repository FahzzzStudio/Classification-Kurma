import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title = "Classification Kurma",
    page_icon = "🫘"
)

model = joblib.load("best_model.joblib")

st.title("🫘 Klasifikasi Jenis Kurma")
st.markdown("Aplikasi machine learning untuk klasifikasi jenis kurma berdasarkan")

tab1, tab2 = st.tabs(["Klasifikasi", "Informasi"])

with tab1:
    berat = st.slider("Berat Kurma",5,20,12)
    kadar_gula = st.slider("Kadar Gula", 50, 100, 75)
    warna = st.pills("Warna Kurma", ["kecoklatan","kehitaman"], default="kecoklatan")
    tekstur = st.pills("Tekstur Kurma", ["halus","agak kasar","kasar"], default="halus")
    tingkat_kematangan = st.pills("Tingkat Kematangan", ["setengah matang", "matang", "terlalu matang"], default="setengah matang")

    if st.button("Prediksi", type="primary"):
        data_baru = pd.DataFrame([[berat, kadar_gula, warna, tekstur, tingkat_kematangan]],
                                 columns=["berat","kadar_gula","warna","tekstur","tingkat_kematangan"])
        prediksi = model.predict(data_baru)[0]
        presentase = max(model.predict_proba(data_baru)[0])
        if prediksi == "ajwa":
            st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
        elif prediksi == "sukkari":
            st.warning(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
        else :
            st.error(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
        
        st.markdown("**Hasil Prediksi :**")
        a,b = st.columns(2)
        a.metric("Prediksi", str(prediksi), delta=None, border=True)
        b.metric("Prediksi", round(presentase*100,2), delta=None, border=True)

        proba_df = pd.DataFrame(
            model.predict_proba(data_baru),
            columns=model.classes_
        )
        st.markdown("**Probabilitas Tiap Jenis Kurma**")
        st.bar_chart(proba_df.T)

        st.balloons()

with tab2:
    st.image("kurma.jpg")

    st.markdown("""
    ### Literatur Kurma
1. **Ajwa (Madinah)**
   * Berat: kecil–sedang (7–10 gram/butir)
   * Warna: hitam keunguan (sering disebut “kehitaman”)
   * Tekstur: halus & lembut
   * Kadar gula: relatif sedang (60–70% brix)
   * Rasa: manis ringan, tidak terlalu pekat
2. **Sukkari (Arab Saudi)**
   * Berat: sedang–besar (9–15 gram)
   * Warna: coklat muda–keemasan
   * Tekstur: agak kasar, cenderung renyah (bila muda)
   * Kadar gula: tinggi (70–80% brix)
   * Rasa: manis legit
3. **Medjool (Yordania, Maroko, AS)**
   * Berat: besar (12–20 gram)
   * Warna: coklat tua gelap
   * Tekstur: agak kasar–kasar
   * Kadar gula: sangat tinggi (75–85% brix)
   * Rasa: sangat manis, legit pekat
""")

st.divider()
st.caption("Dibuat dengan :heart: **oleh Fahmi Dwi Santoso**")
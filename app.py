import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

st.set_page_config(page_title="Gelişmiş Tarım Zekası", layout="wide")
st.title("🌾 Kapsamlı Tarım Yapay Zeka Sistemi")
st.write("Bu sistem, sınıflandırma ve regresyon modellerini bir arada sunar.")

tab1, tab2 = st.tabs(["1. Sınıflandırma Modeli (Ne Ekeyim?)", "2. Regresyon Modeli (Ne Kadar Alırım?)"])

with tab1:
    st.header("🌱 Ürün Tavsiye Sistemi (Sınıflandırma)")
    
    @st.cache_resource
    def siniflandirma_modeli_egit():
        df = pd.read_csv("orijinal_veri.csv")
        X = df.drop('label', axis=1)
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model_sinif = siniflandirma_modeli_egit()

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Azot (N)", value=90)
        p = st.number_input("Fosfor (P)", value=42)
        k = st.number_input("Potasyum (K)", value=43)
    with col2:
        temp = st.number_input("Sıcaklık (°C)", value=20.0)
        hum = st.number_input("Nem (%)", value=82.0)
        ph = st.number_input("pH Değeri", value=6.5)
    with col3:
        rain = st.number_input("Yağış (mm)", value=200.0)

    if st.button("Hangi Ürünü Ekmeliyim?"):
        girdi = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], 
                             columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        tahmin = model_sinif.predict(girdi)[0]
        st.success(f"🌟 Bu tarlaya ekilmesi önerilen ürün: **{tahmin.upper()}**")

with tab2:
    st.header("📊 Ekin Verimi Tahmin Sistemi (Regresyon)")
    st.write("Bu model; Sıcaklık, Yağış, İlaçlama ve Rekolte verilerini 4 ayrı veri setinden birleştirerek öğrenmiştir.")
    
    @st.cache_resource
    def regresyon_modeli_egit():
        df_yield = pd.read_csv('yield.csv')
        df_temp = pd.read_csv('temp.csv')
        df_rain = pd.read_csv('rainfall.csv')
        df_pest = pd.read_csv('pesticides.csv')

        df_yield.columns = df_yield.columns.str.strip()
        df_temp.columns = df_temp.columns.str.strip()
        df_rain.columns = df_rain.columns.str.strip()
        df_pest.columns = df_pest.columns.str.strip()

        df_temp.rename(columns={'country': 'Area', 'year': 'Year'}, inplace=True)
        
        if 'Item' in df_pest.columns:
            df_pest = df_pest.drop('Item', axis=1)
            
        if 'Value' in df_pest.columns:
            df_pest.rename(columns={'Value': 'Pesticides'}, inplace=True)
        elif 'pesticides_tonnes' in df_pest.columns:
            df_pest.rename(columns={'pesticides_tonnes': 'Pesticides'}, inplace=True)

        if 'Value' in df_yield.columns:
            df_yield.rename(columns={'Value': 'Yield'}, inplace=True)
        elif 'hg/ha_yield' in df_yield.columns:
            df_yield.rename(columns={'hg/ha_yield': 'Yield'}, inplace=True)

        df_merged = pd.merge(df_yield, df_temp, on=['Area', 'Year'], how='inner')
        df_merged = pd.merge(df_merged, df_rain, on=['Area', 'Year'], how='inner')
        df_merged = pd.merge(df_merged, df_pest, on=['Area', 'Year'], how='inner')
        
        df_merged.rename(columns={
            'average_rain_fall_mm_per_year': 'Rainfall', 
            'avg_temp': 'Temperature'
        }, inplace=True)
        
        # --- HATALI VERİ TEMİZLEME ---
        sayisal_sutunlar = ['Year', 'Rainfall', 'Pesticides', 'Temperature', 'Yield']
        for sutun in sayisal_sutunlar:
            df_merged[sutun] = pd.to_numeric(df_merged[sutun], errors='coerce')
        df_merged.dropna(inplace=True)
        # -----------------------------

        X = df_merged[['Area', 'Item', 'Year', 'Rainfall', 'Pesticides', 'Temperature']]
        y = df_merged['Yield']
        
        donusturucu = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])], remainder='passthrough')
        model = Pipeline(steps=[('preprocessor', donusturucu), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        model.fit(X, y)
        return model
    
    if os.path.exists("yield.csv") and os.path.exists("temp.csv") and os.path.exists("pesticides.csv") and os.path.exists("rainfall.csv"):
        model_reg = regresyon_modeli_egit()
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            area = st.text_input("Bölge / Ülke (Örn: Turkey, India)", value="Turkey")
            item = st.text_input("Ekin Türü (Örn: Wheat, Rice)", value="Wheat")
            year = st.number_input("Yıl", min_value=1990, max_value=2050, value=2024)
        with r_col2:
            rainfall = st.number_input("Yıllık Yağış (mm)", value=500.0)
            pesticides = st.number_input("Tarım İlacı (Ton)", value=150.0)
            temperature = st.number_input("Ortalama Sıcaklık (°C)", value=15.0)

        if st.button("Verimi Tahmin Et (Ton/Hektar)"):
            yeni_girdi = pd.DataFrame({'Area': [area], 'Item': [item], 'Year': [year], 'Rainfall': [rainfall], 'Pesticides': [pesticides], 'Temperature': [temperature]})
            tahmin_verim = model_reg.predict(yeni_girdi)[0]
            st.success(f"📈 Tahmini Verim: **{tahmin_verim:,.2f} hg/ha**")
    else:
        st.error("⚠️ HATA: 'yield.csv', 'temp.csv', 'rainfall.csv' ve 'pesticides.csv' dosyalarından biri eksik.")
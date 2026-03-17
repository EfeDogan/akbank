import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=== BÖLÜM 1: SINIFLANDIRMA (NE EKELİM?) ===")
print("1. Orijinal veri okunuyor ve ikiye bölünüyor...")
ham_veri = pd.read_csv("orijinal_veri.csv")
ham_veri['Tarla_ID'] = range(1, len(ham_veri) + 1)

toprak_kismi = ham_veri[['Tarla_ID', 'N', 'P', 'K', 'ph']]
iklim_kismi = ham_veri[['Tarla_ID', 'temperature', 'humidity', 'rainfall', 'label']]
toprak_kismi.to_csv("toprak_verileri.csv", index=False)
iklim_kismi.to_csv("iklim_verileri.csv", index=False)
print("-> 'toprak_verileri.csv' ve 'iklim_verileri.csv' oluşturuldu.")

print("2. Veriler ortak ID üzerinden birleştiriliyor...")
df_toprak = pd.read_csv("toprak_verileri.csv")
df_iklim = pd.read_csv("iklim_verileri.csv")
birlesik_veri_1 = pd.merge(df_toprak, df_iklim, on="Tarla_ID")

print("3. Sınıflandırma Modeli Eğitiliyor...")
X1 = birlesik_veri_1.drop(['Tarla_ID', 'label'], axis=1)
y1 = birlesik_veri_1['label']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train1, y_train1)
# Modelin başarısını test ediyoruz
# Classification Report, Accuracy Score
classification_report = model1.score(X_test1, y_test1)
print(f"-> Sınıflandırma Modeli Başarı Oranı: {classification_report:.2f}")
print("-> Sınıflandırma Modeli Hazır!\n")

print("=== BÖLÜM 2: REGRESYON (NE KADAR ÜRÜN ALIRIZ?) ===")
print("1. Dış veri setleri yükleniyor ve temizleniyor...")
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

print("2. Veriler birleştiriliyor...")
df_merged = pd.merge(df_yield, df_temp, on=['Area', 'Year'], how='inner')
df_merged = pd.merge(df_merged, df_rain, on=['Area', 'Year'], how='inner')
df_merged = pd.merge(df_merged, df_pest, on=['Area', 'Year'], how='inner')

df_merged.rename(columns={
    'average_rain_fall_mm_per_year': 'Rainfall', 
    'avg_temp': 'Temperature'
}, inplace=True)

# --- HATALI VERİ (..) TEMİZLEME İŞLEMİ EKLENDİ ---
sayisal_sutunlar = ['Year', 'Rainfall', 'Pesticides', 'Temperature', 'Yield']
for sutun in sayisal_sutunlar:
    # Sayı olmayan her şeyi (mesela '..') NaN (boşluk) yapar
    df_merged[sutun] = pd.to_numeric(df_merged[sutun], errors='coerce')

# İçinde NaN (boş) olan satırları komple siliyoruz ki model çökmesin
df_merged.dropna(inplace=True)
# --------------------------------------------------

print("3. Regresyon Modeli Eğitiliyor...")
X2 = df_merged[['Area', 'Item', 'Year', 'Rainfall', 'Pesticides', 'Temperature']]
y2 = df_merged['Yield']

donusturucu = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])], remainder='passthrough')

model2 = Pipeline(steps=[
    ('preprocessor', donusturucu),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2.fit(X_train2, y_train2)
basari = r2_score(y_test2, model2.predict(X_test2))
print(f"-> Regresyon Modeli Hazır! Başarı Oranı (R2 Score): {basari:.2f}")
print("Sistem başarıyla test edildi.")
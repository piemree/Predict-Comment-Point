import pandas as pd
import zipfile
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

# Archive.zip dosyasını açma
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Reviews.csv dosyasını yükleme
df = pd.read_csv('data/Reviews.csv')

# Gereksiz karakterleri temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['cleaned_text'] = df['Text'].apply(clean_text)

# Özellik ve hedef değişkenlerini ayırma
X = df['cleaned_text']
y = df['Score']

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sınıf dengesizliğini düzeltme
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train.values.reshape(-1, 1), y_train)
X_train_resampled = X_train_resampled.flatten()

# TF-IDF ve Naive Bayes Pipeline'ı oluşturma
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Modeli eğitme
model.fit(X_train_resampled, y_train_resampled)

# Modelin doğruluğunu kontrol etme
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Modeli kaydetme
joblib.dump(model, 'review_model.joblib')


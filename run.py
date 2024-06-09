import re
import string
import joblib

# Gereksiz karakterleri temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Kaydedilmiş modeli yükleme
model = joblib.load('review_model.joblib')

# Yeni metin analizi fonksiyonu
def predict_review(text):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Örnek yeni metinler
# Örnek yeni metinler
new_texts = [
    "This product is excellent and works as expected!",
    "The product was terrible and I am very disappointed.",
    "It is okay, but could be better.",
    "I love this product! Highly recommended.",
    "Not what I expected. Quality is poor.",
    "The product is not bad, but I expected more.",
    "I would not recommend this product to anyone."
]

for new_text in new_texts:
    predicted_score = predict_review(new_text)
    print(f"The predicted score for the review '{new_text}' is: {predicted_score}")
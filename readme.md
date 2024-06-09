# Makine Öğrenmesi ile Yorum Puan Tahmini

Bu proje, kullanıcının yaptığı yorumdan, yorumun puanını tahmin etmek amacıyla makine öğrenmesi tekniklerini kullanır. Veriler, kullanıcıların ürünlere verdikleri puanları ve yorumlarını içerir. Proje, bu verileri kullanarak bir metin sınıflandırma modeli eğitir ve yeni yorumların puanlarını tahmin eder.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız var:

- pandas
- scikit-learn
- joblib
- imbalanced-learn
- joblib

verisetini aşşağıdaki linkden indirebilirsiniz:

[https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews]()

Gerekli kütüphaneleri yüklemek için:

```bash
pip install pandas scikit-learn joblib imbalanced-learn
```

Modelimizi kaydetmek için (review_model.joblib adında model dosyası oluşturur):

```
python create_model.py
```

Projeyi çalıştırmak için:

```
python run.py
```

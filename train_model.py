import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Veri setini oku
df = pd.read_csv("../data/students.csv")

# Girdi ve çıktı değişkenleri
X = df[[
    "study_hours",
    "attendance",
    "previous_grade",
    "assignments_score"
]]

y = df["pass"]

# Eğitim ve test verisi ayır
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model oluştur
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Eğit
model.fit(X_train, y_train)

# Tahmin yap
predictions = model.predict(X_test)

# Başarı oranı
accuracy = accuracy_score(y_test, predictions)

print("Model Doğruluk Oranı:", round(accuracy * 100, 2), "%")

# Modeli kaydet
joblib.dump(model, "student_success_model.pkl")

print("Model başarıyla kaydedildi.")
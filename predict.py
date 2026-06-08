import joblib
import pandas as pd

model = joblib.load("student_success_model.pkl")

print("=== Öğrenci Başarı Tahmin Sistemi ===")

study_hours = float(input("Haftalık çalışma saati: "))
attendance = float(input("Devam oranı (%): "))
previous_grade = float(input("Önceki not ortalaması: "))
assignments_score = float(input("Ödev puanı: "))

new_student = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_grade": previous_grade,
    "assignments_score": assignments_score
}])

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nTahmin Sonucu: Öğrenci dersi GEÇER.")
else:
    print("\nTahmin Sonucu: Öğrenci dersten KALIR.")
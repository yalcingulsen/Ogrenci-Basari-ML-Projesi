import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/students.csv")

print("Toplam öğrenci sayısı:", len(df))
print("Geçen öğrenci sayısı:", df["pass"].sum())
print("Kalan öğrenci sayısı:", len(df) - df["pass"].sum())

print("\nOrtalama Değerler:")
print(df.groupby("pass")[[
    "study_hours",
    "attendance",
    "previous_grade",
    "assignments_score"
]].mean())

plt.figure()
df.groupby("pass")["study_hours"].mean().plot(kind="bar")
plt.title("Geçme Durumuna Göre Ortalama Çalışma Saati")
plt.xlabel("Geçme Durumu (0: Kaldı, 1: Geçti)")
plt.ylabel("Ortalama Çalışma Saati")
plt.tight_layout()
plt.show()

plt.figure()
df.groupby("pass")["attendance"].mean().plot(kind="bar")
plt.title("Geçme Durumuna Göre Ortalama Devam Oranı")
plt.xlabel("Geçme Durumu (0: Kaldı, 1: Geçti)")
plt.ylabel("Ortalama Devam Oranı")
plt.tight_layout()
plt.show()
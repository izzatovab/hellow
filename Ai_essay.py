#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sun'iy intellekt yechimi: Chiziqli regressiya modeli
Ushbu kod C qismidagi talablarni bajarish uchun yozilgan
"""

# Kerakli kutubxonalarni import qilish
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_iris
import warnings

# Ogohlantirishlarni o'chirish
warnings.filterwarnings('ignore')

# Matplotlib sozlamalari
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def ma_lumot_yuklash():
    """
    Ma'lumotlarni yuklash va tayyorlash funksiyasi

    Returns:
        X (DataFrame): Mustaqil o'zgaruvchilar
        y (Series): Bog'liq o'zgaruvchi
        feature_names (list): Xususiyatlar nomlari
    """
    try:
        # California uy-joy narxlari ma'lumotlar to'plamini yuklash
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='MEDV')
        feature_names = california.feature_names
        print("California uy-joy narxlari ma'lumotlar to'plami yuklandi.")
    except:
        # Agar dataset mavjud bo'lmasa, o'zimiz sun'iy ma'lumotlar yaratamiz
        print("California ma'lumotlar to'plami mavjud emas. Sun'iy ma'lumotlar yaratilmoqda...")
        np.random.seed(42)
        n_samples = 500

        # Uy maydoni (kv.metr)
        X1 = np.random.normal(100, 30, n_samples)
        # Xonalar soni
        X2 = np.random.randint(1, 6, n_samples)
        # Uy yoshi (yil)
        X3 = np.random.randint(1, 50, n_samples)
        # Shahar markazidan masofa (km)
        X4 = np.random.normal(15, 5, n_samples)

        # Narx (million so'm) - bog'liq o'zgaruvchi
        noise = np.random.normal(0, 50, n_samples)
        y = 200 + 2.5 * X1 + 30 * X2 - 1.5 * X3 - 10 * X4 + noise

        X = pd.DataFrame({
            'MAYDONI': X1,
            'XONALAR': X2,
            'YOSHI': X3,
            'MASOFA': X4
        })
        y = pd.Series(y, name='NARXI')
        feature_names = X.columns.tolist()
        print("Sun'iy uy-joy narxlari ma'lumotlari yaratildi.")

    return X, y, feature_names


def ma_lumotlarni_tayyorlash(X, y, test_size=0.2, random_state=42):
    """
    Ma'lumotlarni o'qitish va test to'plamlariga ajratish

    Args:
        X (DataFrame): Mustaqil o'zgaruvchilar
        y (Series): Bog'liq o'zgaruvchi
        test_size (float): Test to'plami ulushi
        random_state (int): Tasodifiy sonlar generatori uchun seed

    Returns:
        X_train, X_test, y_train, y_test: Ajratilgan ma'lumotlar
    """
    # Ma'lumotlarni o'qitish va test to'plamlariga ajratish
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"O'qitish to'plami hajmi: {X_train.shape[0]} ta namuna")
    print(f"Test to'plami hajmi: {X_test.shape[0]} ta namuna")

    return X_train, X_test, y_train, y_test


def oddiy_chiziqli_regressiya(X_train, X_test, y_train, y_test, feature_idx=0):
    """
    Oddiy chiziqli regressiya (bitta mustaqil o'zgaruvchi bilan)

    Args:
        X_train, X_test (DataFrame): O'qitish va test uchun mustaqil o'zgaruvchilar
        y_train, y_test (Series): O'qitish va test uchun bog'liq o'zgaruvchi
        feature_idx (int): Qaysi xususiyatni ishlatish indeksi

    Returns:
        model (LinearRegression): O'qitilgan model
        metrics (dict): Baholash metrikalari
    """
    # Bitta xususiyatni tanlash
    feature_name = X_train.columns[feature_idx]
    X_train_single = X_train.iloc[:, [feature_idx]]
    X_test_single = X_test.iloc[:, [feature_idx]]

    # Modelni yaratish va o'qitish
    model = LinearRegression()
    model.fit(X_train_single, y_train)

    # Bashorat qilish
    y_pred = model.predict(X_test_single)

    # Modelni baholash
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / np.maximum(1e-10, np.abs(y_test)))) * 100
    }

    # Natijalarni chop etish
    print("\n=== ODDIY CHIZIQLI REGRESSIYA NATIJALARI ===")
    print(f"Tanlangan xususiyat: {feature_name}")
    print(f"Koeffitsient: {model.coef_[0]:.4f}")
    print(f"Ozod had: {model.intercept_:.4f}")
    print(f"Regressiya tenglamasi: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * {feature_name}")

    print("\nMetrikalar:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Natijalarni vizualizatsiya qilish
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_single, y_test, color='blue', alpha=0.5, label='Haqiqiy qiymatlar')
    plt.plot(X_test_single, y_pred, color='red', linewidth=2, label='Bashorat')
    plt.title(f'Oddiy chiziqli regressiya ({feature_name})')
    plt.xlabel(feature_name)
    plt.ylabel(y_train.name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('oddiy_chiziqli_regressiya.png')

    return model, metrics


def kop_ozgaruvchili_chiziqli_regressiya(X_train, X_test, y_train, y_test):
    """
    Ko'p o'zgaruvchili chiziqli regressiya

    Args:
        X_train, X_test (DataFrame): O'qitish va test uchun mustaqil o'zgaruvchilar
        y_train, y_test (Series): O'qitish va test uchun bog'liq o'zgaruvchi

    Returns:
        model (LinearRegression): O'qitilgan model
        metrics (dict): Baholash metrikalari
    """
    # Modelni yaratish va o'qitish
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Bashorat qilish
    y_pred = model.predict(X_test)

    # Modelni baholash
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / np.maximum(1e-10, np.abs(y_test)))) * 100
    }

    # Natijalarni chop etish
    print("\n=== KO'P O'ZGARUVCHILI CHIZIQLI REGRESSIYA NATIJALARI ===")
    print("Koeffitsientlar:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Ozod had: {model.intercept_:.4f}")

    print("\nMetrikalar:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Haqiqiy va bashorat qilingan qiymatlarni taqqoslash
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Haqiqiy va bashorat qilingan qiymatlar taqqoslash")
    plt.xlabel("Haqiqiy qiymatlar")
    plt.ylabel("Bashorat qilingan qiymatlar")
    plt.grid(True, alpha=0.3)
    plt.savefig('kop_ozgaruvchili_regressiya.png')

    return model, metrics


def logistik_regressiya():
    """
    Logistik regressiya uchun kategoriyali ma'lumotlarni bashorat qilish

    Returns:
        model (LogisticRegression): O'qitilgan model
        metrics (dict): Baholash metrikalari
    """
    # Iris ma'lumotlar to'plamini yuklash
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')

    print("\n=== LOGISTIK REGRESSIYA UCHUN MA'LUMOTLAR ===")
    print(f"Ma'lumotlar hajmi: {X.shape[0]} ta namuna, {X.shape[1]} ta xususiyat")
    print(f"Sinflar: {np.unique(y)}")

    # Ma'lumotlarni o'qitish va test to'plamlariga ajratish
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Ma'lumotlarni standartlashtirish
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelni yaratish va o'qitish
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Bashorat qilish
    y_pred = model.predict(X_test_scaled)

    # Modelni baholash
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    # Natijalarni chop etish
    print("\n=== LOGISTIK REGRESSIYA NATIJALARI ===")
    print(f"Aniqlik ko'rsatkichi: {accuracy:.4f}")
    print("\nTasniflash hisoboti:")
    print(report)

    # Natijalarni vizualizatsiya qilish (faqat 2 ta xususiyat uchun)
    plt.figure(figsize=(10, 6))

    # Faqat birinchi 2 ta xususiyatni olish
    X_test_2d = X_test.iloc[:, :2].values

    # Har bir sinf uchun ranglar
    colors = ['blue', 'green', 'red']

    # Har bir sinf uchun nuqtalarni chizish
    for i, color in enumerate(colors):
        # NumPy array bilan ishlash uchun boolean indekslashni to'g'ri ishlatish
        idx = np.where(y_test.values == i)[0]
        plt.scatter(
            X_test_2d[idx, 0],
            X_test_2d[idx, 1],
            c=color,
            label=iris.target_names[i],
            alpha=0.7
        )

    plt.title('Iris ma\'lumotlar to\'plami (2 ta xususiyat)')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('logistik_regressiya.png')

    metrics = {
        'accuracy': accuracy,
        'report': report
    }

    return model, metrics


def haddan_tashqari_moslashtirish_tushuntirish():
    """
    Haddan tashqari moslashtirish va yetarli darajada moslashtirmaslik tushunchalarini tushuntirish
    """
    print("\n=== HADDAN TASHQARI MOSLASHTIRISH VA YETARLI DARAJADA MOSLASHTIRMASLIK ===")
    print("""
Haddan tashqari moslashtirish (Overfitting):
- Bu model o'qitish ma'lumotlariga juda yaxshi moslashib, lekin yangi ma'lumotlarga yomon bashorat berish holati.
- Model o'qitish ma'lumotlaridagi shovqinlarni ham o'rganib oladi.
- O'qitish ma'lumotlarida yuqori aniqlik, test ma'lumotlarida past aniqlik kuzatiladi.
- Sabablari: model juda murakkab, ma'lumotlar kam, o'qitish davri juda uzoq.
- Yechimlar: regularizatsiya, ma'lumotlarni ko'paytirish, modelni soddalashtrish, cross-validation.

Yetarli darajada moslashtirmaslik (Underfitting):
- Bu model o'qitish ma'lumotlariga ham yaxshi moslasha olmaydigan holat.
- Model ma'lumotlardagi muhim bog'liqliklarni o'rgana olmaydi.
- Ham o'qitish, ham test ma'lumotlarida past aniqlik kuzatiladi.
- Sabablari: model juda sodda, xususiyatlar yetarli emas, o'qitish davri juda qisqa.
- Yechimlar: murakkab model tanlash, ko'proq xususiyatlar qo'shish, o'qitish davrini uzaytirish.

Optimal model:
- O'qitish va test ma'lumotlarida o'xshash aniqlik ko'rsatadi.
- Ma'lumotlardagi asosiy bog'liqliklarni o'rganadi, lekin shovqinlarni e'tiborsiz qoldiradi.
- Yangi ma'lumotlarga yaxshi bashorat beradi.
    """)


def main():
    """
    Asosiy funksiya
    """
    print("=== SUN'IY INTELLEKT YECHIMI: REGRESSIYA MODELLARI ===\n")

    # 1. Ma'lumotlarni yuklash va tayyorlash
    X, y, feature_names = ma_lumot_yuklash()

    # Ma'lumotlar haqida umumiy ma'lumot
    print("\nMa'lumotlar haqida umumiy ma'lumot:")
    print(f"Namunalar soni: {X.shape[0]}")
    print(f"Xususiyatlar soni: {X.shape[1]}")
    print(f"Xususiyatlar: {', '.join(feature_names)}")

    # Ma'lumotlarni o'qitish va test to'plamlariga ajratish
    X_train, X_test, y_train, y_test = ma_lumotlarni_tayyorlash(X, y)

    # 2. Oddiy chiziqli regressiya
    simple_model, simple_metrics = oddiy_chiziqli_regressiya(X_train, X_test, y_train, y_test)

    # 3. Ko'p o'zgaruvchili chiziqli regressiya
    multi_model, multi_metrics = kop_ozgaruvchili_chiziqli_regressiya(X_train, X_test, y_train, y_test)

    # 4. Logistik regressiya
    logistic_model, logistic_metrics = logistik_regressiya()

    # 5. Haddan tashqari moslashtirish va yetarli darajada moslashtirmaslik tushunchalarini tushuntirish
    haddan_tashqari_moslashtirish_tushuntirish()

    print("\n=== XULOSA ===")
    print("Barcha modellar muvaffaqiyatli o'qitildi va baholandi.")
    print("Oddiy chiziqli regressiya R² ko'rsatkichi:", simple_metrics['R2'])
    print("Ko'p o'zgaruvchili chiziqli regressiya R² ko'rsatkichi:", multi_metrics['R2'])
    print("Logistik regressiya aniqlik ko'rsatkichi:", logistic_metrics['accuracy'])

    print("\nGrafiklar saqlandi:")
    print("- oddiy_chiziqli_regressiya.png")
    print("- kop_ozgaruvchili_regressiya.png")
    print("- logistik_regressiya.png")


if __name__ == "__main__":
    main()

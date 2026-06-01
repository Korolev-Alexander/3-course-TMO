import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Прогноз оттока клиентов", layout="wide")

st.title("Прогнозирование оттока клиентов телеком-компании")
st.markdown("---")

# Загрузка и предобработка данных
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Удаляем customerID
    df = df.drop('customerID', axis=1)
    
    # Целевая переменная
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # TotalCharges -> число, пропуски заполняем MonthlyCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Бинарные признаки (Yes/No -> 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Gender
    df['gender'] = (df['gender'] == 'Male').astype(int)
    
    # MultipleLines (создаем бинарный признак)
    df['HasMultipleLines'] = (df['MultipleLines'] == 'Yes').astype(int)
    
    # Contract (порядковый)
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract'] = df['Contract'].map(contract_map)
    
    # InternetService (порядковый)
    internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    df['InternetService'] = df['InternetService'].map(internet_map)
    
    # OnlineSecurity и другие интернет-услуги
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_services:
        df[col] = df[col].replace('No internet service', 'No')
        df[col] = (df[col] == 'Yes').astype(int)
    
    # PaymentMethod
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Отбираем финальные признаки
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'HasMultipleLines', 'InternetService', 'Contract',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'IsElectronicCheck'
    ]
    
    X = df[features]
    y = df['Churn']
    
    return X, y

X, y = load_and_preprocess_data()

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Боковая панель с гиперпараметрами
st.sidebar.header("Настройки модели")
C = st.sidebar.slider("C (обратная регуляризация)", 0.01, 2.0, 0.1, step=0.01,
                       help="Чем меньше C, тем сильнее регуляризация")
max_iter = st.sidebar.select_slider("Максимум итераций", [100, 200, 500, 1000, 2000], value=1000)

# Обучение модели с пайплайном
@st.cache_resource
def train_model(C, max_iter):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LogisticRegression(C=C, max_iter=max_iter, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

@st.cache_resource
def get_test_accuracy(_pipeline):
    return _pipeline.score(X_test, y_test)

model = train_model(C, max_iter)
score = get_test_accuracy(model)

st.sidebar.metric("Точность модели", f"{score:.3f}")
st.sidebar.markdown("---")
st.sidebar.info("При изменении параметров модель переобучается автоматически")

# Основная форма ввода
st.subheader("Введите данные о клиенте")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Пол", ["Женский", "Мужской"])
    senior = st.selectbox("Пенсионер", ["Нет", "Да"])
    partner = st.selectbox("Наличие партнёра", ["Нет", "Да"])
    dependents = st.selectbox("Наличие иждивенцев", ["Нет", "Да"])
    tenure = st.slider("Стаж (месяцы)", 0, 72, 12)

with col2:
    phone = st.selectbox("Телефонная линия", ["Нет", "Да"])
    multiple_lines = st.selectbox("Несколько линий", ["Нет", "Да"])
    internet = st.selectbox("Интернет", ["DSL", "Волоконная оптика", "Нет"])
    contract = st.selectbox("Тип контракта", ["Помесячный", "На год", "На два года"])
    paperless = st.selectbox("Безбумажный счёт", ["Нет", "Да"])

with col3:
    monthly = st.number_input("Ежемесячная плата ($)", 18.0, 120.0, 70.0)
    total = st.number_input("Общая сумма платежей ($)", 0.0, 9000.0, 500.0)
    online_security = st.selectbox("Онлайн-безопасность", ["Нет", "Да"])
    online_backup = st.selectbox("Онлайн-резервирование", ["Нет", "Да"])
    device_protection = st.selectbox("Защита устройств", ["Нет", "Да"])
    tech_support = st.selectbox("Техподдержка", ["Нет", "Да"])
    streaming_tv = st.selectbox("Стриминг TV", ["Нет", "Да"])
    streaming_movies = st.selectbox("Стриминг кино", ["Нет", "Да"])
    payment_method = st.selectbox("Способ оплаты", ["Электронный чек", "Другой"])

# Преобразование в числовой формат
gender_val = 1 if gender == "Мужской" else 0
senior_val = 1 if senior == "Да" else 0
partner_val = 1 if partner == "Да" else 0
dependents_val = 1 if dependents == "Да" else 0
phone_val = 1 if phone == "Да" else 0
multiple_val = 1 if multiple_lines == "Да" else 0
paperless_val = 1 if paperless == "Да" else 0

internet_map = {"DSL": 0, "Волоконная оптика": 1, "Нет": 2}
internet_val = internet_map[internet]

contract_map = {"Помесячный": 0, "На год": 1, "На два года": 2}
contract_val = contract_map[contract]

online_security_val = 1 if online_security == "Да" else 0
online_backup_val = 1 if online_backup == "Да" else 0
device_protection_val = 1 if device_protection == "Да" else 0
tech_support_val = 1 if tech_support == "Да" else 0
streaming_tv_val = 1 if streaming_tv == "Да" else 0
streaming_movies_val = 1 if streaming_movies == "Да" else 0
electronic_check_val = 1 if payment_method == "Электронный чек" else 0

# Формирование вектора признаков
input_data = [[
    gender_val, senior_val, partner_val, dependents_val, tenure,
    phone_val, multiple_val, internet_val, contract_val,
    paperless_val, monthly, total,
    online_security_val, online_backup_val, device_protection_val,
    tech_support_val, streaming_tv_val, streaming_movies_val,
    electronic_check_val
]]

# Предсказание
try:
    proba = model.predict_proba(input_data)[0, 1]
    prediction = 1 if proba > 0.5 else 0
    
    st.markdown("---")
    st.subheader("Результат предсказания")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 1:
            st.error("⚠️ ВЫСОКИЙ РИСК ОТТОКА")
            st.write("Клиент с высокой вероятностью покинет компанию")
        else:
            st.success("✅ НИЗКИЙ РИСК ОТТОКА")
            st.write("Клиент, скорее всего, останется в компании")
    
    with col_res2:
        st.metric("Вероятность оттока", f"{proba:.1%}")
    
    # Прогресс-бар
    st.progress(proba)
    st.caption(f"Порог оттока: 50% | Вероятность: {proba:.1%}")
    
except Exception as e:
    st.error(f"Ошибка при предсказании: {e}")

st.markdown("---")
st.caption("НИРС | Технологии машинного обучения | Королев А.С., ИУ5Ц-81Б | 2026")
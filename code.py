import pandas as pd
import sqlite3
import hashlib
import json
import re
from datetime import datetime, timedelta
import numpy as np
import os
import random

# ---------- 1. Генерация демонстрационных данных ----------
def generate_clients(n=100):
    np.random.seed(42)
    clients = []
    for i in range(1, n+1):
        cat = np.random.choice(['розница', 'бизнес'], p=[0.7, 0.3])
        if cat == 'бизнес':
            inn = str(random.randint(1000000000, 9999999999))  # 10 цифр
            # внесём несколько некорректных ИНН
            if i in [5, 15]:
                inn = "123"  # короткий
            if i == 25:
                inn = "12345678901A"  # не цифра
        else:
            inn = str(random.randint(100000000000, 999999999999))  # 12 цифр
            if i == 35:
                inn = "1234567890"  # 10 цифр для розницы – ошибка
        clients.append({'client_id': i, 'ИНН': inn, 'категория': cat})
    return pd.DataFrame(clients)

def generate_loans(clients_df, n=200):
    np.random.seed(123)
    loans = []
    client_ids = clients_df['client_id'].tolist()
    for i in range(1, n+1):
        cid = np.random.choice(client_ids)
        summ = round(np.random.uniform(5000, 5_000_000), 2)
        # создадим несколько ошибочных значений
        if i in [10, 20]:
            summ = -1000.0  # отрицательная сумма
        if i == 30:
            summ = 0.0
        rate = round(np.random.uniform(3.0, 60.0), 2)  # возможен >50
        issue_date = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365))
        term = np.random.choice([6, 12, 24, 36, 60])
        # остаток долга: не более суммы, для некоторых кредитов просрочка
        balance = round(np.random.uniform(0, summ) if summ>0 else 0.0, 2)
        overdue_days = np.random.choice([0, 0, 0, np.random.randint(1, 200)])  # чаще 0
        # добавим дубликат id для проверки
        if i == 50:
            loan_id = 50  # будет конфликт с i=50? Создадим ещё одну запись с таким же id
            # но сначала создадим настоящую 50-ю, а потом продублируем вручную после цикла
        loans.append({
            'id': i,
            'client_id': cid,
            'сумма': summ,
            'ставка': rate,
            'дата_выдачи': issue_date.strftime('%Y-%m-%d'),
            'срок': term,
            'остаток_долга': balance,
            'дни_просрочки': overdue_days
        })
    # вручную добавим дубликат id=50
    dup = loans[49].copy()
    dup['id'] = 50
    loans.append(dup)
    return pd.DataFrame(loans)

# ---------- 2. Валидация ----------
def validate_inn(inn, category):
    """Проверка ИНН: для бизнеса 10 цифр, для розницы 12 цифр."""
    if not isinstance(inn, str) or not re.fullmatch(r'\d+', inn):
        return False
    if category == 'бизнес' and len(inn) != 10:
        return False
    if category == 'розница' and len(inn) != 12:
        return False
    return True

def validate_loans(loans_df, clients_df):
    """Возвращает кортеж: (чистые кредиты, статистика ошибок)."""
    errors = {
        'inn_format': 0,
        'loan_sum_non_positive': 0,
        'rate_out_of_range': 0,
        'duplicate_ids': 0
    }
    # Проверка дубликатов id кредитов
    dup_mask = loans_df.duplicated(subset='id', keep=False)
    if dup_mask.any():
        dup_ids = loans_df[dup_mask]['id'].unique()
        errors['duplicate_ids'] = len(loans_df[loans_df['id'].isin(dup_ids)])
        # удаляем все дубликаты (оставляем первую нетронутую запись? лучше убрать обе, чтобы не было споров)
        loans_df = loans_df[~dup_mask].copy()
    
    # Присоединяем информацию о клиенте
    merged = loans_df.merge(clients_df, on='client_id', how='left')
    # В реальном проекте пропущенные клиенты – тоже ошибка, здесь клиенты всегда есть.
    
    # Проверка ИНН
    inn_invalid = ~merged.apply(lambda r: validate_inn(r['ИНН'], r['категория']), axis=1)
    errors['inn_format'] = inn_invalid.sum()
    
    # Проверка суммы кредита
    sum_invalid = merged['сумма'] <= 0
    errors['loan_sum_non_positive'] = sum_invalid.sum()
    
    # Проверка ставки
    rate_invalid = (merged['ставка'] < 0) | (merged['ставка'] > 50)
    errors['rate_out_of_range'] = rate_invalid.sum()
    
    # Фильтрация: удаляем записи с любыми ошибками
    invalid_mask = inn_invalid | sum_invalid | rate_invalid
    clean = merged[~invalid_mask].copy()
    skipped = invalid_mask.sum()
    
    return clean, errors, skipped

# ---------- 3. Агрегация ----------
def add_rate_range(rate):
    if rate <= 10:
        return '0-10%'
    elif rate <= 20:
        return '10-20%'
    elif rate <= 30:
        return '20-30%'
    elif rate <= 40:
        return '30-40%'
    else:
        return '40-50%'

def add_overdue_interval(days):
    if days == 0:
        return 'Без просрочки'
    elif days <= 30:
        return '1-30 дней'
    elif days <= 90:
        return '31-90 дней'
    elif days <= 180:
        return '91-180 дней'
    else:
        return '>180 дней'

def aggregate_data(clean_df):
    df = clean_df.copy()
    df['rate_range'] = df['ставка'].apply(add_rate_range)
    df['overdue_interval'] = df['дни_просрочки'].apply(add_overdue_interval)
    
    grouped = df.groupby(['категория', 'rate_range', 'overdue_interval'], as_index=False).agg(
        общая_сумма=('остаток_долга', 'sum'),
        просроченная_сумма=('остаток_долга', lambda x: x[df.loc[x.index, 'дни_просрочки'] > 0].sum()),
        средняя_ставка=('ставка', 'mean'),
        количество_договоров=('id', 'count')
    )
    # переименуем колонки для отчёта
    grouped.rename(columns={
        'категория': 'Категория клиентов',
        'rate_range': 'Диапазон ставок',
        'overdue_interval': 'Интервал просрочки',
        'общая_сумма': 'Общая сумма кредитов',
        'просроченная_сумма': 'Просроченная сумма',
        'средняя_ставка': 'Средняя ставка',
        'количество_договоров': 'Количество договоров'
    }, inplace=True)
    return grouped

# ---------- 4. Версионирование ----------
def generate_version_info(report_df, filename):
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Хеш данных: конвертируем датафрейм в CSV-строку без индекса
    csv_data = report_df.to_csv(index=False).encode('utf-8')
    hash_obj = hashlib.sha256(csv_data).hexdigest()
    version = {
        'дата_генерации': date_str,
        'файл_отчёта': filename,
        'количество_строк': len(report_df),
        'хеш_данных': hash_obj
    }
    return version

# ---------- 5. Главный блок ----------
def main():
    print("Генерация данных...")
    clients_df = generate_clients(100)
    loans_df = generate_loans(clients_df, 200)
    
    print("\n=== Панель контроля валидации ===")
    clean_loans, errors, skipped = validate_loans(loans_df, clients_df)
    total_records = len(loans_df)
    print(f"Всего записей: {total_records}")
    print(f"Пропущено (ошибки): {skipped}")
    print(f"  - дубликаты id: {errors['duplicate_ids']}")
    print(f"  - неверный формат ИНН: {errors['inn_format']}")
    print(f"  - сумма кредита <=0: {errors['loan_sum_non_positive']}")
    print(f"  - ставка вне [0,50]: {errors['rate_out_of_range']}")
    
    print("\nАгрегация данных...")
    report_df = aggregate_data(clean_loans)
    
    output_csv = 'report_0409701_simplified.csv'
    report_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Отчёт сохранён в {output_csv}")
    
    # Версионирование
    version_info = generate_version_info(report_df, output_csv)
    version_file = 'report_version.json'
    with open(version_file, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)
    print(f"Информация о версии сохранена в {version_file}")
    print("\n=== Версия отчёта ===")
    for k, v in version_info.items():
        print(f"{k}: {v}")
    
    print("\nГотово.")

if __name__ == "__main__":
    main()

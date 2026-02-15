import sqlite3

def create_connection():
    conn=sqlite3.connect("patients.db")
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        gender TEXT,
        symptoms TEXT,
        bp INTEGER,
        heart_rate INTEGER,
        temperature REAL,
        conditions TEXT,
        risk TEXT,
        department TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()
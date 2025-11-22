import sqlite3
from datetime import datetime

DB_PATH = "results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            generated_text TEXT,
            activation_traces TEXT,
            explanation TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_experiment(prompt, generated_text, activation_traces, explanation):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO experiments (prompt, generated_text, activation_traces, explanation, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (prompt, generated_text, activation_traces, explanation, datetime.now().isoformat()))
    conn.commit()
    exp_id = c.lastrowid
    conn.close()
    return exp_id

def get_experiment(exp_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,))
    row = c.fetchone()
    conn.close()
    return row

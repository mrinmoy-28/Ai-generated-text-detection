# backend/database.py
import sqlite3
import os
from datetime import datetime

DB_PATH = "backend/detector.db"

def init_db():
    """Create database and tables if they don't exist"""
    os.makedirs("backend", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            text          TEXT NOT NULL,
            verdict       TEXT NOT NULL,
            confidence    REAL NOT NULL,
            stat_score    REAL,
            roberta_score REAL,
            zeroshot_score REAL,
            watermark_score REAL,
            timestamp     TEXT NOT NULL,
            source        TEXT DEFAULT 'text'
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized!")


def save_detection(text, result, source="text"):
    """Save a detection result to database"""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    breakdown = result.get('breakdown', {})

    cursor.execute('''
        INSERT INTO detections
        (text, verdict, confidence, stat_score, roberta_score,
         zeroshot_score, watermark_score, timestamp, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        text[:1000],                          # limit stored text length
        result['verdict'],
        result['confidence'],
        breakdown.get('statistical', 0),
        breakdown.get('roberta', 0),
        breakdown.get('zero_shot', 0),
        breakdown.get('watermark', 0),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source
    ))

    conn.commit()
    conn.close()


def get_history(limit=50):
    """Fetch past detections"""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, text, verdict, confidence, timestamp, source
        FROM detections
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id":         row[0],
            "text":       row[1][:100] + "..." if len(row[1]) > 100 else row[1],
            "verdict":    row[2],
            "confidence": row[3],
            "timestamp":  row[4],
            "source":     row[5]
        }
        for row in rows
    ]


def get_stats():
    """Get aggregated stats for dashboard"""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total counts
    cursor.execute("SELECT COUNT(*) FROM detections")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE verdict = 'AI Generated'")
    ai_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE verdict = 'Human Written'")
    human_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM detections")
    avg_conf = cursor.fetchone()[0] or 0

    # Daily counts for last 7 days
    cursor.execute('''
        SELECT
            DATE(timestamp) as date,
            SUM(CASE WHEN verdict = 'AI Generated'  THEN 1 ELSE 0 END) as ai,
            SUM(CASE WHEN verdict = 'Human Written' THEN 1 ELSE 0 END) as human
        FROM detections
        WHERE timestamp >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date ASC
    ''')
    daily = cursor.fetchall()

    # Score distribution
    cursor.execute("SELECT confidence FROM detections ORDER BY timestamp DESC LIMIT 100")
    scores = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {
        "total_scans":    total,
        "ai_detected":    ai_count,
        "human_detected": human_count,
        "avg_confidence": round(avg_conf, 1),
        "daily_counts": [
            {"date": row[0], "ai": row[1], "human": row[2]}
            for row in daily
        ],
        "score_distribution": scores
    }


def clear_history():
    """Clear all detections — for testing"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()


# Initialize DB when this file is imported
init_db()
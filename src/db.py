"""
Neon Postgres persistence layer.

Stores every prediction made through the Streamlit demo so the interface
can show a history / audit trail, and so the thesis's "system evaluation"
chapter can reference real usage logs, not just offline test-set metrics.

Setup:
  1. Create a free Neon project at https://neon.tech
  2. Copy the connection string (Dashboard -> Connection Details)
  3. Put it in a .env file at the project root:
       NEON_DATABASE_URL=postgresql://user:password@ep-xxxx.neon.tech/tbdetect?sslmode=require
  4. Run `python -m src.db` once to create the table.
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

import psycopg2
import psycopg2.extras

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tb_predictions (
    id SERIAL PRIMARY KEY,
    patient_ref VARCHAR(100),
    image_filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    gradcam_path TEXT,
    clinician_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""


def get_connection():
    if not config.NEON_DATABASE_URL:
        raise RuntimeError(
            "NEON_DATABASE_URL is not set. Add it to a .env file at the project root."
        )
    return psycopg2.connect(config.NEON_DATABASE_URL)


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()
    print("tb_predictions table ready.")


def insert_prediction(
    image_filename: str,
    prediction: str,
    confidence: float,
    model_name: str,
    patient_ref: str = None,
    gradcam_path: str = None,
    clinician_notes: str = None,
) -> int:
    query = """
        INSERT INTO tb_predictions
            (patient_ref, image_filename, prediction, confidence, model_name, gradcam_path, clinician_notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    patient_ref,
                    image_filename,
                    prediction,
                    confidence,
                    model_name,
                    gradcam_path,
                    clinician_notes,
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return new_id


def fetch_history(limit: int = 50):
    query = """
        SELECT id, patient_ref, image_filename, prediction, confidence,
               model_name, gradcam_path, clinician_notes, created_at
        FROM tb_predictions
        ORDER BY created_at DESC
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()


if __name__ == "__main__":
    init_db()

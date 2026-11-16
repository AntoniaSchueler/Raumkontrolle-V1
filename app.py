from __future__ import annotations

from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path
import sqlite3
import hashlib

import numpy as np
import streamlit as st
from PIL import Image

from modules.classifier import TeachableMachineClassifier

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "raumkontrolle.db"
SCHOOL_TIMEZONE = ZoneInfo("Europe/Berlin")

def today_key():
    return datetime.now(SCHOOL_TIMEZONE).date().isoformat()

st.set_page_config(
    page_title="Raumkontrolle",
    page_icon="🏫",
    layout="centered",
)

# Die beiden Testräume können hier später erweitert werden.
ROOMS = ["Raum 174", "Raum 204"]

# Modell-Konfiguration
ROOM_MODEL_PATH = BASE_DIR / "models" / "room" / "keras_model.h5"
ROOM_LABELS_PATH = BASE_DIR / "models" / "room" / "labels.txt"
CHAIR_MODEL_PATH = BASE_DIR / "models" / "chairs" / "keras_model.h5"
CHAIR_LABELS_PATH = BASE_DIR / "models" / "chairs" / "labels.txt"

# -------------------- Datenbank --------------------

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL,
                check_date TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                chair_status TEXT NOT NULL,
                chair_confidence REAL NOT NULL,
                photo_count INTEGER NOT NULL,
                UNIQUE(room, check_date)
            )
        """)
        conn.commit()

def get_today_checks():
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT room, checked_at, chair_status, chair_confidence, photo_count
               FROM checks WHERE check_date = ?""",
            (today_key(),),
        ).fetchall()
    return {
        row[0]: {
            "checked_at": row[1],
            "chair_status": row[2],
            "chair_confidence": row[3],
            "photo_count": row[4],
        }
        for row in rows
    }

def save_check(room, chair_status, chair_confidence, photo_count):
    now = datetime.now(SCHOOL_TIMEZONE).isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO checks
               (room, check_date, checked_at, chair_status, chair_confidence, photo_count)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(room, check_date) DO UPDATE SET
                 checked_at=excluded.checked_at,
                 chair_status=excluded.chair_status,
                 chair_confidence=excluded.chair_confidence,
                 photo_count=excluded.photo_count
            """,
            (
                room,
                today_key(),
                now,
                chair_status,
                float(chair_confidence),
                int(photo_count),
            ),
        )
        conn.commit()

# -------------------- Modelle --------------------

@st.cache_resource
def load_models():
    room_model = TeachableMachineClassifier(ROOM_MODEL_PATH, ROOM_LABELS_PATH)
    chair_model = TeachableMachineClassifier(CHAIR_MODEL_PATH, CHAIR_LABELS_PATH)
    return room_model, chair_model

def reset_capture():
    st.session_state.photos = []
    st.session_state.capture_key += 1
    st.session_state.active_room = None
    st.session_state.saved = False
    st.session_state.seen_hashes = set()

init_db()

st.title("🏫 Raumkontrolle")
st.caption("Bitte fotografieren Sie den Raum aus mehreren Perspektiven.")

# Einstellungen für den Prototyp. Die Werte können später angepasst werden.
with st.sidebar:
    st.header("Erkennung")
    min_photos = st.number_input("Mindestens benötigte Fotos", min_value=2, max_value=6, value=2, step=1)
    confidence_threshold = st.slider(
        "Mindest-Sicherheit der Stuhlerkennung",
        min_value=0.50,
        max_value=0.99,
        value=0.85,
        step=0.01,
    )
    agreement_threshold = st.slider(
        "Übereinstimmung zwischen Fotos",
        min_value=0.50,
        max_value=1.00,
        value=0.75,
        step=0.05,
    )
    room_threshold = st.slider(
        "Mindest-Sicherheit der Raumerkennung",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
    )
    st.caption("Diese Schwellenwerte sind Startwerte und sollten mit echten Testfotos überprüft werden.")

st.subheader("Tagesübersicht")
today_checks = get_today_checks()

cols = st.columns(len(ROOMS))
for col, room in zip(cols, ROOMS):
    with col:
        if room in today_checks:
            st.success(f"**{room}**\n\nKontrolliert")
            st.caption(today_checks[room]["checked_at"].split("T")[-1])
        else:
            st.warning(f"**{room}**\n\nNoch offen")

st.divider()
st.subheader("Raum kontrollieren")

try:
    room_classifier, chair_classifier = load_models()
except Exception as exc:
    st.error("Die KI-Modelle konnten nicht geladen werden.")
    st.exception(exc)
    st.stop()

if "photos" not in st.session_state:
    st.session_state.photos = []
if "capture_key" not in st.session_state:
    st.session_state.capture_key = 0
if "active_room" not in st.session_state:
    st.session_state.active_room = None
if "saved" not in st.session_state:
    st.session_state.saved = False

if st.session_state.saved:
    st.success("Die Raumkontrolle wurde gespeichert. Vielen Dank!")
    if st.button("Nächsten Raum kontrollieren"):
        reset_capture()
        st.rerun()
else:
    st.info(
        "Nehmen Sie mindestens zwei Fotos aus unterschiedlichen Perspektiven auf. "
        "Die Fotos werden nur zur Auswertung verwendet und nicht dauerhaft gespeichert."
    )

    photo = st.camera_input(
        "Foto aufnehmen",
        key=f"camera_{st.session_state.capture_key}",
    )

    if photo is not None:
        raw_bytes = photo.getvalue()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        seen = st.session_state.get("seen_hashes", set())

        if digest not in seen:
            try:
                image = Image.open(photo).convert("RGB")
                room_result = room_classifier.predict(image)
                chair_result = chair_classifier.predict(image)

                detected_room, room_conf = room_result["label"], room_result["confidence"]

                if detected_room not in ROOMS:
                    st.error(f"Das Modell hat einen unbekannten Raum erkannt: {detected_room}")
                elif room_conf < room_threshold:
                    st.warning(
                        f"Der Raum wurde nicht sicher erkannt "
                        f"({detected_room}, {room_conf:.0%}). Bitte erneut fotografieren."
                    )
                elif st.session_state.active_room and detected_room != st.session_state.active_room:
                    st.error(
                        f"Dieses Foto scheint {detected_room} zu zeigen. "
                        f"Die bisherige Kontrolle betrifft {st.session_state.active_room}. "
                        "Bitte starten Sie eine neue Kontrolle."
                    )
                else:
                    st.session_state.active_room = detected_room
                    st.session_state.photos.append({
                        "room": detected_room,
                        "room_confidence": room_conf,
                        "chair_label": chair_result["label"],
                        "chair_confidence": chair_result["confidence"],
                        "chair_probabilities": chair_result["probabilities"],
                    })
                    seen.add(digest)
                    st.session_state.seen_hashes = seen
                    st.session_state.capture_key += 1
                    st.rerun()

            except Exception as exc:
                st.error("Das Foto konnte nicht ausgewertet werden.")
                st.exception(exc)

    photos = st.session_state.photos

    if photos:
        st.markdown(f"**Erkannter Raum: {st.session_state.active_room}**")
        st.write(f"Aufgenommene Fotos: **{len(photos)}**")

        for index, result in enumerate(photos, start=1):
            st.caption(
                f"Foto {index}: Stühle: {result['chair_label']} "
                f"({result['chair_confidence']:.0%}) · "
                f"Raum-Sicherheit: {result['room_confidence']:.0%}"
            )

        # Mittelwert der Wahrscheinlichkeiten über alle Fotos bilden.
        labels = chair_classifier.labels
        avg_probs = {}
        for label in labels:
            avg_probs[label] = float(np.mean([
                p["chair_probabilities"].get(label, 0.0) for p in photos
            ]))

        best_label = max(avg_probs, key=avg_probs.get)
        mean_confidence = avg_probs[best_label]
        agreement = sum(p["chair_label"] == best_label for p in photos) / len(photos)

        st.markdown("**Zusammengeführte Stuhlerkennung**")
        for label, probability in avg_probs.items():
            st.write(f"{label}: **{probability:.0%}**")
            st.progress(min(max(probability, 0.0), 1.0))

        enough_photos = len(photos) >= min_photos
        confidence_ok = mean_confidence >= confidence_threshold
        agreement_ok = agreement >= agreement_threshold
        ready = enough_photos and confidence_ok and agreement_ok

        if ready:
            st.success(
                f"Stühle ausreichend sicher erkannt: {best_label} "
                f"({mean_confidence:.0%}; Übereinstimmung {agreement:.0%})."
            )
            st.warning(
                "Die Fenstererkennung ist noch nicht integriert. "
                "Bitte prüfen Sie die Fenster vor dem Speichern selbst."
            )
            if st.button("Raumkontrolle speichern", type="primary", use_container_width=True):
                save_check(
                    st.session_state.active_room,
                    best_label,
                    mean_confidence,
                    len(photos),
                )
                st.session_state.saved = True
                st.rerun()
        else:
            reasons = []
            if not enough_photos:
                reasons.append(f"Es werden mindestens {int(min_photos)} Fotos benötigt.")
            if not confidence_ok:
                reasons.append("Die Stuhlerkennung ist noch nicht sicher genug.")
            if not agreement_ok:
                reasons.append("Die Ergebnisse der Fotos stimmen noch nicht ausreichend überein.")
            st.warning("Bitte nehmen Sie ein weiteres Foto aus einer anderen Perspektive auf. " + " ".join(reasons))

        if st.button("Aufnahme verwerfen und neu beginnen"):
            reset_capture()
            st.session_state.seen_hashes = set()
            st.rerun()

st.divider()
st.caption(
    "Hinweis: Diese erste Version erkennt keine Personen und kann daher nicht prüfen, "
    "ob sich noch jemand im Raum befindet. Bitte nur leere Räume fotografieren. "
    "Die Lehrkraft muss Fenster und andere noch nicht integrierte Kategorien selbst kontrollieren."
)


from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sqlite3
import hashlib

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# =====================================================
# 1. GRUNDEINSTELLUNGEN
# =====================================================

BASE = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Berlin")
DB = BASE / "data" / "raumkontrolle.db"

ROOMS = ["Raum 174", "Raum 204"]

ROOM_MODEL = BASE / "models" / "room" / "r keras_model.h5"
ROOM_LABELS = BASE / "models" / "room" / "r labels.txt"

CHAIR_MODEL = BASE / "models" / "chairs" / "c keras_model.h5"
CHAIR_LABELS = BASE / "models" / "chairs" / "c labels.txt"

LOGO = BASE / "logo.png"

st.set_page_config(
    page_title="Raumkontrolle",
    page_icon="🏫",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. DESIGN
# =====================================================

st.markdown("""
<style>

/* Hauptfarben */
:root {
    --red: #920D30;
    --dark-red: #700A24;
    --light-red: #F8E9ED;
    --anthracite: #343434;
}

/* Hintergrund */
.stApp {
    background-color: #F8F5F6;
    color: var(--anthracite);
}

/* Hauptbereich */
section.main {
    background-color: #F8F5F6;
}

/* Kopfbereich */
section.main > div {
    padding-top: 2rem;
}

/* Seitenleiste in Dunkelrot */
section[data-testid="stSidebar"] {
    background-color: #920D30;
    border-right: none;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Überschriften */
h1, h2, h3 {
    color: #920D30 !important;
    font-weight: 650;
}

/* Karten und Container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background-color: #920D30;
    color: white;
    border: 1px solid #920D30;
    border-radius: 9px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #700A24;
    border-color: #700A24;
    color: white;
}

/* Datei-Upload */
[data-testid="stFileUploader"] {
    background-color: #F8E9ED;
    border: 2px dashed #920D30;
    border-radius: 12px;
    padding: 12px;
}

/* Eingabefelder */
input, textarea {
    border-radius: 8px !important;
}

/* Fortschrittsanzeigen */
.stProgress > div > div > div {
    background-color: #920D30;
}

/* Auswahlfelder */
.stSelectbox, .stNumberInput {
    border-radius: 8px;
}

/* Trennlinien */
hr {
    border-color: #E6CDD4;
}

</style>
""", unsafe_allow_html=True)

""", unsafe_allow_html=True)


# =====================================================
# 3. LOGO UND HILFSFUNKTIONEN
# =====================================================

def show_logo():
    if LOGO.exists():
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.image(str(LOGO), use_container_width=True)
    else:
        st.warning("Das Schullogo logo.png wurde nicht gefunden.")


def get_connection():
    (BASE / "data").mkdir(exist_ok=True)
    return sqlite3.connect(DB)


def init_database():
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL,
                check_date TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                chair_status TEXT NOT NULL,
                photo_count INTEGER NOT NULL,
                UNIQUE(room, check_date)
            )
        """)
        con.commit()


def today():
    return datetime.now(TZ).date().isoformat()


def save_check(room, chair_status, photo_count):
    now = datetime.now(TZ).isoformat(timespec="seconds")

    with get_connection() as con:
        con.execute("""
            INSERT INTO checks
            (room, check_date, checked_at, chair_status, photo_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(room, check_date)
            DO UPDATE SET
                checked_at=excluded.checked_at,
                chair_status=excluded.chair_status,
                photo_count=excluded.photo_count
        """, (room, today(), now, chair_status, photo_count))
        con.commit()


def get_checked_rooms():
    with get_connection() as con:
        rows = con.execute(
            "SELECT room FROM checks WHERE check_date=?",
            (today(),)
        ).fetchall()

    return [row[0] for row in rows]


def parse_labels(path):
    labels = []

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return labels


def normalise_room(label):
    text = label.lower()

    if "174" in text:
        return "Raum 174"

    if "204" in text:
        return "Raum 204"

    return label


# =====================================================
# 4. KI-MODELLE
# =====================================================

def check_model_files():
    missing = []

    for path in [
        ROOM_MODEL, ROOM_LABELS,
        CHAIR_MODEL, CHAIR_LABELS
    ]:
        if not path.is_file():
            missing.append(str(path.relative_to(BASE)))

    return missing


@st.cache_resource
def load_model(model_path, labels_path):
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    labels = parse_labels(labels_path)

    return model, labels


def predict(model, labels, image):
    image = image.convert("RGB").resize((224, 224))

    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    array = np.expand_dims(array, axis=0)

    result = model.predict(array, verbose=0)[0]

    if len(result) != len(labels):
        raise ValueError("Die Anzahl der KI-Ausgabewerte stimmt nicht mit den Labels überein.")

    probabilities = {
        label: float(result[i])
        for i, label in enumerate(labels)
    }

    best = int(np.argmax(result))

    return {
        "label": labels[best],
        "confidence": float(result[best]),
        "probabilities": probabilities
    }


# =====================================================
# 5. SESSION STATE
# =====================================================

defaults = {
    "stage": "capture",
    "photos": [],
    "active_room": None,
    "camera_key": 0,
    "seen_hashes": set(),
    "notice": "",
    "chair_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_app():
    st.session_state.stage = "capture"
    st.session_state.photos = []
    st.session_state.active_room = None
    st.session_state.camera_key += 1
    st.session_state.seen_hashes = set()
    st.session_state.notice = ""
    st.session_state.chair_result = None


# =====================================================
# 6. STARTSEITE
# =====================================================

init_database()
show_logo()

if st.session_state.stage == "capture":

    st.title("Raumkontrolle")

    st.markdown("""
    <div class="hinweis">
        <p>Bitte nehmen Sie Fotos entsprechend der
        Abbildung im Raum auf.</p>
    </div>
    """, unsafe_allow_html=True)

    missing = check_model_files()

    if missing:
        st.error("Die KI-Modelle wurden nicht gefunden.")
        st.write("Bitte überprüfen Sie die Dateien in Ihrem GitHub-Repository:")

        for item in missing:
            st.code(item)

        st.info(
            "Die Dateien müssen in den angegebenen Ordnern liegen. "
            "Achten Sie auf die genaue Schreibweise."
        )
        st.stop()

    try:
        room_model, room_labels = load_model(
            ROOM_MODEL, ROOM_LABELS
        )

        chair_model, chair_labels = load_model(
            CHAIR_MODEL, CHAIR_LABELS
        )

    except Exception:
        st.error(
            "Die KI-Modelle konnten nicht geladen werden. "
            "Bitte überprüfen Sie die Modell- und Labeldateien."
        )
        st.stop()

    if st.session_state.notice:
        st.warning(st.session_state.notice)

    photo = st.camera_input(
        "Foto aufnehmen",
        key=f"camera_{st.session_state.camera_key}"
    )

    if photo is not None:
        image_bytes = photo.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()

        if digest not in st.session_state.seen_hashes:

            try:
                image = Image.open(photo).convert("RGB")

                room_result = predict(
                    room_model, room_labels, image
                )

                chair_result = predict(
                    chair_model, chair_labels, image
                )

                detected_room = normalise_room(
                    room_result["label"]
                )

                if detected_room not in ROOMS:
                    st.session_state.notice = (
                        "Der Raum konnte nicht eindeutig erkannt werden. "
                        "Bitte fotografieren Sie erneut."
                    )

                elif room_result["confidence"] < 0.70:
                    st.session_state.notice = (
                        "Die Raumerkennung ist unsicher. "
                        "Bitte nehmen Sie ein neues Foto auf."
                    )

                elif (
                    st.session_state.active_room is not None
                    and detected_room != st.session_state.active_room
                ):
                    st.session_state.notice = (
                        "Auf dem Foto wurde ein anderer Raum erkannt. "
                        "Bitte beginnen Sie eine neue Kontrolle."
                    )

                else:
                    st.session_state.active_room = detected_room
                    st.session_state.photos.append({
                        "room": detected_room,
                        "room_confidence": room_result["confidence"],
                        "chair_label": chair_result["label"],
                        "chair_confidence": chair_result["confidence"],
                        "chair_probabilities": chair_result["probabilities"]
                    })

                    st.session_state.seen_hashes.add(digest)
                    st.session_state.notice = ""

                st.session_state.camera_key += 1
                st.rerun()

            except Exception:
                st.error(
                    "Das Foto konnte nicht ausgewertet werden. "
                    "Bitte versuchen Sie es erneut."
                )

    photos = st.session_state.photos

    if photos:
        st.markdown(
            f"<div class='hinweis'><p>Erkannter Raum: "
            f"<b>{st.session_state.active_room}</b></p></div>",
            unsafe_allow_html=True
        )

        st.write(f"Aufgenommene Fotos: {len(photos)}")

        for i, result in enumerate(photos, 1):
            st.caption(
                f"Foto {i}: Stühle: {result['chair_label']} "
                f"({result['chair_confidence']:.0%})"
            )

        # Wahrscheinlichkeiten über alle Fotos mitteln
        avg_probs = {}

        for label in chair_labels:
            avg_probs[label] = float(np.mean([
                p["chair_probabilities"].get(label, 0)
                for p in photos
            ]))

        best_label = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[best_label]

        agreement = sum(
            p["chair_label"] == best_label
            for p in photos
        ) / len(photos)

        enough_photos = len(photos) >= 2
        confidence_ok = confidence >= 0.85
        agreement_ok = agreement >= 0.75

        if enough_photos and confidence_ok and agreement_ok:

            label_text = best_label.lower()

            if "oben" in label_text or "hoch" in label_text:
                st.session_state.chair_result = best_label

                st.session_state.stage = "confirm"
                st.rerun()

            else:
                st.error(
                    "Die Stühle sind noch nicht hochgestellt. "
                    "Bitte korrigieren Sie dies und fotografieren Sie erneut."
                )

                if st.button("Erneut fotografieren"):
                    reset_app()
                    st.rerun()

        else:
            st.info(
                "Bitte nehmen Sie ein weiteres Foto aus einer "
                "anderen Perspektive auf, bis die Erkennung ausreichend sicher ist."
            )

        if st.button("Zurück / Aufnahme neu beginnen"):
            reset_app()
            st.rerun()


# =====================================================
# 7. RAUMBESTÄTIGUNG
# =====================================================

elif st.session_state.stage == "confirm":

    st.title("Raumkontrolle")

    room = st.session_state.active_room

    st.markdown(
        f"""
        <div class="hinweis">
            <p style="font-size:22px;">
                {room} bestätigen
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    confirmed = st.checkbox(
        "Ich bestätige die Raumkontrolle."
    )

    if st.button("Bestätigen", disabled=not confirmed):
        save_check(
            room,
            st.session_state.chair_result,
            len(st.session_state.photos)
        )

        st.session_state.stage = "success"
        st.rerun()

    if st.button("Zurück"):
        st.session_state.stage = "capture"
        st.rerun()


# =====================================================
# 8. ERFOLGSANZEIGE
# =====================================================

elif st.session_state.stage == "success":

    st.title("Raumkontrolle")

    st.markdown("""
    <div class="hinweis">
        <p style="font-size:18px;">
            Raum als ordentlich erkannt,
            vielen Dank für die Unterstützung,
            unsere Räume sauber zu halten!
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Weitere Raumkontrolle"):
        reset_app()
        st.rerun()


# =====================================================
# 9. UNAUFDRINGLICHE TAGESÜBERSICHT
# =====================================================

with st.expander("Tagesübersicht"):

    checked = get_checked_rooms()

    for room in ROOMS:
        if room in checked:
            st.success(f"{room}: Kontrolliert")
        else:
            st.warning(f"{room}: Noch nicht kontrolliert")

    st.caption(
        "Die Übersicht zeigt nur den Kontrollstatus, "
        "keine Namen von Lehrkräften."
    )


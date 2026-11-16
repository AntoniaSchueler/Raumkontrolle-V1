# Raumkontrolle – Streamlit-Prototyp

Eine mobile Streamlit-WebApp für die Raumkontrolle. Diese erste Version ist für zwei Testräume ausgelegt und verwendet zwei mit Teachable Machine exportierte Keras-Modelle.

## Enthaltene Funktionen

- Raumerkennung für **Raum 174** und **Raum 204**
- Stuhlerkennung: **Stühle oben** / **Stühle unten**
- Aufnahme mehrerer Fotos mit der Smartphone-Kamera
- Zusammenführung der Stuhl-Wahrscheinlichkeiten mehrerer Perspektiven
- Konfigurierbare Mindestanzahl an Fotos und Konfidenzschwellen
- Tagesübersicht: kontrollierte und noch offene Räume
- Speicherung der Kontrollergebnisse in einer lokalen SQLite-Datenbank
- Fotos werden nicht dauerhaft gespeichert
- Fenstererkennung ist vorbereitet, aber noch nicht implementiert

## Projektstruktur

```text
raumkontrolle_streamlit/
├── app.py
├── modules/
│   └── classifier.py
├── models/
│   ├── room/
│   │   ├── keras_model.h5
│   │   └── labels.txt
│   └── chairs/
│       ├── keras_model.h5
│       └── labels.txt
├── data/
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md
```

## Lokal starten

Voraussetzung: Python 3.11 (empfohlen).

1. ZIP-Datei entpacken.
2. Im Terminal in den Projektordner wechseln.
3. Virtuelle Umgebung erstellen und aktivieren:

   **Windows**
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Abhängigkeiten installieren:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. App starten:

   ```bash
   streamlit run app.py
   ```

Streamlit zeigt eine lokale Adresse an. Für Tests mit dem Smartphone müssen Computer und Smartphone im selben WLAN sein. Starte Streamlit bei Bedarf mit:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Öffne anschließend auf dem Smartphone die lokale Netzwerkadresse, die Streamlit im Terminal ausgibt.

## Auf GitHub hochladen

1. Erstelle ein neues, möglichst **privates** GitHub-Repository.
2. Entpacke diese ZIP-Datei.
3. Lade den Inhalt des Projektordners in das Repository hoch (nicht den übergeordneten Ordner als zusätzliche Ebene).
4. Für Streamlit Community Cloud: Repository verbinden, `app.py` als Main file path auswählen und deployen.

**Wichtig:** Die Modelldateien sind Teil des Projekts. Prüfe vor dem Hochladen, ob die Schule das Hochladen der Modelle auf GitHub erlaubt. Bei einem öffentlichen Repository wären sie für alle sichtbar.

## Datenschutz und wichtige Grenzen des Prototyps

- Die App speichert Kontrollergebnisse, Uhrzeit, Raum, Stuhlergebnis und Anzahl der Fotos in `data/raumkontrolle.db`.
- Die Fotos selbst werden nicht in der Datenbank gespeichert. Sie werden im Arbeitsspeicher zur Auswertung verarbeitet.
- Es gibt in dieser Version **keine Personen- oder Gesichtserkennung**. Die App kann daher nicht automatisch verhindern, dass Personen auf einem Foto zu sehen sind. Es dürfen nur leere Räume fotografiert werden.
- Die Fenster werden noch nicht durch KI geprüft. Sie müssen vorerst manuell kontrolliert werden.
- Es gibt noch keine Benutzerkonten oder Anmeldung. Die App ist daher zunächst nur für einen geschützten Testbetrieb gedacht.
- SQLite ist für lokale Tests geeignet. Bei Streamlit Community Cloud kann lokaler Speicher bei Neustarts/Neubereitstellungen verloren gehen. Für einen dauerhaften Schulbetrieb sollte eine externe, dauerhaft gespeicherte Datenbank verwendet werden.
- Eine tägliche Kontrolle wird pro Raum und Datum gespeichert. Am nächsten Kalendertag wird der Raum automatisch wieder als offen angezeigt.
- Die angezeigten Konfidenzen sind Modellwerte, keine garantierten Wahrscheinlichkeiten für korrekte Erkennung. Die Schwellwerte müssen mit repräsentativen Fotos getestet werden.

## KI-Modelle ersetzen

Die Modelle können ersetzt werden, indem die Dateien unter den bestehenden Namen ausgetauscht werden:

- `models/room/keras_model.h5` und `models/room/labels.txt`
- `models/chairs/keras_model.h5` und `models/chairs/labels.txt`

Die Labelreihenfolge muss zur Ausgabereihenfolge des jeweiligen Modells passen. Beide Modelle müssen ein Teachable-Machine-Keras-Bildmodell mit 224 × 224 RGB-Eingabe sein.

## Fenstererkennung später ergänzen

Wenn das Fenstermodell fertig ist, kann es analog zur Stuhlerkennung als eigenes Modul eingebunden werden. Danach sollte die Kontrolle erst dann als vollständig gelten, wenn Raum, Stühle und Fenster ausreichend sicher geprüft wurden.

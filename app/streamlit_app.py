"""Streamlit front-end for the brain-tumour inference API.

Run the API first:
    uv run uvicorn diagnostic_des_tumeurs.api:app --port 8000

Then start this app:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

API_URL = os.environ.get("BTD_API_URL", "http://127.0.0.1:8000").rstrip("/")

CLASS_COLORS = {
    "glioma": "#f43f5e",
    "meningioma": "#f59e0b",
    "notumor": "#22c55e",
    "no tumor": "#22c55e",
    "pituitary": "#38bdf8",
}

CLASS_LABELS = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Pas de tumeur",
    "no tumor": "Pas de tumeur",
    "pituitary": "Tumeur hypophysaire",
}


def _class_key(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def _class_label(name: str) -> str:
    return CLASS_LABELS.get(_class_key(name), name.replace("_", " ").title())


def _class_color(name: str) -> str:
    return CLASS_COLORS.get(_class_key(name), "#22d3ee")

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(34, 211, 238, 0.14), transparent 34rem),
            radial-gradient(circle at 90% 10%, rgba(168, 85, 247, 0.12), transparent 30rem),
            #060b16;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .hero {
        padding: 1.65rem 1.8rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.9));
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        font-size: 2.35rem;
        line-height: 1.1;
        margin: 0 0 0.55rem 0;
        color: #f8fafc;
        letter-spacing: -0.04em;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.02rem;
        margin: 0;
        max-width: 760px;
    }
    .tag {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        margin: 0 0.35rem 0.85rem 0;
        border-radius: 999px;
        border: 1px solid rgba(34, 211, 238, 0.3);
        background: rgba(34, 211, 238, 0.08);
        color: #67e8f9;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .result-card {
        padding: 1.35rem 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(34, 211, 238, 0.28);
        background: linear-gradient(135deg, rgba(8, 47, 73, 0.55), rgba(15, 23, 42, 0.92));
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: #67e8f9;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .prediction {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        text-transform: capitalize;
    }
    .confidence {
        color: #cbd5e1;
        margin-top: 0.25rem;
    }
    .score-row {
        margin: 0.75rem 0 0.9rem 0;
    }
    .score-head {
        display: flex;
        justify-content: space-between;
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-bottom: 0.32rem;
    }
    .score-track {
        height: 8px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _post(
    endpoint: str,
    api_url: str,
    filename: str,
    image_bytes: bytes,
    content_type: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{api_url}{endpoint}",
            files={"file": (filename, image_bytes, content_type)},
            data=data or {},
            timeout=180,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Erreur API ({response.status_code}) : {detail}")
            return None
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Impossible de joindre l'API FastAPI : {exc}")
        return None


def _draw_detections(
    image: Image.Image, detections: list[dict[str, Any]]
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(14, round(min(canvas.size) / 32)))
    line_width = max(2, round(max(canvas.size) / 320))

    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        color = _class_color(detection["class_name"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{_class_label(detection['class_name'])}  {detection['confidence']:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        label_y = y1 - text_height - 8
        if label_y < 0:
            label_y = y1 + 4
        draw.rounded_rectangle(
            [x1, label_y, x1 + text_width + 12, label_y + text_height + 8],
            radius=6,
            fill=color,
        )
        draw.text((x1 + 6, label_y + 3), label, fill="#020617", font=font)

    return canvas


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _score_bars(scores: list[dict[str, Any]]) -> None:
    for item in scores:
        name = _class_label(str(item["class_name"]))
        score = float(item["score"])
        st.markdown(
            f"""
            <div class="score-row">
                <div class="score-head">
                    <span>{name}</span>
                    <span>{score:.1%}</span>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width: {score * 100:.1f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="hero">
        <div>
            <span class="tag">InceptionV3</span>
            <span class="tag">YOLOv8</span>
            <span class="tag">FastAPI + MLflow</span>
        </div>
        <h1>Analyse cérébrale assistée par IA</h1>
        <p>
            Téléversez une image IRM, puis choisissez la classification ou la
            détection et la localisation des tumeurs. Les prédictions sont
            calculées par l'API FastAPI et les modèles entraînés du projet.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connexion")
    api_url = st.text_input("URL de l'API FastAPI", value=API_URL).rstrip("/")

    if st.button("Tester l'API", width="stretch"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            st.success(f"API en ligne · statut : {health['status']}")
            classifier = health["models"].get("classification") or "absente"
            detector = health["models"].get("detection") or "absente"
            st.caption(f"Classification : {classifier}")
            st.caption(f"Détection : {detector}")
        except requests.RequestException as exc:
            st.error(f"API indisponible : {exc}")

    st.divider()
    uploaded = st.file_uploader(
        "Image cérébrale",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats acceptés : JPG, JPEG, PNG, BMP.",
    )
    st.caption("Les images ne sont pas conservées par l'interface.")
    st.caption("Démonstrateur — ne remplace pas un diagnostic médical.")


if uploaded is None:
    st.info("Téléversez une image IRM dans la barre latérale pour commencer.")
    st.stop()

image_bytes = uploaded.getvalue()
file_key = hashlib.sha256(image_bytes).hexdigest()[:16]
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.classification_result = None
    st.session_state.detection_result = None

try:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
except OSError:
    st.error("L'image téléversée est invalide ou illisible.")
    st.stop()

image_column, result_column = st.columns([1, 1.25], gap="large")

with image_column:
    st.image(image, caption=uploaded.name, width="stretch")
    st.caption(f"Dimensions : {image.width} × {image.height} px")

with result_column:
    classification_tab, detection_tab = st.tabs(
        ["🔬 Classification", "🎯 Détection"]
    )

    with classification_tab:
        st.subheader("Classification de l'image")
        st.caption(
            "Le modèle InceptionV3 retourne une probabilité pour chaque classe."
        )
        if st.button(
            "Classifier l'image",
            type="primary",
            width="stretch",
            key="classify_button",
        ):
            with st.spinner("Classification en cours..."):
                result = _post(
                    "/classify",
                    api_url,
                    uploaded.name,
                    image_bytes,
                    uploaded.type or "image/jpeg",
                )
            if result is not None:
                st.session_state.classification_result = result

        classification = st.session_state.get("classification_result")
        if classification:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="eyebrow">Classe prédite</div>
                    <div class="prediction">{_class_label(classification['predicted_class'])}</div>
                    <div class="confidence">
                        Confiance : {classification['confidence']:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _score_bars(classification["scores"])
            st.caption(f"Modèle utilisé : {classification['model']}")
        else:
            st.info("Lancez la classification pour afficher les scores.")

    with detection_tab:
        st.subheader("Détection et localisation")
        st.caption(
            "YOLOv8 localise les tumeurs avec des boîtes englobantes et un score "
            "de confiance."
        )
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Augmentez le seuil pour ne garder que les détections sûres.",
        )
        if st.button(
            "Détecter les tumeurs",
            type="primary",
            width="stretch",
            key="detect_button",
        ):
            with st.spinner("Détection en cours..."):
                result = _post(
                    "/detect",
                    api_url,
                    uploaded.name,
                    image_bytes,
                    uploaded.type or "image/jpeg",
                    data={
                        "conf": f"{confidence:.2f}",
                        "iou": "0.50",
                        "imgsz": "640",
                    },
                )
            if result is not None:
                st.session_state.detection_result = result

        detection = st.session_state.get("detection_result")
        if detection:
            metric_columns = st.columns(3)
            metric_columns[0].metric("Détections", detection["count"])
            metric_columns[1].metric(
                "Classes",
                len({item["class_name"] for item in detection["detections"]}),
            )
            metric_columns[2].metric(
                "Confiance max",
                (
                    f"{max(item['confidence'] for item in detection['detections']):.1%}"
                    if detection["detections"]
                    else "—"
                ),
            )

            annotated = _draw_detections(image, detection["detections"])
            st.image(
                annotated,
                caption="Détections YOLOv8",
                width="stretch",
            )

            if detection["detections"]:
                st.dataframe(
                    [
                        {
                            "Classe": _class_label(item["class_name"]),
                            "Confiance": f"{item['confidence']:.1%}",
                            "x1": round(item["xyxy"][0], 1),
                            "y1": round(item["xyxy"][1], 1),
                            "x2": round(item["xyxy"][2], 1),
                            "y2": round(item["xyxy"][3], 1),
                        }
                        for item in detection["detections"]
                    ],
                    hide_index=True,
                )
            else:
                st.info(
                    "Aucune tumeur détectée au seuil de confiance sélectionné."
                )
            st.caption(f"Modèle utilisé : {detection['model']}")
        else:
            st.info("Lancez la détection pour afficher les boîtes englobantes.")

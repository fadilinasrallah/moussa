"""Interface Streamlit pour un projet de traitement d'images sur le LSB."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import TARGET_SIZE, SteganographyDetector
from feature_extractor import FeatureExtractor
from lsb_steganography import LSBSteganography


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


TXT = {
    "title": "D\u00e9tection de st\u00e9ganographie LSB",
    "caption": "Projet de traitement d'images : cacher, extraire et d\u00e9tecter un message par bit de poids faible.",
    "detect": "D\u00e9tection",
    "create": "Cr\u00e9er des images de test",
    "extract": "Extraire le texte",
    "metrics": "M\u00e9triques et m\u00e9thode",
    "clean": "IMAGE PROPRE",
    "stego": "ST\u00c9GANOGRAPHIE D\u00c9TECT\u00c9E",
    "uncertain": "R\u00c9SULTAT INCERTAIN",
}


st.set_page_config(page_title=TXT["title"], layout="wide")

st.markdown(
    """
    <style>
    .verdict-clean, .verdict-stego, .verdict-warn {
        padding: 1.05rem 1.2rem;
        border-radius: 8px;
        margin: 0.25rem 0 0.8rem 0;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.01em;
    }
    .verdict-clean { background: #f0fdf4; border-left: 6px solid #16a34a; color: #14532d; }
    .verdict-stego { background: #fef2f2; border-left: 6px solid #dc2626; color: #7f1d1d; }
    .verdict-warn { background: #fffbeb; border-left: 6px solid #d97706; color: #78350f; }
    .step-title {
        font-size: 1.05rem;
        font-weight: 750;
        margin-top: 1.1rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_detectors(model_dir: str):
    root = Path(model_dir)
    if not (root / "svm_model.pkl").exists() or not (root / "rf_model.pkl").exists():
        return None
    svm = SteganographyDetector(model_name="svm")
    rf = SteganographyDetector(model_name="rf")
    svm.load(str(root))
    rf.load(str(root))
    return {"svm": svm, "rf": rf}


def file_time(path: Path) -> str:
    if not path.exists():
        return "absent"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def uploaded_size(uploaded) -> int:
    pos = uploaded.tell()
    uploaded.seek(0, 2)
    size = uploaded.tell()
    uploaded.seek(pos)
    return size


def uploaded_to_temp(uploaded, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded.seek(0)
        tmp.write(uploaded.read())
        return Path(tmp.name)


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def prepared_image(uploaded) -> Image.Image:
    return Image.open(uploaded).convert("RGB").resize(
        (TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS
    )


def run_both_models(detectors: dict, image_path: Path) -> tuple[dict, float]:
    start = time.time()
    svm = detectors["svm"].detect(str(image_path))
    rf = detectors["rf"].detect(str(image_path))
    elapsed_ms = (time.time() - start) * 1000
    return {
        "svm": svm,
        "rf": rf,
        "agreement": svm["label"] == rf["label"],
        "final_label": rf["label"] if svm["label"] == rf["label"] else None,
    }, elapsed_ms


def decode_lsb_header(arr: np.ndarray) -> dict:
    bits = arr.reshape(-1) & 1
    if bits.size < LSBSteganography.HEADER_BITS:
        return {"payload_bits": 0, "valid": False, "capacity_bits": int(bits.size)}
    payload_bits = 0
    for bit in bits[: LSBSteganography.HEADER_BITS]:
        payload_bits = (payload_bits << 1) | int(bit)
    payload_capacity = int(bits.size - LSBSteganography.HEADER_BITS)
    return {
        "payload_bits": int(payload_bits),
        "valid": 0 < payload_bits <= payload_capacity,
        "capacity_bits": int(bits.size),
        "payload_capacity_bits": payload_capacity,
    }


def channel_image(arr: np.ndarray, channel: int) -> Image.Image:
    out = np.zeros_like(arr)
    out[:, :, channel] = arr[:, :, channel]
    return Image.fromarray(out)


def lsb_plane_image(arr: np.ndarray, channel: int) -> Image.Image:
    return Image.fromarray(((arr[:, :, channel] & 1) * 255).astype(np.uint8))


def lbp_visual(arr: np.ndarray) -> Image.Image:
    green = arr[:, :, 1]
    lbp = local_binary_pattern(green, 8, 1, method="uniform")
    if lbp.max() > 0:
        lbp = (lbp / lbp.max()) * 255
    return Image.fromarray(lbp.astype(np.uint8))


def difference_visual(clean_img: Image.Image, stego_img: Image.Image) -> tuple[Image.Image, int, int]:
    clean = np.asarray(clean_img, dtype=np.uint8)
    stego = np.asarray(stego_img, dtype=np.uint8)
    diff = np.abs(stego.astype(np.int16) - clean.astype(np.int16)).astype(np.uint8)
    changed_values = int(np.sum(diff > 0))
    changed_pixels = int(np.sum(np.any(diff > 0, axis=2)))
    amplified = np.clip(diff * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified), changed_values, changed_pixels


def feature_rows(image: Image.Image) -> tuple[np.ndarray, list[dict], np.ndarray]:
    prepared = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(prepared, dtype=np.uint8)
    start = time.time()
    features = FeatureExtractor().extract(arr)
    elapsed = (time.time() - start) * 1000
    ratios = [(arr[:, :, c] & 1).mean() for c in range(3)]
    header = decode_lsb_header(arr)
    rows = [
        {"Mesure": "Dimensions analys\u00e9es", "Valeur": f"{TARGET_SIZE} x {TARGET_SIZE}"},
        {"Mesure": "Pixels analys\u00e9s", "Valeur": f"{TARGET_SIZE * TARGET_SIZE:,}"},
        {"Mesure": "Valeurs RGB analys\u00e9es", "Valeur": f"{TARGET_SIZE * TARGET_SIZE * 3:,}"},
        {"Mesure": "Caract\u00e9ristiques extraites", "Valeur": f"{features.size}"},
        {"Mesure": "Temps extraction features", "Valeur": f"{elapsed:.1f} ms"},
        {"Mesure": "Ratio LSB rouge", "Valeur": f"{ratios[0]:.4f}"},
        {"Mesure": "Ratio LSB vert", "Valeur": f"{ratios[1]:.4f}"},
        {"Mesure": "Ratio LSB bleu", "Valeur": f"{ratios[2]:.4f}"},
        {"Mesure": "Ent\u00eate LSB : longueur annonc\u00e9e", "Valeur": f"{header['payload_bits']:,} bits"},
        {"Mesure": "Ent\u00eate LSB dans la capacit\u00e9", "Valeur": "oui" if header["valid"] else "non"},
    ]
    return features, rows, arr


def render_verdict(final_label: int | None, bundle: dict) -> None:
    if final_label == 1:
        st.markdown(
            f"<div class='verdict-stego'>{TXT['stego']}<br>"
            "SVM et Random Forest classent l'image comme stego.</div>",
            unsafe_allow_html=True,
        )
    elif final_label == 0:
        st.markdown(
            f"<div class='verdict-clean'>{TXT['clean']}<br>"
            "SVM et Random Forest classent l'image comme propre.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-warn'>{TXT['uncertain']}<br>"
            "Les deux mod\u00e8les ne donnent pas le m\u00eame verdict.</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("SVM - P(stego)", f"{bundle['svm']['stego_probability'] * 100:.1f}%")
        st.caption(f"Seuil SVM : {bundle['svm'].get('svm_threshold', 0.5) * 100:.1f}%")
    with c2:
        st.metric("Random Forest - P(stego)", f"{bundle['rf']['stego_probability'] * 100:.1f}%")
        st.caption(f"Seuil RF : {bundle['rf'].get('rf_threshold', 0.5) * 100:.1f}%")


def render_detection_steps(image: Image.Image, base_rows: list[dict], bundle: dict, elapsed_ms: float) -> None:
    features, rows, arr = feature_rows(image)
    model_rows = [
        {"Mesure": "Temps total SVM + RF", "Valeur": f"{elapsed_ms:.1f} ms"},
        {"Mesure": "Verdict SVM", "Valeur": "stego" if bundle["svm"]["label"] == 1 else "propre"},
        {"Mesure": "Verdict Random Forest", "Valeur": "stego" if bundle["rf"]["label"] == 1 else "propre"},
        {"Mesure": "Accord des mod\u00e8les", "Valeur": "oui" if bundle["agreement"] else "non"},
    ]

    st.markdown("<div class='step-title'>1. Image pr\u00e9par\u00e9e pour l'analyse</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns([1, 1, 1])
    p1.image(image, caption=f"Image re\u00e7ue : {image.width} x {image.height}", width=260)
    p2.image(Image.fromarray(arr), caption=f"Image analys\u00e9e : {TARGET_SIZE} x {TARGET_SIZE}", width=260)
    p3.dataframe(base_rows + rows[:5], hide_index=True, use_container_width=True)

    st.markdown("<div class='step-title'>2. S\u00e9paration en canaux couleur</div>", unsafe_allow_html=True)
    r_col, g_col, b_col = st.columns(3)
    r_col.image(channel_image(arr, 0), caption="Canal rouge R", width=220)
    g_col.image(channel_image(arr, 1), caption="Canal vert G", width=220)
    b_col.image(channel_image(arr, 2), caption="Canal bleu B", width=220)

    st.markdown("<div class='step-title'>3. Plans LSB extraits des trois canaux</div>", unsafe_allow_html=True)
    lr, lg, lb = st.columns(3)
    lr.image(lsb_plane_image(arr, 0), caption="LSB du canal R", width=220)
    lg.image(lsb_plane_image(arr, 1), caption="LSB du canal G", width=220)
    lb.image(lsb_plane_image(arr, 2), caption="LSB du canal B", width=220)

    st.markdown("<div class='step-title'>4. Filtre de texture utilis\u00e9 dans les features</div>", unsafe_allow_html=True)
    f1, f2 = st.columns([1, 1.4])
    f1.image(lbp_visual(arr), caption="Carte LBP calcul\u00e9e sur le canal vert", width=260)
    f2.dataframe(rows[5:] + model_rows, hide_index=True, use_container_width=True)


def embedding_rows(clean_img: Image.Image, stego_img: Image.Image, message: str, psnr: float) -> tuple[list[dict], Image.Image]:
    diff_img, changed_values, changed_pixels = difference_visual(clean_img, stego_img)
    total_values = clean_img.width * clean_img.height * 3
    payload_bits = len((message + LSBSteganography.END_MARKER).encode("utf-8")) * 8
    total_bits = payload_bits + LSBSteganography.HEADER_BITS
    rows = [
        {"Mesure": "Dimensions de travail", "Valeur": f"{clean_img.width} x {clean_img.height} RGB"},
        {"Mesure": "Capacit\u00e9 LSB totale", "Valeur": f"{total_values:,} bits"},
        {"Mesure": "Message utilisateur", "Valeur": f"{len(message)} caract\u00e8res"},
        {"Mesure": "Payload UTF-8 + marqueur", "Valeur": f"{payload_bits:,} bits"},
        {"Mesure": "Ent\u00eate de longueur", "Valeur": f"{LSBSteganography.HEADER_BITS} bits"},
        {"Mesure": "Bits \u00e9crits", "Valeur": f"{total_bits:,} bits"},
        {"Mesure": "Occupation", "Valeur": f"{total_bits / total_values * 100:.4f}%"},
        {"Mesure": "Valeurs RGB modifi\u00e9es", "Valeur": f"{changed_values:,} / {total_values:,}"},
        {"Mesure": "Pixels touch\u00e9s", "Valeur": f"{changed_pixels:,} / {clean_img.width * clean_img.height:,}"},
        {"Mesure": "PSNR", "Valeur": f"{psnr:.2f} dB"},
    ]
    return rows, diff_img


def extraction_rows(image: Image.Image, text: str, elapsed_ms: float) -> list[dict]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    header = decode_lsb_header(arr)
    return [
        {"Mesure": "Dimensions lues", "Valeur": f"{image.width} x {image.height} RGB"},
        {"Mesure": "Capacit\u00e9 LSB", "Valeur": f"{header['capacity_bits']:,} bits"},
        {"Mesure": "Longueur annonc\u00e9e par l'ent\u00eate", "Valeur": f"{header['payload_bits']:,} bits"},
        {"Mesure": "Longueur valide", "Valeur": "oui" if header["valid"] else "non"},
        {"Mesure": "Marqueur de fin trouv\u00e9", "Valeur": "oui" if text else "non"},
        {"Mesure": "Texte extrait", "Valeur": f"{len(text)} caract\u00e8res"},
        {"Mesure": "Temps d'extraction", "Valeur": f"{elapsed_ms:.1f} ms"},
    ]


def render_metrics(metrics: dict) -> None:
    rows = []
    for name, values in metrics.items():
        rows.append(
            {
                "Mod\u00e8le": "SVM lin\u00e9aire calibr\u00e9" if name == "SVM" else "Random Forest calibr\u00e9",
                "Accuracy": round(values["accuracy"], 4),
                "Pr\u00e9cision": round(values["precision"], 4),
                "Recall": round(values["recall"], 4),
                "F1": round(values["f1"], 4),
                "AUC": round(values["auc_roc"], 4),
                "Seuil": round(values.get("threshold", 0.5), 4),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


summary = read_json(str(RESULTS_DIR / "training_summary.json"))
metrics = read_json(str(RESULTS_DIR / "metrics.json"))
thresholds = read_json(str(MODELS_DIR / "thresholds.json"))
detectors = load_detectors(str(MODELS_DIR))
models_ready = detectors is not None


st.title(TXT["title"])
st.caption(TXT["caption"])

with st.sidebar:
    st.header("\u00c9tat r\u00e9el du projet")
    if models_ready:
        st.success("Mod\u00e8les charg\u00e9s")
    else:
        st.error("Mod\u00e8les absents")
    st.write(f"Taille d'analyse : {TARGET_SIZE} x {TARGET_SIZE}")
    st.write(f"SVM : {file_time(MODELS_DIR / 'svm_model.pkl')}")
    st.write(f"Random Forest : {file_time(MODELS_DIR / 'rf_model.pkl')}")
    thresholds_display = thresholds if thresholds else "non charg\u00e9s"
    st.write(f"Seuils : {thresholds_display}")
    st.divider()
    st.subheader("Donn\u00e9es d'entra\u00eenement")
    if summary:
        st.write(f"Source : `{summary.get('alaska_dir', 'inconnue')}`")
        st.write(f"Images cover : {summary.get('n_cover_images', 0):,}")
        st.write(f"Train : {summary.get('train_images', 0):,}")
        st.write(f"Validation : {summary.get('val_images', 0):,}")
        st.write(f"Test : {summary.get('test_images', 0):,}")
        st.write(f"Features : {summary.get('features', 0)}")


tab_detect, tab_create, tab_extract, tab_metrics = st.tabs(
    [TXT["detect"], TXT["create"], TXT["extract"], TXT["metrics"]]
)

with tab_detect:
    st.subheader("Analyser une image")
    uploaded = st.file_uploader(
        "Image \u00e0 analyser",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="detect_upload",
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        size = uploaded_size(uploaded)
        raw_image = Image.open(uploaded)
        image = raw_image.convert("RGB")
        base_rows = [
            {"Mesure": "Fichier re\u00e7u", "Valeur": uploaded.name},
            {"Mesure": "Taille fichier", "Valeur": f"{size:,} octets"},
            {"Mesure": "Format", "Valeur": raw_image.format or suffix.upper().replace(".", "")},
            {"Mesure": "Dimensions originales", "Valeur": f"{raw_image.width} x {raw_image.height}"},
            {"Mesure": "Mode converti", "Valeur": image.mode},
        ]

        top_left, top_right = st.columns([1, 1.15])
        with top_left:
            st.image(image, caption=f"Image entr\u00e9e : {uploaded.name}", width=380)
            if suffix.lower() in {".jpg", ".jpeg"}:
                st.info("JPEG accept\u00e9 pour la d\u00e9tection. Pour extraire un message LSB, il faut un PNG non recompress\u00e9.")

        with top_right:
            if not models_ready:
                st.error("Les mod\u00e8les ne sont pas charg\u00e9s.")
            else:
                tmp = uploaded_to_temp(uploaded, suffix)
                try:
                    bundle, elapsed = run_both_models(detectors, tmp)
                finally:
                    tmp.unlink(missing_ok=True)
                render_verdict(bundle["final_label"], bundle)

        if models_ready:
            st.divider()
            st.subheader("\u00c9tapes visuelles calcul\u00e9es sur cette image")
            render_detection_steps(image, base_rows, bundle, elapsed)

with tab_create:
    st.subheader("Cr\u00e9er une image propre et une image stego")
    st.write("On fabrique deux sorties comparables : une image PNG propre et la m\u00eame image apr\u00e8s insertion LSB.")

    source = st.file_uploader("Image source", type=["png", "jpg", "jpeg", "bmp", "webp"], key="create_source")
    message = st.text_area("Message \u00e0 cacher", height=130, max_chars=20000)

    if source is not None:
        src_img = Image.open(source)
        prep = prepared_image(source)
        rows = [
            {"Mesure": "Fichier source", "Valeur": source.name},
            {"Mesure": "Format source", "Valeur": src_img.format or "inconnu"},
            {"Mesure": "Dimensions source", "Valeur": f"{src_img.width} x {src_img.height}"},
            {"Mesure": "Dimensions pr\u00e9par\u00e9es", "Valeur": f"{TARGET_SIZE} x {TARGET_SIZE}"},
            {"Mesure": "Capacit\u00e9 utile", "Valeur": f"{(TARGET_SIZE * TARGET_SIZE * 3 - 32) // 8:,} octets environ"},
        ]
        c_prev, c_data = st.columns([1, 1.3])
        c_prev.image(prep, caption="Aper\u00e7u pr\u00e9par\u00e9", width=320)
        c_data.dataframe(rows, hide_index=True, use_container_width=True)

    if st.button("G\u00e9n\u00e9rer les deux images", type="primary", disabled=source is None or not message):
        prep = prepared_image(source)
        lsb = LSBSteganography()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as clean_tmp:
            clean_path = Path(clean_tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as stego_tmp:
            stego_path = Path(stego_tmp.name)
        try:
            prep.save(clean_path, format="PNG")
            psnr = lsb.embed(str(clean_path), message, str(stego_path))
            clean_img = Image.open(clean_path).convert("RGB")
            stego_img = Image.open(stego_path).convert("RGB")
            rows, diff_img = embedding_rows(clean_img, stego_img, message, psnr)

            st.markdown("<div class='step-title'>1. Sorties g\u00e9n\u00e9r\u00e9es</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.image(clean_img, caption="PNG propre", width=260)
            c2.image(stego_img, caption="PNG stego", width=260)
            c3.image(diff_img, caption="Diff\u00e9rence amplifi\u00e9e x255", width=260)

            d1, d2 = st.columns(2)
            d1.download_button("T\u00e9l\u00e9charger le PNG propre", data=to_png_bytes(clean_img), file_name="controle_propre.png", mime="image/png")
            d2.download_button("T\u00e9l\u00e9charger le PNG stego", data=to_png_bytes(stego_img), file_name="image_stego.png", mime="image/png")

            st.markdown("<div class='step-title'>2. Mesures r\u00e9elles de l'insertion</div>", unsafe_allow_html=True)
            st.dataframe(rows, hide_index=True, use_container_width=True)

            st.markdown("<div class='step-title'>3. Plans LSB apr\u00e8s insertion</div>", unsafe_allow_html=True)
            arr_stego = np.asarray(stego_img, dtype=np.uint8)
            l1, l2, l3 = st.columns(3)
            l1.image(lsb_plane_image(arr_stego, 0), caption="LSB rouge de l'image stego", width=220)
            l2.image(lsb_plane_image(arr_stego, 1), caption="LSB vert de l'image stego", width=220)
            l3.image(lsb_plane_image(arr_stego, 2), caption="LSB bleu de l'image stego", width=220)
        except ValueError as exc:
            st.error(str(exc))
        finally:
            clean_path.unlink(missing_ok=True)
            stego_path.unlink(missing_ok=True)

with tab_extract:
    st.subheader("Extraire un message cach\u00e9")
    stego_upload = st.file_uploader("Image PNG \u00e0 extraire", type=["png"], key="extract_upload")

    if stego_upload is not None:
        image = Image.open(stego_upload).convert("RGB")
        c_img, c_info = st.columns([1, 1.3])
        c_img.image(image, caption=f"Image re\u00e7ue : {stego_upload.name}", width=340)
        c_info.write("L'extraction lit les 32 premiers bits LSB comme longueur, puis lit le payload et cherche le marqueur de fin.")

    if st.button("Extraire le message", disabled=stego_upload is None):
        tmp = uploaded_to_temp(stego_upload, ".png")
        image = Image.open(tmp).convert("RGB")
        start = time.time()
        try:
            text = LSBSteganography().extract(str(tmp))
        finally:
            elapsed = (time.time() - start) * 1000
            tmp.unlink(missing_ok=True)

        st.markdown("<div class='step-title'>1. Plans LSB lus pendant l'extraction</div>", unsafe_allow_html=True)
        arr = np.asarray(image, dtype=np.uint8)
        e1, e2, e3 = st.columns(3)
        e1.image(lsb_plane_image(arr, 0), caption="LSB rouge lu", width=220)
        e2.image(lsb_plane_image(arr, 1), caption="LSB vert lu", width=220)
        e3.image(lsb_plane_image(arr, 2), caption="LSB bleu lu", width=220)

        st.markdown("<div class='step-title'>2. Donn\u00e9es r\u00e9elles de l'extraction</div>", unsafe_allow_html=True)
        st.dataframe(extraction_rows(image, text, elapsed), hide_index=True, use_container_width=True)
        if text:
            st.success("Message trouv\u00e9")
            st.text_area("Texte extrait", value=text, height=200)
        else:
            st.warning("Aucun message valide trouv\u00e9.")

with tab_metrics:
    st.subheader("M\u00e9triques r\u00e9elles du dernier entra\u00eenement")
    render_metrics(metrics)
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images cover", f"{summary.get('n_cover_images', 0):,}")
        m2.metric("Train", f"{summary.get('train_images', 0):,}")
        m3.metric("Validation", f"{summary.get('val_images', 0):,}")
        m4.metric("Test", f"{summary.get('test_images', 0):,}")

    st.subheader("M\u00e9thode")
    st.write(
        "Les images propres et stego sont toutes deux export\u00e9es en PNG apr\u00e8s le m\u00eame pr\u00e9traitement. "
        "On \u00e9vite donc le raccourci JPEG contre PNG."
    )
    st.write(
        "Les filtres et mesures utilis\u00e9s sont visibles dans l'onglet D\u00e9tection : canaux R/V/B, plans LSB, LBP, "
        "statistiques LSB, histogrammes RGB, GLCM et moments statistiques."
    )
    st.code("python run_pipeline.py --n 4000 --workers 15 --tune --force-stego --force-features", language="bash")

    p1, p2 = st.columns(2)
    if (RESULTS_DIR / "confusion_matrices.png").exists():
        p1.image(str(RESULTS_DIR / "confusion_matrices.png"), caption="Matrices de confusion")
    if (RESULTS_DIR / "roc_curves.png").exists():
        p2.image(str(RESULTS_DIR / "roc_curves.png"), caption="Courbes ROC")

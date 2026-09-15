import streamlit as st
import cv2
import numpy as np
import os
import time
import tensorflow as tf
import pandas as pd

from datetime import datetime
from collections import Counter
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PCB VISION",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

os.makedirs("outputs", exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

if "inspection_results" not in st.session_state:
    st.session_state.inspection_results = None

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0


# =========================================================
# AI MODEL SETTINGS
# =========================================================

MODEL_PATH = "pcb_defect_model.keras"

IMAGE_SIZE = 64


# =========================================================
# DEFECT CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Copper",
    "Mousebite",
    "Open",
    "Pin-hole",
    "Short",
    "Spur"
]


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        model = tf.keras.models.load_model(MODEL_PATH)

        return model

    except Exception:

        return None


ai_model = load_ai_model()


# =========================================================
# AI DEFECT CLASSIFICATION
# =========================================================

def classify_defect(image):

    if ai_model is None:
        return "AI Model Not Loaded", 0.0

    try:

        if image is None or image.size == 0:
            return "Invalid Region", 0.0

        image_resized = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image_rgb = cv2.cvtColor(
            image_resized,
            cv2.COLOR_BGR2RGB
        )

        image_array = np.array(
            image_rgb,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = ai_model.predict(
            image_array,
            verbose=0
        )

        predicted_index = np.argmax(prediction[0])

        confidence = float(
            prediction[0][predicted_index]
        ) * 100

        defect_name = CLASS_NAMES[predicted_index]

        return defect_name, confidence

    except Exception:

        return "Classification Error", 0.0


# =========================================================
# DEFECT SEVERITY
# =========================================================

def calculate_severity(area, confidence):

    if area >= 1500:
        return "HIGH"

    elif area >= 500:
        return "MEDIUM"

    else:
        return "LOW"


# =========================================================
# SEVERITY COLOR
# =========================================================

def get_severity_color(severity):

    if severity == "HIGH":
        return "#ef4444"

    elif severity == "MEDIUM":
        return "#facc15"

    return "#22c55e"


# =========================================================
# IMAGE TO BYTES
# =========================================================

def image_to_bytes(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image
    )

    if success:
        return buffer.tobytes()

    return None


# =========================================================
# PDF REPORT
# =========================================================

def generate_pdf_report(
    defect_count,
    status,
    resolution,
    processing_time,
    detected_defects,
    result_image
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]


    # =====================================================
    # TITLE
    # =====================================================

    elements.append(

        Paragraph(
            "PCB Fault Detection & Inspection Report",
            title_style
        )

    )

    elements.append(
        Spacer(1, 20)
    )


    # =====================================================
    # REPORT INFORMATION
    # =====================================================

    date_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    report_data = [

        [
            "Inspection Date",
            date_time
        ],

        [
            "Inspection Status",
            status
        ],

        [
            "Defects Found",
            str(defect_count)
        ],

        [
            "Image Resolution",
            resolution
        ],

        [
            "Processing Time",
            f"{processing_time:.3f} seconds"
        ],

        [
            "Detection Method",
            "OpenCV + TensorFlow CNN"
        ]

    ]

    report_table = Table(
        report_data,
        colWidths=[180, 300]
    )

    report_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#0b1f3a")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (0, -1),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                10
            )

        ])

    )

    elements.append(report_table)

    elements.append(
        Spacer(1, 25)
    )


    # =====================================================
    # DEFECT DETAILS
    # =====================================================

    if defect_count > 0:

        elements.append(

            Paragraph(
                "Detected Defects",
                heading_style
            )

        )

        elements.append(
            Spacer(1, 10)
        )

        table_data = [

            [
                "No.",
                "Defect Type",
                "Confidence",
                "Severity",
                "Area"
            ]

        ]

        for defect in detected_defects:

            table_data.append([

                str(defect["number"]),

                str(defect["type"]),

                f'{defect["confidence"]:.2f}%',

                str(defect["severity"]),

                f'{defect["area"]:.0f}'

            ])

        defect_table = Table(
            table_data,
            repeatRows=1
        )

        defect_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#0891b2")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )

            ])

        )

        elements.append(defect_table)

        elements.append(
            Spacer(1, 25)
        )


    # =====================================================
    # RESULT IMAGE
    # =====================================================

    elements.append(

        Paragraph(
            "Inspection Result",
            heading_style
        )

    )

    elements.append(
        Spacer(1, 10)
    )

    try:

        image_path = "outputs/pdf_result.jpg"

        cv2.imwrite(
            image_path,
            result_image
        )

        pdf_image = Image(
            image_path,
            width=6.5 * inch,
            height=4 * inch
        )

        elements.append(pdf_image)

    except Exception:

        pass


    # =====================================================
    # FOOTER
    # =====================================================

    elements.append(
        Spacer(1, 25)
    )

    footer = Paragraph(

        "PCB Fault Detection & Inspection System<br/>"
        "Built using Python, OpenCV, TensorFlow, CNN and Streamlit.",

        normal_style

    )

    elements.append(footer)


    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(elements)

    buffer.seek(0)

    return buffer


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,255,255,0.10), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(37,99,235,0.15), transparent 30%),
        linear-gradient(135deg, #07111f, #0b1b33, #07111f);
    color: white;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #050b16, #0b1f3a, #07111f);
    border-right: 1px solid rgba(0,255,255,0.25);
}

section[data-testid="stSidebar"] * {
    color: #e5f6ff !important;
}

.hero-container {
    padding: 45px 40px;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(5,15,30,0.98),
            rgba(10,45,80,0.95),
            rgba(8,80,100,0.90)
        );
    border: 1px solid rgba(0,255,255,0.30);
    box-shadow: 0px 10px 45px rgba(0,255,255,0.12);
    margin-bottom: 30px;
}

.hero-kicker {
    color: #67e8f9;
    font-size: 13px;
    letter-spacing: 3px;
    font-weight: 700;
    margin-bottom: 15px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero-highlight {
    color: #22d3ee;
}

.hero-description {
    color: #b6d5e8;
    font-size: 17px;
    max-width: 800px;
    line-height: 1.7;
    margin-top: 15px;
}

.hero-badge {
    display: inline-block;
    margin-top: 22px;
    padding: 10px 18px;
    border-radius: 25px;
    background: rgba(0,255,255,0.10);
    border: 1px solid rgba(0,255,255,0.25);
    color: #67e8f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.section-title {
    color: white;
    font-size: 28px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-description {
    color: #9fb9c9;
    font-size: 15px;
    margin-bottom: 20px;
}

.dashboard-card {
    background:
        linear-gradient(
            145deg,
            rgba(16,33,55,0.95),
            rgba(8,22,40,0.95)
        );
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 18px;
    padding: 22px;
    min-height: 115px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.20);
}

.card-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.card-value {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin-top: 12px;
}

.card-red {
    border-top: 3px solid #ef4444;
}

.card-green {
    border-top: 3px solid #22c55e;
}

.card-blue {
    border-top: 3px solid #3b82f6;
}

.card-purple {
    border-top: 3px solid #a855f7;
}

.image-card {
    background:
        linear-gradient(
            145deg,
            rgba(15,35,55,0.9),
            rgba(5,15,30,0.9)
        );
    border: 1px solid rgba(0,255,255,0.15);
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 15px;
}

div[data-testid="stFileUploader"] {
    background: rgba(10,30,50,0.70);
    border: 1px dashed rgba(34,211,238,0.50);
    border-radius: 15px;
    padding: 12px;
}

.stButton > button {
    border-radius: 14px;
    min-height: 52px;
    font-size: 15px;
    font-weight: 800;
}

.status-pass {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.45);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.status-defect {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.45);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.status-label {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.status-pass-title {
    color: #4ade80;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.status-defect-title {
    color: #f87171;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.ai-result-card {
    background:
        linear-gradient(
            145deg,
            rgba(88,28,135,0.30),
            rgba(30,41,59,0.80)
        );
    border: 1px solid rgba(168,85,247,0.45);
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 35px;
    font-size: 13px;
    margin-top: 40px;
    border-top: 1px solid rgba(148,163,184,0.12);
}

h1, h2, h3, h4 {
    color: white !important;
}

/* ================= DASHBOARD UI ================= */
.dashboard-heading {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
    margin: 10px 0 6px 0;
}

.dashboard-subtitle {
    color: #b6c8d8;
    font-size: 15px;
    margin-bottom: 22px;
}

.dashboard-card {
    background: linear-gradient(145deg, rgba(20,42,68,0.96), rgba(9,25,45,0.98));
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 118px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.22);
}

.card-label {
    color: #b8c7d8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.card-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 800;
    line-height: 1.2;
    word-break: break-word;
}

.card-red { border-top: 3px solid #ef4444; }
.card-green { border-top: 3px solid #22c55e; }
.card-blue { border-top: 3px solid #38bdf8; }
.card-purple { border-top: 3px solid #a855f7; }

.dashboard-alert {
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0 8px 0;
}

.alert-pass {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.42);
}

.alert-defect {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.42);
}

.alert-title {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.alert-pass .alert-title { color: #4ade80; }
.alert-defect .alert-title { color: #f87171; }

.alert-text {
    color: #d7e3ec;
    font-size: 15px;
}

.severity-card {
    background: linear-gradient(145deg, rgba(20,42,68,0.90), rgba(9,25,45,0.96));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 15px;
    padding: 18px 20px;
}

.severity-label {
    color: #c4d1dc;
    font-size: 13px;
    font-weight: 700;
}

.severity-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.classification-card {
    background: linear-gradient(145deg, rgba(17,38,64,0.96), rgba(10,25,45,0.98));
    border: 1px solid rgba(56,189,248,0.20);
    border-radius: 18px;
    padding: 22px;
    margin: 12px 0 16px 0;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.classification-title {
    color: #ffffff;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 18px;
}

.classification-metric {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 12px;
    padding: 14px 16px;
}

.classification-label {
    color: #b9c9d8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.classification-value {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
    margin-top: 7px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔬 PCB VISION")

    st.caption(
        "AI-POWERED INSPECTION"
    )

    st.divider()

    st.subheader("🧭 SYSTEM")

    st.write("◈ Dashboard")
    st.write("◈ PCB Inspector")
    st.write("◈ AI Classification")
    st.write("◈ Analytics")
    st.write("◈ Inspection History")

    st.divider()

    st.subheader(
        "⚙️ DETECTION PARAMETERS"
    )

    threshold_value = st.slider(
        "Difference Threshold",
        min_value=1,
        max_value=100,
        value=30
    )

    min_area = st.slider(
        "Minimum Defect Area",
        min_value=1,
        max_value=500,
        value=50
    )

    st.divider()

    st.subheader("🤖 AI MODEL")

    if ai_model is not None:

        st.success(
            "✓ AI Model Loaded"
        )

        st.caption(
            "TensorFlow CNN Defect Classifier"
        )

    else:

        st.warning(
            "AI Model Not Found"
        )

        st.caption(
            "Computer Vision will still work."
        )

        st.caption(
            "Check: pcb_defect_model.keras"
        )

    st.divider()

    st.subheader(
        "🔄 DETECTION PIPELINE"
    )

    st.write("01  Image Comparison")
    st.write("02  Difference Detection")
    st.write("03  Contour Detection")
    st.write("04  Defect Region Extraction")
    st.write("05  CNN AI Classification")

    st.divider()

    st.subheader("🛠 TECHNOLOGY")

    st.write("🐍 Python")
    st.write("👁 OpenCV")
    st.write("🤖 TensorFlow")
    st.write("🧠 CNN")
    st.write("🌐 Streamlit")

    st.divider()

    if st.button(
        "🔄 RESET INSPECTION",
        width="stretch"
    ):

        st.session_state.inspection_results = None

        st.session_state.reset_counter += 1

        st.rerun()


# =========================================================
# HERO HEADER
# =========================================================

st.title("🔬 PCB VISION")

st.subheader(
    "PCB Fault Detection & Inspection"
)

st.caption(
    "INTELLIGENT AI & COMPUTER VISION SYSTEM"
)

st.markdown("""
An intelligent PCB inspection platform that combines
**Computer Vision and Artificial Intelligence** to compare PCB images,
detect abnormal regions, identify potential manufacturing defects and
classify them using a trained **CNN-based AI model**.
""")

st.info(
    "👁 COMPUTER VISION   •   🤖 AI / ML   •   🧠 CNN   •   ⚡ AUTOMATED INSPECTION"
)


# =========================================================
# INSPECTION SECTION
# =========================================================

st.divider()

st.header("📁 PCB Inspection Workspace")

st.caption(
    "Upload a Reference PCB image and a Test PCB image. "
    "The system compares both images using OpenCV and classifies "
    "detected defect regions using Artificial Intelligence."
)


# =========================================================
# FILE UPLOAD
# =========================================================

col1, col2 = st.columns(2)


with col1:

    reference_file = st.file_uploader(

        "📘 Reference PCB Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key=f"reference_{st.session_state.reset_counter}"

    )


with col2:

    test_file = st.file_uploader(

        "📕 Test PCB Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key=f"test_{st.session_state.reset_counter}"

    )


# =========================================================
# IMAGE PROCESSING
# =========================================================

if reference_file is not None and test_file is not None:


    # =====================================================
    # READ REFERENCE IMAGE
    # =====================================================

    reference_bytes = np.asarray(

        bytearray(
            reference_file.getvalue()
        ),

        dtype=np.uint8

    )

    reference = cv2.imdecode(
        reference_bytes,
        cv2.IMREAD_COLOR
    )


    # =====================================================
    # READ TEST IMAGE
    # =====================================================

    test_bytes = np.asarray(

        bytearray(
            test_file.getvalue()
        ),

        dtype=np.uint8

    )

    test = cv2.imdecode(
        test_bytes,
        cv2.IMREAD_COLOR
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    if reference is None or test is None:

        st.error(
            "❌ Unable to read one or both images."
        )

        st.stop()


    # =====================================================
    # RESIZE TEST IMAGE
    # =====================================================

    if reference.shape[:2] != test.shape[:2]:

        test = cv2.resize(

            test,

            (
                reference.shape[1],
                reference.shape[0]
            )

        )


    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    st.divider()

    st.header("🖼 PCB Image Preview")

    st.caption(
        "Verify both PCB images before starting the AI-powered inspection."
    )

    image_col1, image_col2 = st.columns(2)


    with image_col1:

        st.subheader("📘 REFERENCE PCB")

        reference_rgb = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            reference_rgb,
            width="stretch"
        )


    with image_col2:

        st.subheader("📕 TEST PCB")

        test_rgb = cv2.cvtColor(
            test,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            test_rgb,
            width="stretch"
        )


    # =====================================================
    # START INSPECTION
    # =====================================================

    st.divider()

    left, center, right = st.columns([1, 2, 1])


    with center:

        detect = st.button(

            "🔍 START AI PCB INSPECTION",

            type="primary",

            width="stretch"

        )


    # =====================================================
    # DEFECT DETECTION
    # =====================================================

    if detect:


        start_time = time.time()


        with st.spinner(

            "🔬 Analyzing PCB using Computer Vision and AI..."

        ):


            # =================================================
            # GRAYSCALE
            # =================================================

            gray_reference = cv2.cvtColor(

                reference,

                cv2.COLOR_BGR2GRAY

            )


            gray_test = cv2.cvtColor(

                test,

                cv2.COLOR_BGR2GRAY

            )


            # =================================================
            # IMAGE DIFFERENCE
            # =================================================

            difference = cv2.absdiff(

                gray_reference,

                gray_test

            )


            # =================================================
            # THRESHOLD
            # =================================================

            _, threshold = cv2.threshold(

                difference,

                threshold_value,

                255,

                cv2.THRESH_BINARY

            )


            # =================================================
            # MORPHOLOGICAL PROCESSING
            # =================================================

            kernel = cv2.getStructuringElement(

                cv2.MORPH_RECT,

                (5, 5)

            )


            cleaned = cv2.morphologyEx(

                threshold,

                cv2.MORPH_OPEN,

                kernel

            )


            cleaned = cv2.morphologyEx(

                cleaned,

                cv2.MORPH_CLOSE,

                kernel

            )


            # =================================================
            # FIND CONTOURS
            # =================================================

            contours, _ = cv2.findContours(

                cleaned,

                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_SIMPLE

            )


            # =================================================
            # INITIALIZE RESULTS
            # =================================================

            defect_count = 0

            result = test.copy()

            detected_defects = []


            # =================================================
            # PROCESS CONTOURS
            # =================================================

            for contour in contours:


                area = cv2.contourArea(
                    contour
                )


                if area < min_area:

                    continue


                x, y, w, h = cv2.boundingRect(
                    contour
                )


                if w < 5 or h < 5:

                    continue


                defect_count += 1


                # =============================================
                # EXTRACT DEFECT REGION
                # =============================================

                defect_region = test[
                    y:y + h,
                    x:x + w
                ]


                # =============================================
                # AI CLASSIFICATION
                # =============================================

                defect_name, confidence = classify_defect(
                    defect_region
                )


                # =============================================
                # SEVERITY
                # =============================================

                severity = calculate_severity(
                    area,
                    confidence
                )


                # =============================================
                # STORE DEFECT
                # =============================================

                detected_defects.append({

                    "number": defect_count,

                    "type": defect_name,

                    "confidence": confidence,

                    "severity": severity,

                    "area": area,

                    "x": x,

                    "y": y,

                    "width": w,

                    "height": h

                })


                # =============================================
                # BOX COLOR
                # =============================================

                if severity == "HIGH":

                    box_color = (
                        0,
                        0,
                        255
                    )


                elif severity == "MEDIUM":

                    box_color = (
                        0,
                        255,
                        255
                    )


                else:

                    box_color = (
                        0,
                        255,
                        0
                    )


                # =============================================
                # DRAW BOX
                # =============================================

                cv2.rectangle(

                    result,

                    (x, y),

                    (x + w, y + h),

                    box_color,

                    3

                )


                # =============================================
                # LABEL
                # =============================================

                label = (
                    f"{defect_name} | "
                    f"{confidence:.1f}%"
                )


                text_y = max(
                    y - 10,
                    30
                )


                cv2.rectangle(

                    result,

                    (
                        x,
                        text_y - 25
                    ),

                    (
                        min(
                            x + 220,
                            result.shape[1] - 5
                        ),

                        text_y + 5
                    ),

                    box_color,

                    -1

                )


                cv2.putText(

                    result,

                    label,

                    (
                        x + 5,
                        text_y - 5
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    (
                        0,
                        0,
                        0
                    ),

                    1

                )


        # =====================================================
        # PROCESSING TIME
        # =====================================================

        end_time = time.time()

        processing_time = end_time - start_time


        # =====================================================
        # STATUS
        # =====================================================

        if defect_count == 0:

            status = "PASS"

        else:

            status = "DEFECT DETECTED"


        # =====================================================
        # SAVE RESULT IMAGE
        # =====================================================

        result_path = "outputs/result.jpg"

        cv2.imwrite(
            result_path,
            result
        )


        # =====================================================
        # SAVE SESSION RESULTS
        # =====================================================

        st.session_state.inspection_results = {

            "defect_count": defect_count,

            "status": status,

            "resolution":
                f"{reference.shape[1]} × "
                f"{reference.shape[0]}",

            "processing_time":
                processing_time,

            "detected_defects":
                detected_defects,

            "result":
                result,

            "difference":
                difference,

            "timestamp":

                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )

        }


        # =====================================================
        # HISTORY
        # =====================================================

        history_item = {

            "Inspection":

                len(
                    st.session_state.inspection_history
                ) + 1,


            "Date & Time":

                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),


            "Defects":

                defect_count,


            "Status":

                status,


            "Processing Time":

                f"{processing_time:.3f} sec"

        }


        st.session_state.inspection_history.append(
            history_item
        )


# =========================================================
# DISPLAY RESULTS
# =========================================================

if st.session_state.inspection_results is not None:


    inspection = st.session_state.inspection_results

    defect_count = inspection["defect_count"]

    status = inspection["status"]

    resolution = inspection["resolution"]

    processing_time = inspection["processing_time"]

    detected_defects = inspection["detected_defects"]

    result = inspection["result"]

    difference = inspection["difference"]


    # =====================================================
    # RESULTS DASHBOARD - CUSTOM CARD DESIGN
    # =====================================================

    st.divider()

    st.markdown("""
    <div class="dashboard-heading">📊 AI Inspection Dashboard</div>
    <div class="dashboard-subtitle">
        Computer Vision detection results combined with TensorFlow CNN-based AI classification.
    </div>
    """, unsafe_allow_html=True)

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.markdown(f"""
        <div class="dashboard-card card-red">
            <div class="card-label">🔴 DEFECTS FOUND</div>
            <div class="card-value">{defect_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with metric2:
        status_class = "card-green" if defect_count == 0 else "card-red"
        st.markdown(f"""
        <div class="dashboard-card {status_class}">
            <div class="card-label">📋 STATUS</div>
            <div class="card-value">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    with metric3:
        st.markdown(f"""
        <div class="dashboard-card card-blue">
            <div class="card-label">📐 RESOLUTION</div>
            <div class="card-value">{resolution}</div>
        </div>
        """, unsafe_allow_html=True)

    with metric4:
        st.markdown(f"""
        <div class="dashboard-card card-purple">
            <div class="card-label">⚡ PROCESSING TIME</div>
            <div class="card-value">{processing_time:.3f}s</div>
        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # STATUS ALERT
    # =====================================================

    if defect_count == 0:

        st.markdown("""
        <div class="dashboard-alert alert-pass">
            <div class="alert-title">✓ PCB INSPECTION PASSED</div>
            <div class="alert-text">No significant defect regions were detected.</div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="dashboard-alert alert-defect">
            <div class="alert-title">⚠ DEFECT DETECTED</div>
            <div class="alert-text">{defect_count} potential defect region(s) detected.</div>
        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # SEVERITY ANALYSIS
    # =====================================================

    if defect_count > 0:

        st.divider()

        st.markdown("""
        <div class="dashboard-heading" style="font-size:30px;">⚠️ Defect Severity Analysis</div>
        <div class="dashboard-subtitle">Severity distribution of all detected PCB defect regions.</div>
        """, unsafe_allow_html=True)

        severity_counts = Counter(
            defect["severity"]
            for defect in detected_defects
        )

        severity_col1, severity_col2, severity_col3 = st.columns(3)

        with severity_col1:
            st.markdown(f"""
            <div class="severity-card" style="border-top:3px solid #ef4444;">
                <div class="severity-label">🔴 HIGH SEVERITY</div>
                <div class="severity-value">{severity_counts.get("HIGH", 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with severity_col2:
            st.markdown(f"""
            <div class="severity-card" style="border-top:3px solid #facc15;">
                <div class="severity-label">🟡 MEDIUM SEVERITY</div>
                <div class="severity-value">{severity_counts.get("MEDIUM", 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with severity_col3:
            st.markdown(f"""
            <div class="severity-card" style="border-top:3px solid #22c55e;">
                <div class="severity-label">🟢 LOW SEVERITY</div>
                <div class="severity-value">{severity_counts.get("LOW", 0)}</div>
            </div>
            """, unsafe_allow_html=True)


    # =====================================================
    # AI CLASSIFICATION
    # =====================================================

    if defect_count > 0:

        st.divider()

        st.markdown("""
        <div class="dashboard-heading" style="font-size:30px;">🤖 AI Defect Classification</div>
        <div class="dashboard-subtitle">CNN-based classification results for every detected defect region.</div>
        """, unsafe_allow_html=True)

        for defect in detected_defects:

            st.markdown(
                f"""
                <div class="classification-card">
                    <div class="classification-title">
                        Defect {defect['number']} — {defect['type']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            severity_color = get_severity_color(defect["severity"])

            with col1:
                st.markdown(f"""
                <div class="classification-metric">
                    <div class="classification-label">🤖 AI CONFIDENCE</div>
                    <div class="classification-value">{defect['confidence']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="classification-metric">
                    <div class="classification-label">⚠ SEVERITY</div>
                    <div class="classification-value" style="color:{severity_color};">{defect['severity']}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="classification-metric">
                    <div class="classification-label">📐 REGION AREA</div>
                    <div class="classification-value">{defect['area']:.0f} px</div>
                </div>
                """, unsafe_allow_html=True)


    # =====================================================
    # DEFECT DETAILS
    # =====================================================

    if defect_count > 0:


        st.divider()

        st.header("📋 Defect Details")


        table_data = []


        for defect in detected_defects:


            table_data.append({

                "Defect No.":
                    defect["number"],


                "Defect Type":
                    defect["type"],


                "AI Confidence":
                    f'{defect["confidence"]:.2f}%',


                "Severity":
                    defect["severity"],


                "Area (pixels)":
                    f'{defect["area"]:.0f}',


                "Location":
                    f'({defect["x"]}, '
                    f'{defect["y"]})',


                "Bounding Box":
                    f'{defect["width"]} × '
                    f'{defect["height"]}'

            })


        defect_dataframe = pd.DataFrame(
            table_data
        )


        st.dataframe(

            defect_dataframe,

            width="stretch",

            hide_index=True

        )


    # =====================================================
    # DEFECT STATISTICS
    # =====================================================

    if defect_count > 0:


        st.divider()

        st.header("📊 Defect Statistics")


        defect_types = Counter(

            defect["type"]

            for defect in detected_defects

        )


        statistics_dataframe = pd.DataFrame({

            "Defect Type":

                list(defect_types.keys()),


            "Count":

                list(defect_types.values())

        })


        st.bar_chart(

            statistics_dataframe,

            x="Defect Type",

            y="Count"

        )


    # =====================================================
    # ANALYTICS
    # =====================================================

    if defect_count > 0:


        st.divider()

        st.header("📈 Inspection Analytics")


        analytics_col1, analytics_col2 = \
            st.columns(2)


        # AI CONFIDENCE

        with analytics_col1:


            st.subheader(
                "🤖 AI Confidence Analysis"
            )


            confidence_dataframe = pd.DataFrame({

                "Defect":

                    [

                        f"Defect {d['number']}"

                        for d in detected_defects

                    ],


                "Confidence":

                    [

                        d["confidence"]

                        for d in detected_defects

                    ]

            })


            st.bar_chart(

                confidence_dataframe,

                x="Defect",

                y="Confidence"

            )


        # DEFECT AREA

        with analytics_col2:


            st.subheader(
                "📐 Defect Area Analysis"
            )


            area_dataframe = pd.DataFrame({

                "Defect":

                    [

                        f"Defect {d['number']}"

                        for d in detected_defects

                    ],


                "Area":

                    [

                        d["area"]

                        for d in detected_defects

                    ]

            })


            st.bar_chart(

                area_dataframe,

                x="Defect",

                y="Area"

            )


    # =====================================================
    # DEFECT VISUALIZATION
    # =====================================================

    st.divider()

    st.header(
        "🔎 AI Defect Visualization"
    )


    st.caption(

        "Detected defect regions are highlighted "
        "using bounding boxes and classified by the AI model."

    )


    result_rgb = cv2.cvtColor(

        result,

        cv2.COLOR_BGR2RGB

    )


    st.image(

        result_rgb,

        width="stretch"

    )


    # =====================================================
    # DIFFERENCE MAP
    # =====================================================

    with st.expander(

        "🔬 View Image Difference Analysis"

    ):


        st.write(

            "The difference image shows pixel-level "
            "variations between the Reference PCB "
            "and Test PCB."

        )


        st.image(

            difference,

            caption="Image Difference Map",

            width="stretch"

        )


    # =====================================================
    # DOWNLOAD SECTION
    # =====================================================

    st.divider()

    st.header(
        "⬇️ Download Inspection Results"
    )


    download_col1, download_col2 = \
        st.columns(2)


    # =====================================================
    # DOWNLOAD RESULT IMAGE
    # =====================================================

    with download_col1:


        result_bytes = image_to_bytes(
            result
        )


        if result_bytes is not None:


            st.download_button(

                label=
                    "🖼 DOWNLOAD RESULT IMAGE",


                data=
                    result_bytes,


                file_name=
                    "PCB_AI_Inspection_Result.jpg",


                mime=
                    "image/jpeg",


                width="stretch"

            )


    # =====================================================
    # DOWNLOAD PDF
    # =====================================================

    with download_col2:


        try:


            pdf_report = generate_pdf_report(

                defect_count,

                status,

                resolution,

                processing_time,

                detected_defects,

                result

            )


            st.download_button(

                label=
                    "📄 DOWNLOAD PDF REPORT",


                data=
                    pdf_report,


                file_name=
                    "PCB_Inspection_Report.pdf",


                mime=
                    "application/pdf",


                width="stretch"

            )


        except Exception:


            st.warning(

                "PDF report could not be generated. "
                "Install reportlab if required."

            )


# =========================================================
# EMPTY STATE
# =========================================================

else:


    st.info(

        "👆 Upload both Reference PCB and Test PCB "
        "images to begin AI-powered automated inspection."

    )


# =========================================================
# INSPECTION HISTORY
# =========================================================

if len(st.session_state.inspection_history) > 0:


    st.divider()

    st.header(
        "🕒 Inspection History"
    )


    st.caption(

        "History of PCB inspections performed "
        "during the current application session."

    )


    history_dataframe = pd.DataFrame(

        st.session_state.inspection_history

    )


    st.dataframe(

        history_dataframe,

        width="stretch",

        hide_index=True

    )


    history_csv = history_dataframe.to_csv(
        index=False
    )


    st.download_button(

        label=
            "⬇️ DOWNLOAD INSPECTION HISTORY",


        data=
            history_csv,


        file_name=
            "PCB_Inspection_History.csv",


        mime=
            "text/csv",


        width="stretch"

    )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.header("💡 How It Works")


step1, step2, step3, step4, step5 = \
    st.columns(5)


with step1:

    st.info(

        "📤 **1. Upload**\n\n"
        "Upload Reference and Test PCB images."

    )


with step2:

    st.info(

        "👁 **2. Compare**\n\n"
        "OpenCV compares both PCB images."

    )


with step3:

    st.info(

        "🔎 **3. Detect**\n\n"
        "Image differences and contours are detected."

    )


with step4:

    st.info(

        "🤖 **4. Classify**\n\n"
        "CNN AI classifies defect regions."

    )


with step5:

    st.info(

        "📊 **5. Report**\n\n"
        "Results, analytics and PDF report are generated."

    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander(
    "ℹ️ About This Project"
):


    st.markdown("""

### PCB Fault Detection & Inspection

This project is an AI-powered PCB inspection system designed to
automatically detect possible differences and defect regions between
a reference PCB image and a test PCB image.

### Technologies Used

- Python
- OpenCV
- TensorFlow
- Convolutional Neural Network (CNN)
- NumPy
- Pandas
- Streamlit

### Working Principle

The Computer Vision module compares the Reference PCB and Test PCB
images using image difference techniques. Contours are extracted from
significant differences and each detected region is passed to a trained
CNN model for AI-based defect classification.

""")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🔬 PCB VISION | PCB Fault Detection & Inspection System"
)

st.caption(
    "AI-Powered Computer Vision • TensorFlow CNN"
)

st.caption(
    "Built with Python • OpenCV • TensorFlow • CNN • Streamlit"
)
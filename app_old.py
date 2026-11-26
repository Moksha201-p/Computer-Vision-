from pathlib import Path

code = r'''import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

from modules.enhancement import ZeroDCEEnhancer
from modules.evaluation import ImageQualityEvaluator


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LUMINA AI | Low-Light Image Enhancement",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL WHITE THEME
# =========================================================
st.markdown("""
<style>
:root {
    --navy: #17365D;
    --blue: #2563EB;
    --pale: #EFF6FF;
    --ink: #172033;
    --muted: #64748B;
    --line: #E2E8F0;
}

.stApp {
    background: #F7F9FC;
    color: var(--ink);
}
[data-testid="stHeader"] {
    background: rgba(247,249,252,0.92);
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5EAF1;
}
[data-testid="stSidebar"] * {
    color: #26364B;
}
h1, h2, h3, h4, p, label, .stMarkdown {
    color: var(--ink);
}
.hero {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #2563EB;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 4px 0 24px 0;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.045);
}
.hero h1 {
    color: #17365D !important;
    font-size: 34px;
    font-weight: 750;
    margin: 0 0 8px 0;
}
.hero p {
    color: #64748B !important;
    font-size: 15px;
    margin: 0;
}
.eyebrow {
    color: #2563EB !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 13px;
    padding: 17px 18px;
    min-height: 105px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
}
.metric-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .45px;
}
.metric-value {
    color: #17365D;
    font-size: 25px;
    font-weight: 750;
    margin-top: 8px;
}
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 13px;
    padding: 20px;
    margin-bottom: 14px;
}
div.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: 1px solid #2563EB;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 650;
}
div.stButton > button:hover {
    background: #1D4ED8;
    color: #FFFFFF;
    border-color: #1D4ED8;
}
div[data-testid="stDownloadButton"] > button {
    background: #FFFFFF;
    color: #1D4ED8;
    border: 1px solid #2563EB;
    border-radius: 9px;
    font-weight: 650;
}
hr {
    border-color: #E2E8F0;
}
.footer {
    color: #94A3B8;
    text-align: center;
    padding: 22px 0 8px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND EVALUATOR
# =========================================================
@st.cache_resource
def load_model():
    return ZeroDCEEnhancer()


evaluator = ImageQualityEvaluator()


# =========================================================
# SESSION STATE
# =========================================================
if "results" not in st.session_state:
    st.session_state.results = None

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def analyze_illumination(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness < 40:
        condition = "Extremely low light"
        recommendation = "Strong enhancement is recommended."
    elif brightness < 85:
        condition = "Low light"
        recommendation = "AI enhancement is recommended."
    elif brightness < 130:
        condition = "Moderate light"
        recommendation = "Moderate enhancement may be useful."
    else:
        condition = "Normal light"
        recommendation = "The image already has reasonable illumination."

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "condition": condition,
        "recommendation": recommendation
    }


def metric_value(metrics, key):
    """Safely read a metric from the evaluator's returned dictionary."""
    value = metrics.get(key, None) if isinstance(metrics, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def show_metric_card(label, value, suffix=""):
    display = "—" if value is None else f"{value:.2f}{suffix}"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_histogram(original_rgb, enhanced_rgb):
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(original_gray.ravel(), bins=256, range=(0, 256),
            alpha=0.48, label="Original", density=True)
    ax.hist(enhanced_gray.ravel(), bins=256, range=(0, 256),
            alpha=0.48, label="Enhanced", density=True)
    ax.set_title("Pixel intensity distribution")
    ax.set_xlabel("Grayscale intensity (0–255)")
    ax.set_ylabel("Normalized frequency")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_header(section, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{section}</div>
            <h1>🌙 LUMINA AI</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 14px 0;">
            <div style="font-size:25px;font-weight:800;color:#17365D;">🌙 LUMINA AI</div>
            <div style="font-size:13px;color:#64748B;">Intelligent Image Enhancement</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Enhancement Studio",
            "Image Analytics",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**SYSTEM INFORMATION**")
    st.caption("AI model: Zero-DCE++")
    st.caption("Execution: CPU")
    st.caption("Framework: PyTorch")
    st.caption("Project: Computer Vision Capstone")


# =========================================================
# DASHBOARD HOME
# =========================================================
if page == "Dashboard":
    render_header(
        "AI-POWERED IMAGE PROCESSING",
        "Adaptive low-light image enhancement and visual quality optimization."
    )

    st.subheader("Welcome to your image enhancement workspace")
    st.write(
        "Improve the visibility of photographs captured in poor lighting. "
        "Upload an image in the Enhancement Studio, run the pretrained "
        "Zero-DCE++ model, and review the measurements in Image Analytics."
    )

    total_processed = len(st.session_state.history)
    last_file = (
        st.session_state.results["filename"]
        if st.session_state.results is not None else "No image processed yet"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric_card("AI architecture", "Zero-DCE++")
    with c2:
        show_metric_card("Execution device", "CPU")
    with c3:
        show_metric_card("Images processed", str(total_processed))

    st.markdown("### Workspace")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#17365D;">✨ Enhancement Studio</h3>
                <p style="color:#64748B;">Upload a JPG, JPEG or PNG image, inspect its illumination,
                choose the enhancement blend intensity, and generate an enhanced result.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#17365D;">📊 Image Analytics</h3>
                <p style="color:#64748B;">Review brightness, contrast, entropy when available,
                processing time, and the original-versus-enhanced intensity histogram.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(f"Latest image: {last_file}")
    if st.session_state.results is None:
        st.info("To begin, open Enhancement Studio and process a low-light image.")


# =========================================================
# ENHANCEMENT STUDIO
# =========================================================
elif page == "Enhancement Studio":
    render_header(
        "ENHANCEMENT WORKSPACE",
        "Upload a low-light photograph and enhance it using the pretrained Zero-DCE++ model."
    )

    uploaded_file = st.file_uploader(
        "Choose a low-light image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

    if uploaded_file is not None:
        image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        original_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if original_bgr is None:
            st.error("The selected image could not be read. Please try another image.")
            st.stop()

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        analysis = analyze_illumination(original_bgr)
        height, width = original_bgr.shape[:2]

        st.markdown("### Image inspection")
        a, b, c = st.columns(3)
        a.metric("Brightness", f"{analysis['brightness']:.2f}/255")
        b.metric("Contrast", f"{analysis['contrast']:.2f}")
        c.metric("Resolution", f"{width} × {height}")

        st.info(
            f"Lighting condition: **{analysis['condition']}** — "
            f"{analysis['recommendation']}"
        )

        with st.expander("Preview uploaded image", expanded=True):
            st.image(original_rgb, caption=uploaded_file.name, use_container_width=True)

        st.markdown("### Enhancement settings")
        intensity = st.slider(
            "Enhancement blend intensity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="0 keeps the original image; 1 uses the full Zero-DCE++ output. "
                 "Intermediate values blend the two images."
        )

        if st.button("Run AI enhancement", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading Zero-DCE++ model..."):
                    model = load_model()

                with st.spinner("Enhancing image. Please wait..."):
                    start_time = time.perf_counter()
                    ai_result = model.enhance(original_bgr)
                    processing_time = time.perf_counter() - start_time

                    final_result = cv2.addWeighted(
                        original_bgr,
                        1.0 - intensity,
                        ai_result,
                        intensity,
                        0
                    )

                original_metrics = evaluator.evaluate_image(original_bgr)
                enhanced_metrics = evaluator.evaluate_image(final_result)
                enhanced_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

                # Keep the complete result in session state so Analytics
                # can display it after the user changes pages.
                st.session_state.results = {
                    "original": original_rgb.copy(),
                    "enhanced": enhanced_rgb.copy(),
                    "metrics_original": original_metrics,
                    "metrics_enhanced": enhanced_metrics,
                    "time": processing_time,
                    "filename": uploaded_file.name,
                    "intensity": intensity
                }

                st.session_state.history.append({
                    "filename": uploaded_file.name,
                    "brightness": metric_value(enhanced_metrics, "Brightness"),
                    "processing_time": processing_time
                })

                st.success("Enhancement completed successfully.")
            except Exception as error:
                st.error(f"Enhancement failed: {error}")

    # Show the most recently processed image even after a page rerun.
    if st.session_state.results is not None:
        result = st.session_state.results
        st.divider()
        st.markdown("### Before and after comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(result["original"], caption="Original image", use_container_width=True)
        with col2:
            st.image(result["enhanced"], caption="Enhanced image", use_container_width=True)

        st.markdown("### Split comparison")
        comparison = st.slider(
            "Move the divider",
            min_value=0,
            max_value=100,
            value=50,
            key="studio_comparison"
        )
        original_array = result["original"]
        enhanced_array = result["enhanced"]
        split = int(original_array.shape[1] * comparison / 100)
        comparison_image = original_array.copy()
        comparison_image[:, :split] = enhanced_array[:, :split]
        st.image(
            comparison_image,
            caption="Enhanced on the left | Original on the right",
            use_container_width=True
        )

        png_bytes = Image.fromarray(result["enhanced"]).convert("RGB")
        import io
        buffer = io.BytesIO()
        png_bytes.save(buffer, format="PNG")
        st.download_button(
            "Download enhanced image",
            data=buffer.getvalue(),
            file_name="lumina_ai_enhanced.png",
            mime="image/png",
            use_container_width=True
        )


# =========================================================
# IMAGE ANALYTICS
# =========================================================
elif page == "Image Analytics":
    render_header(
        "IMAGE QUALITY REPORT",
        "Compare measurable image properties before and after AI enhancement."
    )

    result = st.session_state.results

    if result is None:
        st.info(
            "No analysis is available yet. Open Enhancement Studio, upload an image, "
            "and click **Run AI enhancement**. Your analytics will appear here."
        )
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#17365D;">What will appear here?</h3>
                <p style="color:#64748B;">Brightness, contrast, entropy (if returned by the
                evaluator), processing time, and an original-versus-enhanced pixel
                intensity histogram.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        original_metrics = result["metrics_original"]
        enhanced_metrics = result["metrics_enhanced"]

        st.markdown(f"**Analysed image:** {result['filename']}")
        st.caption(
            f"Enhancement blend intensity: {result['intensity']:.2f} · "
            f"AI processing time: {result['time']:.3f} seconds"
        )

        # Read only metrics actually returned by the evaluation module.
        metric_keys = ["Brightness", "Contrast", "Entropy"]
        titles = ["Brightness", "Contrast", "Entropy"]

        st.markdown("### Before and after measurements")
        cols = st.columns(3)
        for col, key, title in zip(cols, metric_keys, titles):
            before = metric_value(original_metrics, key)
            after = metric_value(enhanced_metrics, key)
            with col:
                show_metric_card(f"{title} · Original", before)
            # Add enhanced values below each original value in a compact card.
        cols2 = st.columns(3)
        for col, key, title in zip(cols2, metric_keys, titles):
            after = metric_value(enhanced_metrics, key)
            with col:
                show_metric_card(f"{title} · Enhanced", after)

        st.markdown("### Change summary")
        summary_cols = st.columns(3)
        for col, key, title in zip(summary_cols, metric_keys, titles):
            before = metric_value(original_metrics, key)
            after = metric_value(enhanced_metrics, key)
            if before is None or after is None:
                delta_text = "Not available"
            else:
                delta_text = f"{after - before:+.2f}"
            with col:
                st.metric(f"{title} change", delta_text)

        st.markdown("### Pixel intensity distribution")
        fig = make_histogram(result["original"], result["enhanced"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("### Visual comparison")
        image_col1, image_col2 = st.columns(2)
        with image_col1:
            st.image(result["original"], caption="Original", use_container_width=True)
        with image_col2:
            st.image(result["enhanced"], caption="Enhanced", use_container_width=True)

        st.markdown("### Processing information")
        p1, p2 = st.columns(2)
        p1.metric("AI processing time", f"{result['time']:.3f} s")
        p2.metric("Enhancement blend intensity", f"{result['intensity']:.2f}")

        st.caption(
            "Note: Brightness, contrast and entropy are descriptive image statistics. "
            "PSNR and SSIM are not shown because this dashboard does not currently "
            "calculate them against a paired reference image."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":
    render_header(
        "PROJECT INFORMATION",
        "Adaptive Low-Light Image Enhancement and Visual Quality Optimization Using Artificial Intelligence."
    )

    st.subheader("Project overview")
    st.write(
        "LUMINA AI is a computer vision application that improves the visibility "
        "of images captured under insufficient lighting. It uses a pretrained "
        "Zero-DCE++ model to generate an enhanced image."
    )

    st.markdown("### Technology stack")
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color:#17365D;">Python</h4>
                <p style="color:#64748B;">Main programming language.</p>
                <h4 style="color:#17365D;">Streamlit</h4>
                <p style="color:#64748B;">Interactive application interface.</p>
                <h4 style="color:#17365D;">PyTorch</h4>
                <p style="color:#64748B;">Loads and runs the pretrained deep learning model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with tech_col2:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color:#17365D;">Zero-DCE++</h4>
                <p style="color:#64748B;">Low-light image enhancement model.</p>
                <h4 style="color:#17365D;">OpenCV</h4>
                <p style="color:#64748B;">Image loading and processing operations.</p>
                <h4 style="color:#17365D;">NumPy / Matplotlib</h4>
                <p style="color:#64748B;">Numerical calculations and analytics visualization.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Main workflow")
    st.info(
        "Image upload → preprocessing and illumination analysis → "
        "Zero-DCE++ enhancement → image-quality measurements → "
        "visual comparison and download."
    )

    st.warning(
        "Implementation note: this application uses a pretrained Zero-DCE++ model "
        "for inference; it does not train the model from scratch."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">LUMINA AI · Computer Vision Capstone Project</div>',
    unsafe_allow_html=True
)
'''
out = Path("/mnt/data/LUMINA_AI_professional_app.py")
out.write_text(code, encoding="utf-8")
print(f"Created {out} ({out.stat().st_size} bytes)")

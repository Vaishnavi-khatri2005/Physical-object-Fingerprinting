"""
ShoePrint: Instance-Level Physical Object Identification
Interactive Streamlit Application with Direct Smartphone Camera Capture

Uses the validated RootSIFT + RANSAC matching engine against the local shoe gallery.
Applies the experimentally determined EER verification threshold (22 inliers)
to reject unknown or unconfident matches.

Features:
- Direct Smartphone Camera Capture (st.camera_input)
- Fallback File Upload (st.file_uploader)
- Quick-test Sample Picker from gallery
- Complete Match Results, Top 3 Candidates, Visual Correspondences & Inlier metrics

Usage:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import discover_shoe_images
from evaluate_rootsift import transform_to_rootsift
from src.config import DEFAULT_OUTPUTS_DIR
from src.features import SIFTFeatureExtractor
from src.matcher import ImageMatchResult, SIFTRansacMatcher, aggregate_shoe_scores

# Experimentally determined EER cutoff threshold from biometric evaluation
EER_REJECTION_THRESHOLD = 22

# Page configuration
st.set_page_config(
    page_title="ShoePrint — Instance-Level Object Identification",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading registered shoe gallery features...")
def load_gallery_features(dataset_path: str) -> Tuple[Dict[str, List[Path]], Dict[Path, Tuple[List, np.ndarray, str]]]:
    """
    Loads and caches RootSIFT features for all registered gallery images.
    Caches in memory so subsequent query matches execute in under 1 second.
    """
    dataset_dir = Path(dataset_path).resolve()
    shoe_dict = discover_shoe_images(dataset_dir)

    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=2500, cache_dir=cache_dir)

    ref_features = {}
    for shoe_id, img_paths in shoe_dict.items():
        for p in img_paths:
            kp, desc = extractor.extract(p, use_cache=True)
            desc_root = transform_to_rootsift(desc)
            ref_features[p] = (kp, desc_root, shoe_id)

    return shoe_dict, ref_features


def render_feature_correspondence(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    match_result: ImageMatchResult,
) -> np.ndarray:
    """Generates side-by-side match image with green connecting lines for RANSAC inliers."""
    matches_mask = None
    if match_result.inlier_mask is not None and match_result.inlier_count > 0:
        matches_mask = match_result.inlier_mask.ravel().tolist()

    draw_params = dict(
        matchColor=(34, 197, 94),     # Vibrant green lines
        singlePointColor=None,
        matchesMask=matches_mask,      # Only draw verified inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    canvas_bgr = cv2.drawMatches(
        query_bgr,
        match_result.query_keypoints,
        ref_bgr,
        match_result.reference_keypoints,
        match_result.good_matches,
        None,
        **draw_params,
    )
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def main():
    # --- Sidebar: System Information & Status ---
    with st.sidebar:
        st.title("👟 ShoePrint")
        st.markdown("**Instance-Level Physical Object Fingerprinting**")
        st.markdown("---")

        st.markdown("### 📊 Benchmark Performance")
        st.markdown(
            """
            **RootSIFT Baseline Evaluation:**
            * **Top-1 Accuracy:** `78.3%`
            * **Top-3 Accuracy:** `93.5%`
            * **EER Threshold:** `22 inliers`
            * **Equal Error Rate:** `30.4%`
            
            *(Evaluated on 7 physical shoes, 154 photos, seed=42 protocol).*
            """
        )
        st.caption("⚠️ Experimental evaluation results on the current 7-shoe dataset; not a production accuracy claim.")

        st.markdown("---")
        st.markdown("### ℹ️ Research Prototype Notice")
        st.info(
            "This is a research prototype evaluated on a small controlled dataset. "
            "Performance may change with different cameras, lighting, viewpoints, surfaces and objects."
        )

        st.markdown("---")
        st.markdown("### ⚙️ Matching Parameters")
        threshold_override = st.slider(
            "Rejection Threshold (Inliers)",
            min_value=5,
            max_value=60,
            value=EER_REJECTION_THRESHOLD,
            help="Minimum geometrically verified inlier score required to accept an identification match."
        )

        st.markdown("---")
        st.markdown("### 📱 Mobile Access")
        st.markdown(
            """
            To use this on your phone:
            1. Ensure phone & PC are on the same Wi-Fi.
            2. Run: `streamlit run app.py --server.address 0.0.0.0`
            3. Open `http://<laptop-ip>:8501` on mobile browser.
            """
        )

    # --- Header Banner ---
    st.markdown("# SHOEPRINT")
    st.markdown("### *Instance-Level Object Identification*")
    st.markdown(
        "Capture or upload a photograph of a physical shoe to identify its exact individual instance "
        "using surface micro-features, creasing, stitching, and wear patterns."
    )
    st.markdown("---")

    # Load gallery
    gallery_dir = "photos"
    if not Path(gallery_dir).exists():
        st.error(f"Gallery directory '{gallery_dir}' not found. Please ensure the 'photos/' folder is present.")
        return

    shoe_dict, ref_features = load_gallery_features(gallery_dir)

    # --- Section 1: Capture or Upload Shoe Image ---
    st.subheader("1. Input Shoe Image")

    input_tab1, input_tab2 = st.tabs(["📷 Take a photo", "📁 Or upload an image"])

    camera_image = None
    uploaded_file = None

    with input_tab1:
        st.markdown("Point your smartphone camera directly at the shoe:")
        camera_image = st.camera_input("Take a photo of the shoe")

    with input_tab2:
        uploaded_file = st.file_uploader(
            "Choose a shoe photo (JPEG, PNG, WEBP)...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photograph from your device or gallery."
        )

    # Determine active query input
    active_query = None
    query_source_label = ""
    is_path_query = False

    if camera_image is not None:
        active_query = camera_image
        query_source_label = "Camera Capture"
    elif uploaded_file is not None:
        active_query = uploaded_file
        query_source_label = getattr(uploaded_file, "name", "Uploaded Image")
    else:
        # Fallback quick-test buttons from gallery
        st.markdown("##### Or pick a sample from the dataset to test:")
        sample_cols = st.columns(3)
        sample_paths = [
            ("shoe1 (Sample)", "photos/shoe1/WhatsApp Image 2026-10-03 at 10.29.40 PM (1).jpeg"),
            ("shoe4 (Sample)", "photos/shoe4/WhatsApp Image 2026-10-03 at 10.29.49 PM (1).jpeg"),
            ("shoe6 (Sample)", "photos/shoe6/WhatsApp Image 2026-10-03 at 10.29.16 PM (1).jpeg"),
        ]
        for idx, (label, s_path) in enumerate(sample_paths):
            if Path(s_path).exists():
                with sample_cols[idx]:
                    if st.button(f"Load {label}", key=f"btn_sample_{idx}"):
                        st.session_state["sample_file"] = s_path
                        st.rerun()

        if "sample_file" in st.session_state and Path(st.session_state["sample_file"]).exists():
            active_query = Path(st.session_state["sample_file"])
            query_source_label = active_query.name
            is_path_query = True
        else:
            st.info("👆 Use the camera above or upload a photograph to begin identification.")
            return

    # Process query
    t_start = time.time()

    # Load PIL query respecting EXIF
    if is_path_query and isinstance(active_query, Path):
        pil_img = Image.open(active_query)
    else:
        pil_img = Image.open(active_query)

    pil_img = ImageOps.exif_transpose(pil_img)
    query_rgb = np.array(pil_img.convert("RGB"))
    query_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
    query_bgr = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)

    # Extract query RootSIFT features
    sift = cv2.SIFT_create(nfeatures=2500)
    q_kp, q_desc = sift.detectAndCompute(query_gray, None)

    if q_desc is None or len(q_kp) == 0:
        st.error("No distinct features could be extracted from this image. Please capture a clearer, well-lit photo.")
        return

    q_desc_root = transform_to_rootsift(q_desc)

    matcher = SIFTRansacMatcher(ratio_thresh=0.75, ransac_reproj_thresh=5.0)

    # Match query against all registered gallery reference images
    match_results = []
    for r_path, (r_kp, r_desc_root, ref_shoe_id) in ref_features.items():
        # Prevent self-matching if testing directly from gallery file
        if is_path_query and isinstance(active_query, Path) and r_path.resolve() == active_query.resolve():
            continue

        res = matcher.match_pair(
            query_kp=q_kp,
            query_desc=q_desc_root,
            ref_kp=r_kp,
            ref_desc=r_desc_root,
            shoe_id=ref_shoe_id,
            ref_path=r_path,
        )
        match_results.append(res)

    ranked_shoes = aggregate_shoe_scores(match_results)
    total_latency = time.time() - t_start

    # Top match details
    best_candidate = ranked_shoes[0] if ranked_shoes else None
    top1_score = best_candidate.score if best_candidate else 0
    top1_shoe = best_candidate.shoe_id if best_candidate else "none"

    is_accepted = (top1_score >= threshold_override)

    st.markdown("---")

    # --- Section 2: Identification Result ---
    st.subheader("2. IDENTIFICATION RESULT")

    if is_accepted:
        st.success(
            f"### ✅ Top Match: **{top1_shoe.upper()}**\n\n"
            f"**Score:** `{top1_score}` verified inliers &nbsp;|&nbsp; "
            f"**Decision:** `MATCH` (Threshold: `{threshold_override}`)"
        )
    else:
        st.error(
            f"### 🛑 Decision: UNKNOWN / NO CONFIDENT MATCH\n\n"
            f"**Top Candidate:** `{top1_shoe}` (Score: `{top1_score}` inliers) &nbsp;|&nbsp; "
            f"**Decision:** `REJECTED` (Required Threshold: `{threshold_override}` inliers)\n\n"
            "The inlier score is below the operational threshold. The query is rejected rather than assigned falsely."
        )

    # --- Query vs Best Match Reference Images ---
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("#### 📷 Captured Image")
        st.image(query_rgb, use_container_width=True, caption=f"Source: {query_source_label}")

    with img_col2:
        if best_candidate and best_candidate.best_reference_image.exists():
            ref_pil = ImageOps.exif_transpose(Image.open(best_candidate.best_reference_image))
            st.markdown(f"#### 👟 Best Reference Match: `{best_candidate.shoe_id}`")
            st.image(
                ref_pil,
                use_container_width=True,
                caption=f"Reference: {best_candidate.best_reference_image.name} (Score: {top1_score} inliers)"
            )
        else:
            st.markdown("#### 👟 Best Reference Match")
            st.write("No matching reference image found.")

    st.markdown("---")

    # --- Section 3: Top Matches ---
    st.subheader("3. Top 3 Candidates")
    top_cols = st.columns(3)

    for rank_idx in range(min(3, len(ranked_shoes))):
        shoe_score = ranked_shoes[rank_idx]
        with top_cols[rank_idx]:
            is_winner = (rank_idx == 0 and is_accepted)
            badge = "🥇 Best Match" if rank_idx == 0 else f"#{rank_idx + 1} Candidate"
            card_icon = "🟢" if is_winner else ("🟡" if rank_idx == 1 else "⚪")

            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 10px;">
                    <div style="font-size: 13px; color: #64748b; font-weight: bold;">{badge}</div>
                    <div style="font-size: 22px; font-weight: bold; color: #0f172a; margin: 4px 0;">{card_icon} {shoe_score.shoe_id}</div>
                    <div style="font-size: 15px; color: #334155;">Score: <b>{shoe_score.score}</b> inliers</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Ref: {shoe_score.best_reference_image.name[:22]}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Section 4: Visual Feature Correspondence ---
    st.subheader("4. Visual Correspondence")
    st.markdown(
        "Green connecting lines indicate spatial feature correspondences verified by RANSAC homography "
        "between the query photograph and the top gallery candidate:"
    )

    if best_candidate and best_candidate.score > 0:
        best_ref_path = best_candidate.best_reference_image
        with Image.open(best_ref_path) as ref_img_raw:
            ref_rgb = np.array(ImageOps.exif_transpose(ref_img_raw).convert("RGB"))
            ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2RGB) # OpenCV expects BGR for drawMatches
            ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

        vis_rgb = render_feature_correspondence(query_bgr, ref_bgr, best_candidate.best_match_result)
        st.image(
            vis_rgb,
            use_container_width=True,
            caption=f"Verified Inliers: {best_candidate.shoe_id} ({best_candidate.score} inliers)"
        )
    else:
        st.info("No spatial inliers found to draw connecting lines.")

    st.markdown("---")

    # --- Section 5: Processing Information ---
    st.subheader("5. Processing Information")
    proc_col1, proc_col2, proc_col3, proc_col4 = st.columns(4)

    with proc_col1:
        st.metric("Processing Time", f"{total_latency:.2f} s")
    with proc_col2:
        st.metric("Query Keypoints", f"{len(q_kp)}")
    with proc_col3:
        st.metric("Gallery Comparisons", f"{len(match_results)}")
    with proc_col4:
        decision_label = "MATCH ACCEPTED" if is_accepted else "REJECTED (UNKNOWN)"
        st.metric("Final Decision", decision_label)


if __name__ == "__main__":
    main()

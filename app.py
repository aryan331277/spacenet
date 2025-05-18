import streamlit as st
import torch
from PIL import Image
import os
import tempfile
from inference import load_controlnet_pipeline, preprocess_sketch, generate_city

# --- PAGE CONFIG ---
st.set_page_config(page_title="🌆 Sketch-to-City Generator", layout="wide")

# --- HEADER ---
st.title("🌆 Sketch-to-City Generator – Turn Your Sketch Into a Realistic City With AI")
st.markdown("Upload a color-coded or line sketch, and watch the AI bring it to life as a realistic city.")
st.markdown("Built during Project 17/52 | By Aryan")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("🛠 How It Works")
    st.markdown("""
    1. Upload a sketch (hand-drawn or digital)  
    2. AI uses ControlNet + Stable Diffusion to generate a realistic city  
    3. Download or share the output!
    """)
    st.image("assets/logo.png", use_column_width=True)

# --- MAIN UI ---
uploaded_file = st.file_uploader("📁 Upload Your Sketch (.png or .jpg)", type=["png", "jpg"])

if uploaded_file:
    # Save uploaded file to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.getvalue())
    temp_path = tfile.name

    # Show uploaded image
    st.success(f"✅ '{uploaded_file.name}' uploaded successfully.")

    # Preprocess sketch
    st.subheader("🔍 Processing Your Sketch...")
    try:
        processed_sketch = preprocess_sketch(temp_path)
        st.image(processed_sketch, caption="Processed Sketch (Canny Edge)", use_column_width=True)

        # Prompt Selection
        st.subheader("🎯 Select City Style")
        style_options = {
            "Modern": "modern city, skyscrapers, dense urban area",
            "Futuristic": "futuristic city, neon lights, flying cars, high-tech buildings",
            "Eco-Friendly": "green city, solar panels, parks, sustainable architecture",
            "Historic": "historic city, old buildings, narrow streets, cultural heritage"
        }

        selected_style = st.selectbox("Choose city generation style:", list(style_options.keys()))
        final_prompt = f"Satellite view of {style_options[selected_style]}"

        if st.button("🌆 Generate City"):
            st.subheader("🤖 Generating Realistic City View...")

            with st.spinner("Generating..."):
                # Load model
                pipe = load_controlnet_pipeline()

                # Run generation
                generated_city = generate_city(pipe, prompt=final_prompt, sketch=processed_sketch)

                # Show result
                st.image(generated_city, caption=f"Generated City ({selected_style})", use_column_width=True)

                # Download button
                st.download_button(
                    label="💾 Download Generated City",
                    data=generated_city.tobytes(),
                    file_name="generated_city.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"❌ Error processing sketch: {e}")
else:
    st.info("📂 Please upload a sketch to begin generating your city.")

# --- FOOTER ---
st.markdown("---")
st.markdown("#### 🔬 Built by Aryan · Part of Project 17/52 · v1 Demo")
st.markdown("Next version will include 3D toggle, pollution overlay, and traffic simulation.")

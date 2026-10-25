
import os
import time

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")
st.title("🎵 AI Music Generator")
st.write("Generate instrumental music from a text prompt.")

prompt = st.text_area(
    "Enter your music prompt",
    value="Relaxing instrumental music with piano, soft flute, a gentle melody, slow tempo and a peaceful atmosphere, no vocals."
)

duration = st.slider("Duration (seconds)", 5, 10, 5)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return processor, model, device


if st.button("Generate Music", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a music prompt.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    try:
        status.info("Step 1/3: Loading MusicGen model. First download may take time.")
        processor, model, device = load_model()
        progress.progress(25)

        status.info(f"Step 2/3: Preparing prompt on {device.upper()}.")
        inputs = processor(
            text=[prompt.strip()],
            padding=True,
            return_tensors="pt"
        ).to(device)
        progress.progress(40)

        status.info("Step 3/3: Generating music. This can take several minutes on CPU.")

        start = time.time()
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 50,
                do_sample=True
            )

        audio = audio_values[0, 0].float().cpu().numpy()
        sample_rate = model.config.audio_encoder.sampling_rate

        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        output_path = os.path.join(OUTPUT_DIR, "generated_music.wav")
        wavfile.write(output_path, sample_rate, audio_int16)

        progress.progress(100)
        status.success(
            f"Music generated in {time.time() - start:.1f} seconds!"
        )

        st.audio(output_path, format="audio/wav")

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Music",
                data=f.read(),
                file_name="generated_music.wav",
                mime="audio/wav"
            )

    except Exception as error:
        status.error(f"Music generation failed: {error}")
        st.exception(error)
        st.info(
            "If your computer runs out of memory or generation takes too long, "
            "use the Google Colab GPU notebook."
        )

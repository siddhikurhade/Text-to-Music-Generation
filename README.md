# AI Music Generator using MusicGen

## Aim
To generate novel music compositions from text prompts
using a pretrained MusicGen foundation model.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- Meta MusicGen

## Model
facebook/musicgen-small

## Features
- Text-to-music generation
- Multiple music styles
- Audio playback
- WAV audio download
- No dataset collection or model training

## Run the application

1. Create a virtual environment.
2. Install requirements.
3. Run the Streamlit application.

Commands:

pip install -r requirements.txt
streamlit run app.py

Note: Model inference may require a GPU or substantial RAM.
For low-memory computers, use Google Colab with a GPU.
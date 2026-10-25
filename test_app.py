import streamlit as st

st.set_page_config(page_title="Streamlit Test")

st.title("🎵 AI Music Generator")
st.write("The application interface is working!")
st.text_input("Enter a music prompt")
st.button("Generate Music")
# app.py - minimal Render debug version

import os
import sys
import socket
import streamlit as st

st.set_page_config(page_title="NTTV Chatbot – Debug", page_icon="🧭")

st.title("NTTV Chatbot – Render debug")
st.write("If you see this in your browser, the service is *actually* running.")

st.subheader("Environment sanity check")
st.write("Python:", sys.version)
st.write("Hostname:", socket.gethostname())

st.write("Relevant env vars:")
interesting = {k: v for k, v in os.environ.items()
               if k.startswith("OPENAI") or k.startswith("NTTV") or k.startswith("RENDER")}
st.json(interesting)

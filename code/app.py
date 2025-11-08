import streamlit as st
from langchain_core.messages import HumanMessage
from main import app
from dotenv import load_dotenv
import os

st.set_page_config(page_title="SQL Assistant", page_icon="💬", layout="centered")

st.title("💬 SQL Assistant")

# Champ texte utilisateur
user_input = st.text_area("Salut, je suis Albert ton assistant Data, tu cherches quelque chose en particulier dans ta base de données ?", placeholder="Ex: Show me 10 movies from 2021")

# Bouton d'envoi
if st.button("Envoyer"):
    if user_input.strip():
        with st.spinner("Analyse en cours..."):
            response = app.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
        st.markdown("### Réponse :")
        st.write(response["messages"][-1].content)
    else:
        st.warning("Veuillez entrer un message avant d'envoyer.")

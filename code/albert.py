"""
Interface Streamlit SIMPLE pour l'Agent SQL
Lancer avec : streamlit run app.py
"""

import streamlit as st
import os
import json
import sqlite3
import requests
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
st.set_page_config(page_title="Agent SQL", page_icon="🤖")
st.title("🤖 Agent SQL Netflix")

# === RÉCUPÉRATION DES CLÉS (depuis .env) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY manquant dans .env")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === ÉTAT AGENT ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    schema: str
    next_tool: str

# === OUTILS (ton code original) ===
@tool
def get_db_schema(db_path: str = "../data") -> str:
    """Récupère le schéma de toutes les bases SQLite du dossier."""
    result = {"databases": [], "error": None}
    
    try:
        db_files = [f for f in os.listdir(db_path) if f.endswith(('.db', '.sqlite', '.sqlite3'))]
    except FileNotFoundError:
        return json.dumps({"error": f"Dossier {db_path} introuvable"})
    
    if not db_files:
        return json.dumps({"error": "Aucune base SQLite trouvée"})
    
    for db_file in db_files:
        db_path_full = os.path.join(db_path, db_file)
        try:
            conn = sqlite3.connect(db_path_full)
            cursor = conn.cursor()
            
            database = {"name": db_file, "tables": []}
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
                database["tables"].append({"name": table_name, "columns": columns})
            
            result["databases"].append(database)
            conn.close()
        except Exception as e:
            result["databases"].append({"name": db_file, "error": str(e)})
    
    return json.dumps(result, indent=2)

@tool
def execute_sql_query(query: str, db_path: str = "../data/netflix.db") -> str:
    """Exécute une requête SQL sur netflix.db."""
    if not os.path.exists(db_path):
        return json.dumps({"error": f"Base {db_path} introuvable"})
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Erreur SQL: {str(e)}"})

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Recherche web via DuckDuckGo."""
    try:
        search = DuckDuckGoSearchResults(num_results=num_results)
        return search.run(query)
    except Exception as e:
        return json.dumps({"error": f"Erreur web search: {str(e)}"})

@tool
def omdb_api(by: str = "search", i: str = None, t: str = None, 
             s: str = None, y: str = None, plot: str = "short") -> str:
    """Interroge l'API OMDb pour infos films/séries."""
    if not OMDB_API_KEY:
        return json.dumps({"error": "Clé OMDB_API_KEY manquante"})
    
    params = {"apikey": OMDB_API_KEY, "plot": plot}
    
    if by == "id" and i:
        params["i"] = i
    elif by == "title" and t:
        params["t"] = t
    elif by == "search" and s:
        params["s"] = s
    else:
        return json.dumps({"error": "Paramètres manquants (i/t/s selon 'by')"})
    
    if y:
        params["y"] = y
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Erreur API OMDb: {str(e)}"})

# Lier les outils
tools = [execute_sql_query, web_search, omdb_api]
llm_with_tools = llm.bind_tools(tools)

# === NŒUDS (ton code original) ===
def get_schema_node(state: AgentState) -> dict:
    schema = get_db_schema.invoke({})
    return {"schema": schema, "messages": [AIMessage(content=f"✅ Schéma chargé")]}

def chief_agent_node(state: AgentState) -> dict:
    prompt = f"""Tu es un assistant SQL/données. Tu as accès à :
1. execute_sql_query : pour interroger netflix.db
2. web_search : pour recherches web générales
3. omdb_api : pour infos précises sur films/séries

Schéma disponible :
{state['schema']}

Analyse la requête utilisateur et choisis L'OUTIL le plus adapté."""
    
    messages = [{"role": "system", "content": prompt}] + [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]
    
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        next_tool = {
            "execute_sql_query": "sql",
            "web_search": "web",
            "omdb_api": "omdb"
        }.get(tool_name, "none")
        
        return {"next_tool": next_tool, "messages": [response]}
    else:
        return {"next_tool": "none", "messages": [response]}

def tool_executor_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [AIMessage(content="⚠️ Aucun outil à exécuter")]}
    
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    tool_map = {
        "execute_sql_query": execute_sql_query,
        "web_search": web_search,
        "omdb_api": omdb_api
    }
    
    tool_func = tool_map.get(tool_name)
    if not tool_func:
        return {"messages": [AIMessage(content=f"⚠️ Outil {tool_name} inconnu")]}
    
    try:
        result = tool_func.invoke(tool_args)
        return {"messages": [AIMessage(content=f"📊 Résultat {tool_name}:\n{result}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Erreur {tool_name}: {str(e)}")]}

def synthesize_node(state: AgentState) -> dict:
    prompt = """Tu es un assistant qui synthétise les résultats.
Fournis une réponse claire, concise et en français à l'utilisateur.
Utilise les données disponibles dans l'historique des messages."""
    
    messages = [{"role": "system", "content": prompt}] + [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def route_after_chief(state: AgentState) -> Literal["tool_executor", "synthesize"]:
    return "tool_executor" if state["next_tool"] != "none" else "synthesize"

# === CONSTRUCTION DU GRAPHE ===
@st.cache_resource
def build_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("chief_agent", chief_agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("synthesize", synthesize_node)
    
    workflow.add_edge(START, "get_schema")
    workflow.add_edge("get_schema", "chief_agent")
    workflow.add_conditional_edges(
        "chief_agent",
        route_after_chief,
        {"tool_executor": "tool_executor", "synthesize": "synthesize"}
    )
    workflow.add_edge("tool_executor", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

app = build_agent()

# === INTERFACE STREAMLIT SIMPLE ===

# Initialiser l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Exécuter l'agent
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "schema": "",
                "next_tool": ""
            }
            
            # Récupérer la réponse finale
            result = None
            for step in app.stream(inputs, stream_mode="values"):
                result = step
            
            # Afficher la réponse
            if result:
                final_message = result["messages"][-1]
                response_text = final_message.content
                st.markdown(response_text)
                
                # Sauvegarder dans l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
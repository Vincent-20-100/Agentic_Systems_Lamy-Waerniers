"""
SQL Agent with Sequential Flow: SQL → OMDB → Web → Synthesis
V3.1 Simplified: Removed HITL/Clarification loop only
Launch: streamlit run albert_v3.1_simplified.py
"""

import streamlit as st
import os
import json
import sqlite3
import requests
import re
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import pathlib

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="Albert Query v3.1", page_icon="🧙‍♂️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stChatMessage[data-testid="user-message"] {
        flex-direction: row-reverse;
        text-align: right;
    }
    .stChatMessage[data-testid="user-message"] > div {
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Paths
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT = pathlib.Path("C:/Users/Vincent/GitHub/Vincent-20-100/Agentic_Systems_Project_Vlamy")
DB_FOLDER_PATH = str(PROJECT_ROOT / "data" / "databases")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === AGENT STATE (removed clarification fields) ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    db_catalog: dict

    # Collected data
    sql_result: str
    omdb_result: str
    web_result: str

    # Control flags
    needs_sql: bool
    needs_omdb: bool
    needs_web: bool

    # Metadata
    sources_used: list
    confidence_score: float
    current_step: str

# === HELPER FUNCTIONS ===
def clean_json(text: str) -> str:
    """Clean JSON from markdown formatting"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def extract_urls(text: str) -> list:
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def build_db_catalog(folder_path: str) -> dict:
    """Build complete database catalog with paths and schemas"""
    catalog = {
        "folder_path": folder_path,
        "databases": {},
        "error": None
    }

    try:
        db_files = [f for f in os.listdir(folder_path)
                   if f.endswith(('.db', '.sqlite', '.sqlite3'))]
    except FileNotFoundError:
        catalog["error"] = f"Folder {folder_path} not found"
        return catalog

    if not db_files:
        catalog["error"] = "No SQLite databases found"
        return catalog

    for db_file in db_files:
        db_path_full = os.path.join(folder_path, db_file)
        db_name = os.path.splitext(db_file)[0]

        try:
            conn = sqlite3.connect(db_path_full)
            cursor = conn.cursor()

            db_info = {
                "file_name": db_file,
                "full_path": db_path_full,
                "tables": {}
            }

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()

                db_info["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "primary_key": bool(col[5])
                        } for col in columns_info
                    ],
                    "column_names": [col[1] for col in columns_info]
                }

            catalog["databases"][db_name] = db_info
            conn.close()

        except Exception as e:
            catalog["databases"][db_name] = {
                "file_name": db_file,
                "full_path": db_path_full,
                "error": str(e)
            }

    return catalog

def format_catalog_for_llm(catalog: dict) -> str:
    """Format catalog for LLM"""
    if catalog.get("error"):
        return f"ERROR: {catalog['error']}"

    formatted = "Available Databases:\n\n"

    for db_name, db_info in catalog["databases"].items():
        if "error" in db_info:
            formatted += f"ERROR {db_name}: {db_info['error']}\n"
            continue

        formatted += f"**Database: {db_name}** (file: {db_info['file_name']})\n"

        for table_name, table_info in db_info["tables"].items():
            cols = ", ".join([f"{col['name']} ({col['type']})"
                            for col in table_info["columns"]])
            formatted += f"  • Table `{table_name}`: {cols}\n"

        formatted += "\n"

    return formatted

# === TOOLS ===
@tool
def execute_sql_query(query: str, db_name: str, state_catalog: dict) -> str:
    """Execute SQL query using catalog state"""
    catalog = state_catalog

    if db_name not in catalog["databases"]:
        available = ", ".join(catalog["databases"].keys())
        return json.dumps({
            "error": f"Database '{db_name}' not found. Available: {available}"
        })

    db_info = catalog["databases"][db_name]

    if "error" in db_info:
        return json.dumps({"error": f"Database error: {db_info['error']}"})

    db_path = db_info["full_path"]

    if not os.path.exists(db_path):
        return json.dumps({"error": f"Database file not found: {db_path}"})

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
        return json.dumps({"error": f"SQL Error: {str(e)}"})

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Web search via DuckDuckGo"""
    try:
        search = DuckDuckGoSearchResults(num_results=num_results)
        return search.run(query)
    except Exception as e:
        return json.dumps({"error": f"Web search error: {str(e)}"})

@tool
def omdb_api(by: str = "search", i: str = None, t: str = None,
             s: str = None, y: str = None, plot: str = "short") -> str:
    """Query OMDb API for movie/series info"""
    if not OMDB_API_KEY:
        return json.dumps({"error": "OMDB_API_KEY missing"})

    params = {"apikey": OMDB_API_KEY, "plot": plot}

    if by == "id" and i:
        params["i"] = i
    elif by == "title" and t:
        params["t"] = t
    elif by == "search" and s:
        params["s"] = s
    else:
        return json.dumps({"error": "Missing parameters"})

    if y:
        params["y"] = y

    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"OMDb API error: {str(e)}"})

# === WORKFLOW NODES ===

def get_schema_node(state: AgentState) -> dict:
    """Load database catalog"""
    catalog = build_db_catalog(DB_FOLDER_PATH)

    if catalog.get("error"):
        return {
            "db_catalog": catalog,
            "current_step": "schema_error"
        }

def pre_agent_node(state: AgentState) -> dict:
    """Analyze query, resolve context, decide which tools to use"""
    question = state.get("original_question", "")
    history = state.get("messages", [])

    prompt = f"""Analyze the question and decide which tools are needed.

QUESTION: "{question}"
HISTORY: {json.dumps(history[-3:], ensure_ascii=False)}

TOOLS:
- SQL: Movie/series database queries
- OMDB: Detailed info (posters, actors, plot)
- Web: Recent news, trending, current events

Resolve references from history (e.g., "that movie" → actual name).

JSON response:
{{
  "resolved_query": "clear reformulated query",
  "needs_sql": true/false,
  "needs_omdb": true/false,
  "needs_web": true/false
}}"""

    response = llm.invoke(prompt)
    
    try:
        plan = json.loads(clean_json(response.content))
        return {
            "resolved_query": plan.get("resolved_query", question),
            "needs_sql": plan.get("needs_sql", True),
            "needs_omdb": plan.get("needs_omdb", False),
            "needs_web": plan.get("needs_web", False),
            "current_step": "planned"
        }
    except:
        return {
            "resolved_query": question,
            "needs_sql": True,
            "needs_omdb": False,
            "needs_web": False,
            "current_step": "planned"
        }

def agent_node(state: AgentState) -> dict:
    """Execute planned tools"""
    results = {
        "sql_result": "[]",
        "omdb_result": "{}",
        "web_result": "{}",
        "sources_used": []
    }
    
    # Load schema if needed
    if state.get("needs_sql") and not state.get("db_catalog"):
        catalog = build_db_catalog(DB_FOLDER_PATH)
        results["db_catalog"] = catalog
    else:
        results["db_catalog"] = state.get("db_catalog", {})
    
    # Execute SQL
    if state.get("needs_sql"):
        sql_res = sql_tool(state, results["db_catalog"])
        results.update(sql_res)
    
    # Execute OMDB
    if state.get("needs_omdb"):
        omdb_res = omdb_tool(state, results["sql_result"])
        results.update(omdb_res)
    
    # Execute Web
    if state.get("needs_web"):
        web_res = web_tool(state)
        results.update(web_res)
    
    results["current_step"] = "executed"
    return results


def sql_tool(state: AgentState, catalog: dict) -> dict:
    """Execute SQL query"""
    catalog_info = format_catalog_for_llm(catalog)
    query = state.get("resolved_query", "")
    
    prompt = f"""Generate SQL query for: "{query}"

DATABASES:
{catalog_info}

Use LIKE with %, LIMIT 10, ORDER BY when needed.

JSON:
{{
  "can_answer": true/false,
  "db_name": "name",
  "query": "SELECT ..."
}}"""

    response = llm.invoke(prompt)
    
    try:
        dec = json.loads(clean_json(response.content))
        if dec.get("can_answer"):
            result = execute_sql_query.invoke({
                "query": dec["query"],
                "db_name": dec["db_name"],
                "state_catalog": catalog
            })
            return {
                "sql_result": result,
                "sources_used": [f"DB: {dec['db_name']}"]
            }
    except:
        pass
    
    return {"sql_result": "[]"}


def omdb_tool(state: AgentState, sql_result: str) -> dict:
    """Get OMDB data"""
    title = None
    
    try:
        data = json.loads(sql_result)
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            title = item.get("title") or item.get("Title") or item.get("name")
        
        if not title:
            query = state.get("resolved_query", "")
            match = re.search(r'"([^"]+)"', query)
            if match:
                title = match.group(1)
        
        if title:
            result = omdb_api.invoke({"by": "title", "t": title, "plot": "full"})
            return {
                "omdb_result": result,
                "sources_used": [f"OMDB: {title}"]
            }
    except:
        pass
    
    return {"omdb_result": "{}"}


def web_tool(state: AgentState) -> dict:
    """Web search"""
    query = state.get("resolved_query", "")
    
    try:
        result = web_search.invoke({"query": query})
        return {
            "web_result": result,
            "sources_used": ["Web"]
        }
    except:
        return {"web_result": "{}"}

def synthesizer_node(state: AgentState) -> dict:
    """Generate final response from all data"""
    question = state.get("original_question", "")
    sql = state.get("sql_result", "[]")
    omdb = state.get("omdb_result", "{}")
    web = state.get("web_result", "{}")
    sources = state.get("sources_used", [])
    
    prompt = f"""Answer the question using available data.

QUESTION: "{question}"

SQL DATA: {sql}
OMDB DATA: {omdb}
WEB DATA: {web}

SOURCES: {', '.join(sources)}

Write a natural, clear response. Cite sources for specific facts."""

    response = llm.invoke(prompt)
    
    return {
        "final_response": response.content,
        "current_step": "complete"
    }

# === ROUTING ===

def route_after_intent(state: AgentState) -> Literal["sql_agent", "omdb_enrichment", "web_search"]:
    """Route based on intent analysis"""
    if state.get("needs_sql", True):
        return "sql_agent"
    elif state.get("needs_omdb", False):
        return "omdb_enrichment"
    else:
        return "web_search"

def route_after_evaluator(state: AgentState) -> Literal["omdb_enrichment", "web_search"]:
    """Route to OMDB if needed, else skip to web"""
    return "omdb_enrichment" if state.get("needs_omdb", False) else "web_search"

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
    """Build and compile workflow WITHOUT clarification loop"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("pre_agent_node", pre_agent_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("sql_search", sql_tool)
    workflow.add_node("omdb_enrichment", omdb_tool)
    workflow.add_node("web_search", web_tool)
    workflow.add_node("synthesize", synthesizer_node)

    # Simple flow: START → schema → intent → sql/omdb/web → synthesize → END
    workflow.add_edge(START, "get_schema")
    workflow.add_edge("get_schema", "pre_agent_node")
    workflow.add_edge("pre_agent_node", "agent_node")
    workflow.add_edge("agent_node", "sql_search")
    workflow.add_conditional_edges(
        "agent_node",
        route_after_evaluator,
        {"omdb_enrichment": "omdb_enrichment", "web_search": "web_search"}
    )
    workflow.add_edge("omdb_enrichment", "synthesize")
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

app = build_agent()

# === STREAMLIT INTERFACE ===

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "db_catalog" not in st.session_state:
    st.session_state.db_catalog = {}

if "sources" not in st.session_state:
    st.session_state.sources = []

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_session"

# Show welcome message
if not st.session_state.welcome_shown:
    catalog = build_db_catalog(DB_FOLDER_PATH)
    st.session_state.db_catalog = catalog

    if catalog.get("error"):
        welcome_msg = f"**Loading Error:** {catalog['error']}"
    else:
        welcome_msg = "**Hey! I'm Albert v3.1 (No Clarifications)**\n\nI'll help you make sense of your data!\n\n"
        welcome_msg += "**Available Databases:**\n\n"

        for db_name, db_info in catalog["databases"].items():
            if "error" in db_info:
                welcome_msg += f"ERROR **{db_name}**: {db_info['error']}\n"
                continue

            welcome_msg += f"**{db_name}** ({db_info['file_name']})\n"
            for table_name, table_info in db_info["tables"].items():
                cols = ", ".join([f"`{col['name']}`" for col in table_info["columns"]])
                welcome_msg += f"  • Table `{table_name}`: {cols}\n"
            welcome_msg += "\n"

        welcome_msg += "**Available Tools:**\n"
        welcome_msg += "- SQL Queries (always checked first)\n"
        welcome_msg += "- OMDb API enrichment (automatic)\n"
        welcome_msg += "- Web search (when needed)\n\n"
        welcome_msg += "**Go ahead, ask me anything!**"

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": welcome_msg
    })
    st.session_state.welcome_shown = True

# Display chat history
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append(HumanMessage(content=prompt))

    # Keep message history manageable
    if len(st.session_state.agent_messages) > 100:
        st.session_state.agent_messages = st.session_state.agent_messages[-100:]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        inputs = {
            "messages": st.session_state.agent_messages,
            "db_catalog": st.session_state.db_catalog,
            "sql_result": "",
            "omdb_result": "",
            "web_result": "",
            "original_question": prompt,
            "needs_sql": False,
            "needs_omdb": False,
            "needs_web": False,
            "sources_used": [],
            "confidence_score": 0.0,
            "current_step": ""
        }

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        result = None

        for step in app.stream(inputs, config=config, stream_mode="values"):
            result = step
            current_step = step.get("current_step", "")

            # Show progress
            if current_step == "schema_loaded":
                status_placeholder.info("Loading catalog...")
            elif current_step == "intent_analyzed":
                status_placeholder.info("Intent detected...")
            elif current_step == "sql_executed":
                status_placeholder.info("SQL data retrieved...")
            elif current_step == "omdb_enriched":
                status_placeholder.info("OMDb enrichment...")
            elif current_step == "web_searched":
                status_placeholder.info("Web search...")
            elif current_step == "synthesis_complete":
                status_placeholder.success("Answer ready!")

        if result:
            status_placeholder.empty()

            # Get the final synthesized message
            final_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
            if final_messages:
                response_text = final_messages[-1].content
                response_placeholder.markdown(response_text)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                # Update session state - keep only user question + final answer
                user_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
                last_user = user_msgs[-1] if user_msgs else None

                if last_user:
                    st.session_state.agent_messages = [last_user, final_messages[-1]]

                st.session_state.db_catalog = result.get("db_catalog", st.session_state.db_catalog)
                st.session_state.sources = result.get("sources_used", [])

"""
SQL Agent with Memory, Sources, and Human-in-loop
Launch: streamlit run app.py
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
from dotenv import load_dotenv
import pathlib

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="Albert Query", page_icon="🧙‍♂️", layout="wide")
#st.title("🧙‍♂️ Albert Query")

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Chemin absolu vers les databases
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DB_FOLDER_PATH = str(SCRIPT_DIR.parent / "data")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === AGENT STATE ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    db_catalog: dict
    next_tool: str
    sources_used: list
    needs_clarification: bool
    clarification_question: str
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
    """🆕 Construit un catalogue complet des databases avec chemins et schémas"""
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
        db_name = os.path.splitext(db_file)[0]  # nom sans extension
        
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
    """🆕 Formate le catalogue pour le LLM"""
    if catalog.get("error"):
        return f"ERROR: {catalog['error']}"
    
    formatted = "📊 Available Databases:\n\n"
    
    for db_name, db_info in catalog["databases"].items():
        if "error" in db_info:
            formatted += f"❌ {db_name}: {db_info['error']}\n"
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
    """🆕 Execute SQL query using catalog state"""
    catalog = state_catalog
    
    # Vérifier que la database existe dans le catalogue
    if db_name not in catalog["databases"]:
        available = ", ".join(catalog["databases"].keys())
        return json.dumps({
            "error": f"Database '{db_name}' not found in catalog. Available: {available}"
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
        return json.dumps({"error": "Missing parameters (i/t/s depending on 'by')"})
    
    if y:
        params["y"] = y
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"OMDb API error: {str(e)}"})

# Bind tools to LLM (sans execute_sql_query pour l'instant)
tools = [web_search, omdb_api]
llm_with_tools = llm.bind_tools(tools)

# === WORKFLOW NODES ===

def get_schema_node(state: AgentState) -> dict:
    """🆕 Charge le catalogue complet des databases"""
    catalog = build_db_catalog(DB_FOLDER_PATH)
    
    if catalog.get("error"):
        return {
            "db_catalog": catalog,
            "current_step": "schema_error",
            "messages": [AIMessage(content=f"❌ Error loading databases: {catalog['error']}")]
        }
    
    return {
        "db_catalog": catalog,
        "current_step": "schema_loaded",
        "messages": [AIMessage(content="✅ Database catalog loaded")]
    }

def clarify_question_node(state: AgentState) -> dict:
    """Analyze question and ask for clarification if needed"""
    user_question = state["messages"][-1].content
    
    # Skip clarification if it's a response to a previous clarification
    if len(state["messages"]) > 2 and state.get("needs_clarification"):
        return {
            "needs_clarification": False,
            "current_step": "clarification_answered",
            "messages": [AIMessage(content="✅ Thanks for clarification")]
        }
    
    catalog_info = format_catalog_for_llm(state["db_catalog"])
    
    prompt = f"""Analyze this question: "{user_question}"

Available context:
{catalog_info}
- OMDb API (detailed movie/series info)
- Web search (news, recent releases)

Is the question:
- CLEAR: can answer directly
- AMBIGUOUS: need clarification

Examples of AMBIGUOUS:
- "a movie" → which movie?
- "when released?" → which title?
- "good series" → genre? year?

Respond ONLY in JSON:
{{
  "status": "clear" or "ambiguous",
  "clarification": "Question to ask user" (if ambiguous),
  "reasoning": "Why it's ambiguous"
}}"""
    
    response = llm.invoke(prompt)
    
    try:
        decision = json.loads(clean_json(response.content))
        
        if decision["status"] == "ambiguous":
            return {
                "needs_clarification": True,
                "clarification_question": decision["clarification"],
                "current_step": "waiting_clarification",
                "messages": [AIMessage(content=decision["clarification"])]
            }
        else:
            return {
                "needs_clarification": False,
                "current_step": "question_clear",
                "messages": [AIMessage(content="✅ Question clear, analyzing...")]
            }
    except:
        return {
            "needs_clarification": False,
            "current_step": "question_clear",
            "messages": [AIMessage(content="✅ Analyzing...")]
        }

def chief_agent_node(state: AgentState) -> dict:
    """Chief analyzes and calls appropriate tool with catalog context"""
    catalog_info = format_catalog_for_llm(state["db_catalog"])
    
    prompt = f"""You are a SQL/data assistant. You have access to:

1. **execute_sql_query**: Query databases
   - You MUST provide: query (SQL), db_path (full path from catalog)
   - Use the catalog below to know which db_path and tables to use

2. **web_search**: General web searches

3. **omdb_api**: Detailed movie/series info

{catalog_info}

⚠️ IMPORTANT for SQL queries:
- Always specify the correct 'db_path' (use full_path from catalog)
- Refer to the catalog above for exact table and column names
- Write valid SQLite syntax

Analyze the user request and choose THE MOST appropriate tool."""
    
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
        
        return {
            "next_tool": next_tool,
            "current_step": f"calling_{tool_name}",
            "messages": [response]
        }
    else:
        return {
            "next_tool": "none",
            "current_step": "direct_answer",
            "messages": [response]
        }

def tool_executor_node(state: AgentState) -> dict:
    """Execute the chosen tool and track sources"""
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [AIMessage(content="⚠️ No tool to execute")]}
    
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
        return {"messages": [AIMessage(content=f"⚠️ Unknown tool {tool_name}")]}
    
    try:
        result = tool_func.invoke(tool_args)
        
        # Track source
        source = ""
        if tool_name == "execute_sql_query":
            db_path = tool_args.get("db_path", "unknown")
            source = f"🗄️ Database: [{db_path}]"
        elif tool_name == "omdb_api":
            title = tool_args.get('t', tool_args.get('i', 'unknown'))
            url = f"{OMDB_BASE_URL}?t={title.replace(' ', '+')}&apikey=***"
            source = f"🎬 OMDb API: [{title}]({url})"
        elif tool_name == "web_search":
            urls = extract_urls(result)
            if urls:
                source = f"🌐 Web: [DuckDuckGo]({urls[0]})"
            else:
                source = "🌐 Web: DuckDuckGo Search"
        
        return {
            "messages": [AIMessage(content=f"📊 Result:\n{result}")],
            "sources_used": state.get("sources_used", []) + [source],
            "current_step": f"{tool_name}_completed"
        }
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Error {tool_name}: {str(e)}")]}

def synthesize_node(state: AgentState) -> dict:
    """Synthesize results and add sources"""
    sources = state.get("sources_used", [])
    sources_text = "\n".join([f"- {s}" for s in sources]) if sources else ""
    
    prompt = f"""You are an assistant that synthesizes results.
Provide a clear, concise answer in French to the user.
Use the data available in the message history.

At the END of your response, add a section:

**📚 Sources:**
{sources_text}

Be natural, don't mention "based on the data" or similar."""
    
    messages = [{"role": "system", "content": prompt}] + [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]
    
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "current_step": "synthesis_complete"
    }

# === ROUTING ===

def route_after_clarify(state: AgentState) -> Literal["wait_user", "chief_agent"]:
    """Route based on clarification need"""
    return "wait_user" if state.get("needs_clarification") else "chief_agent"

def route_after_chief(state: AgentState) -> Literal["tool_executor", "synthesize"]:
    """Route based on tool choice"""
    return "tool_executor" if state["next_tool"] != "none" else "synthesize"

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
    """Build and compile the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("clarify_question", clarify_question_node)
    workflow.add_node("chief_agent", chief_agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("wait_user", lambda state: state)
    
    workflow.add_edge(START, "get_schema")
    workflow.add_edge("get_schema", "clarify_question")
    
    workflow.add_conditional_edges(
        "clarify_question",
        route_after_clarify,
        {"wait_user": END, "chief_agent": "chief_agent"}
    )
    
    workflow.add_conditional_edges(
        "chief_agent",
        route_after_chief,
        {"tool_executor": "tool_executor", "synthesize": "synthesize"}
    )
    
    workflow.add_edge("tool_executor", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

app = build_agent()

# === STREAMLIT INTERFACE ===

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "db_catalog" not in st.session_state:
    st.session_state.db_catalog = {}

if "db_catalog" not in st.session_state:
    st.session_state.db_catalog = {}

if "sources" not in st.session_state:
    st.session_state.sources = []

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

# Show welcome message on first load
if not st.session_state.welcome_shown:
    catalog = build_db_catalog(DB_FOLDER_PATH)
    st.session_state.db_catalog = catalog
    
    if catalog.get("error"):
        welcome_msg = f"⚠️ **Erreur de chargement:** {catalog['error']}"
    else:
        welcome_msg = "##### 👋 **Salut !** Moi c'est **Albert**,\n\n###### 🧐 Je vais t'aider à y voir clair dans tes données !\n\n"
        welcome_msg += "\n\n ###### 📊 Bases de données disponibles :\n\n"
        
        for db_name, db_info in catalog["databases"].items():
            if "error" in db_info:
                welcome_msg += f"❌ **{db_name}**: {db_info['error']}\n"
                continue
                
            welcome_msg += f"**{db_name}** ({db_info['file_name']})\n"
            for table_name, table_info in db_info["tables"].items():
                cols = ", ".join([f"`{col['name']}`" for col in table_info["columns"]])
                welcome_msg += f"  • Table `{table_name}` : {cols}\n"
            welcome_msg += "\n"
        
        welcome_msg += "###### 🔧 Outils disponibles :\n"
        welcome_msg += "- 🗄️ Requêtes SQL sur les bases\n"
        welcome_msg += "- 🎬 OMDb API pour infos films/séries détaillées\n"
        welcome_msg += "- 🌐 Recherche web pour actualités\n\n"
        welcome_msg += "💬 **Vas-y, pose-moi une question !**"
    
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
            "next_tool": "",
            "sources_used": [],
            "needs_clarification": False,
            "clarification_question": "",
            "current_step": ""
        }
        
        result = None
        
        for step in app.stream(inputs, stream_mode="values"):
            result = step
            current_step = step.get("current_step", "")
            
            if current_step == "schema_loaded":
                status_placeholder.info("📂 Loading catalog...")
            elif current_step == "question_clear":
                status_placeholder.info("✅ Question analyzed...")
            elif current_step.startswith("calling_"):
                tool = current_step.replace("calling_", "")
                if tool == "execute_sql_query":
                    status_placeholder.info("🗄️ Querying database...")
                elif tool == "web_search":
                    status_placeholder.info("🌐 Searching the web...")
                elif tool == "omdb_api":
                    status_placeholder.info("🎬 Calling OMDb API...")
            elif current_step.endswith("_completed"):
                status_placeholder.success("✅ Data retrieved!")
            elif current_step == "synthesis_complete":
                status_placeholder.success("💬 Response ready!")
        
        if result:
            status_placeholder.empty()
            
            final_message = result["messages"][-1]
            response_text = final_message.content
            
            response_placeholder.markdown(response_text)
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            st.session_state.agent_messages = result["messages"]
            st.session_state.db_catalog = result.get("db_catalog", st.session_state.db_catalog)
            st.session_state.sources = result.get("sources_used", [])
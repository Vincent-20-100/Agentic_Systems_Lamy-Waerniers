################ PROBLEME DE CHEMIN DE BASE DE DONNEES A REGARDER ################

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

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="Agent SQL", page_icon="🤖", layout="wide")
st.title("🤖 Agent SQL Netflix")

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"
DB_FOLDER_PATH = os.getenv("DB_FOLDER_PATH", "../data")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === AGENT STATE ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    schema: str
    next_tool: str
    sources_used: list  # Track sources
    needs_clarification: bool
    clarification_question: str
    current_step: str  # For real-time display

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

# === TOOLS ===
@tool
def get_db_schema(DB_FOLDER_PATH: str = "../data") -> str:
    """Get schema of all SQLite databases in folder"""
    result = {"databases": [], "error": None}
    
    try:
        db_files = [f for f in os.listdir(DB_FOLDER_PATH) if f.endswith(('.db', '.sqlite', '.sqlite3'))]
    except FileNotFoundError:
        return json.dumps({"error": f"Folder {DB_FOLDER_PATH} not found"})

    if not db_files:
        return json.dumps({"error": "No SQLite databases found"})
    
    for db_file in db_files:
        db_path_full = os.path.join(DB_FOLDER_PATH, db_file)
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
def execute_sql_query(query: str, db_path: str = None) -> str:
    """Execute SQL query on the database"""
    path = db_path
    
    if not os.path.exists(path):
        return json.dumps({"error": f"Database {path} not found"})
    
    try:
        conn = sqlite3.connect(path)
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

# Bind tools to LLM
tools = [execute_sql_query, web_search, omdb_api]
llm_with_tools = llm.bind_tools(tools)

# === WORKFLOW NODES ===

def get_schema_node(state: AgentState) -> dict:
    """Load database schema"""
    schema = get_db_schema.invoke({})
    return {
        "schema": schema,
        "current_step": "schema_loaded",
        "messages": [AIMessage(content="✅ Schema loaded")]
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
    
    prompt = f"""Analyze this question: "{user_question}"

Available context:
- Database netflix.db (movies, series, years, ratings)
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
        # If JSON parsing fails, assume clear
        return {
            "needs_clarification": False,
            "current_step": "question_clear",
            "messages": [AIMessage(content="✅ Analyzing...")]
        }

def chief_agent_node(state: AgentState) -> dict:
    """Chief analyzes and calls appropriate tool"""
    prompt = f"""You are a SQL/data assistant. You have access to:
1. execute_sql_query: query netflix.db
2. web_search: general web searches
3. omdb_api: detailed movie/series info

Available schema:
{state['schema']}

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
            source = f"🗄️ Database: [{os.path.basename(DB_FOLDER_PATH)}](file://{os.path.abspath(DB_FOLDER_PATH)})"
        elif tool_name == "omdb_api":
            title = tool_args.get('t', tool_args.get('i', 'unknown'))
            url = f"{OMDB_BASE_URL}?t={title.replace(' ', '+')}&apikey=***"
            source = f"🎬 OMDb API: [{title}]({url})"
        elif tool_name == "web_search":
            # Extract URLs from result
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
    workflow.add_node("wait_user", lambda state: state)  # Dummy node
    
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

if "schema" not in st.session_state:
    st.session_state.schema = ""

if "sources" not in st.session_state:
    st.session_state.sources = []

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

# Show welcome message on first load
if not st.session_state.welcome_shown:
    # Load schema to display available resources
    schema_result = get_db_schema.invoke({})
    st.session_state.schema = schema_result
    
    try:
        schema_data = json.loads(schema_result)
        
        # Format welcome message
        welcome_msg = "👋 **Bienvenue !** Voici les ressources disponibles :\n\n"
        welcome_msg += "### 📊 Bases de données :\n"
        
        for db in schema_data.get("databases", []):
            welcome_msg += f"\n**{db['name']}**\n"
            for table in db.get("tables", []):
                cols = ", ".join([f"`{col['name']}`" for col in table.get("columns", [])])
                welcome_msg += f"  • Table `{table['name']}` : {cols}\n"
        
        welcome_msg += "\n### 🔧 Outils disponibles :\n"
        welcome_msg += "- 🗄️ Requêtes SQL sur les bases\n"
        welcome_msg += "- 🎬 OMDb API pour infos films/séries détaillées\n"
        welcome_msg += "- 🌐 Recherche web pour actualités\n\n"
        welcome_msg += "💬 **Posez votre question !**"
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
        st.session_state.welcome_shown = True
        
    except:
        # Fallback if schema parsing fails
        welcome_msg = "👋 **Bienvenue !** Je suis prêt à vous aider avec la base Netflix. Posez votre question !"
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
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append(HumanMessage(content=prompt))
    
    # Keep only last 100 messages
    if len(st.session_state.agent_messages) > 100:
        st.session_state.agent_messages = st.session_state.agent_messages[-100:]
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Execute agent with real-time display
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        inputs = {
            "messages": st.session_state.agent_messages,
            "schema": st.session_state.schema,
            "next_tool": "",
            "sources_used": [],
            "needs_clarification": False,
            "clarification_question": "",
            "current_step": ""
        }
        
        result = None
        
        # Stream with status updates
        for step in app.stream(inputs, stream_mode="values"):
            result = step
            current_step = step.get("current_step", "")
            
            # Update status based on step
            if current_step == "schema_loaded":
                status_placeholder.info("📂 Loading schema...")
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
        
        # Display final response
        if result:
            status_placeholder.empty()
            
            final_message = result["messages"][-1]
            response_text = final_message.content
            
            response_placeholder.markdown(response_text)
            
            # Save to history
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Update agent messages and schema
            st.session_state.agent_messages = result["messages"]
            st.session_state.schema = result.get("schema", st.session_state.schema)
            st.session_state.sources = result.get("sources_used", [])
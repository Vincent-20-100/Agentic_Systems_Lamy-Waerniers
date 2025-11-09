"""
SQL Agent with Sequential Flow: SQL → OMDB → Web → Synthesis
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

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Absolute path to databases
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
    sql_result: str  # Store SQL result
    omdb_result: str  # Store OMDB result
    needs_web: bool  # Flag if web is needed
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

def sql_agent_node(state: AgentState) -> dict:
    """ALWAYS query SQL first"""
    catalog_info = format_catalog_for_llm(state["db_catalog"])
    
    # Get the actual user question (skip system messages)
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    prompt = f"""You are a SQL expert. Your job is to query the database to answer the user's question.

Available databases:
{catalog_info}

User question: "{user_question}"

Generate a SQL query to answer this question.
- Use the exact table/column names from the catalog above
- Return ONLY valid SQLite syntax
- If the question cannot be answered with the available data, set can_answer_with_sql to false

Respond ONLY in JSON format:
{{
  "can_answer_with_sql": true or false,
  "db_name": "database_name",
  "query": "SELECT ... FROM ... WHERE ...",
  "reasoning": "why this query will answer the question"
}}"""
    
    response = llm.invoke(prompt)
    
    try:
        decision = json.loads(clean_json(response.content))
        
        if decision.get("can_answer_with_sql", False):
            # Execute SQL
            sql_result = execute_sql_query.invoke({
                "query": decision["query"],
                "db_name": decision["db_name"],
                "state_catalog": state["db_catalog"]
            })
            
            db_name = decision["db_name"]
            source = f"🗄️ Database: {db_name}"
            
            return {
                "sql_result": sql_result,
                "sources_used": state.get("sources_used", []) + [source],
                "current_step": "sql_executed",
                "messages": state["messages"] + [AIMessage(content=f"📊 SQL Result:\n{sql_result}")]
            }
        else:
            return {
                "sql_result": "[]",
                "current_step": "sql_no_data",
                "messages": state["messages"] + [AIMessage(content="No relevant data in database")]
            }
    except Exception as e:
        return {
            "sql_result": f'{{"error": "{str(e)}"}}',
            "current_step": "sql_error",
            "messages": state["messages"] + [AIMessage(content=f"❌ SQL error: {str(e)}")]
        }

def result_evaluator_node(state: AgentState) -> dict:
    """Evaluate if we need OMDB enrichment"""
    sql_result = state.get("sql_result", "[]")
    
    # Check if SQL returned data
    try:
        data = json.loads(sql_result)
        has_data = isinstance(data, list) and len(data) > 0 and "error" not in data
        
        if has_data:
            return {
                "current_step": "evaluation_needs_omdb",
                "messages": state["messages"] + [AIMessage(content="✅ SQL data found, enriching with OMDB...")]
            }
        else:
            return {
                "current_step": "evaluation_skip_omdb",
                "messages": state["messages"] + [AIMessage(content="⚠️ No SQL data, checking web...")]
            }
    except:
        return {
            "current_step": "evaluation_skip_omdb",
            "messages": state["messages"] + [AIMessage(content="⚠️ SQL parsing error, checking web...")]
        }

def omdb_enrichment_node(state: AgentState) -> dict:
    """Enrich SQL results with OMDB data"""
    sql_result = state.get("sql_result", "[]")
    
    try:
        data = json.loads(sql_result)
        
        # Extract first title from SQL result
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            
            # Try different possible title fields
            title = (first_item.get("title") or 
                    first_item.get("Title") or
                    first_item.get("show_id") or
                    first_item.get("name") or "")
            
            if title:
                omdb_result = omdb_api.invoke({
                    "by": "title",
                    "t": title,
                    "plot": "full"
                })
                
                source = f"🎬 OMDb API: {title}"
                
                return {
                    "omdb_result": omdb_result,
                    "sources_used": state.get("sources_used", []) + [source],
                    "current_step": "omdb_enriched",
                    "messages": state["messages"] + [AIMessage(content=f"🎬 OMDB Data:\n{omdb_result}")]
                }
        
        return {
            "omdb_result": "{}",
            "current_step": "omdb_skipped",
            "messages": state["messages"] + [AIMessage(content="⚠️ No title found for OMDB enrichment")]
        }
    except Exception as e:
        return {
            "omdb_result": "{}",
            "current_step": "omdb_error",
            "messages": state["messages"] + [AIMessage(content=f"⚠️ OMDB error: {str(e)}")]
        }

def web_search_node(state: AgentState) -> dict:
    """Search web ONLY if explicitly needed"""
    
    # Get the actual user question
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    sql_result = state.get("sql_result", "[]")
    
    prompt = f"""Decide if we need to search the web for additional information.

User question: "{user_question}"
SQL result available: {sql_result != "[]" and "error" not in sql_result}

ONLY search the web if the question explicitly requires:
- Recent news or current events ("what's new", "trending", "latest news")
- Information that cannot be in our database (real-time data, today's info)
- External context not related to our data

DO NOT search if:
- We have data from SQL (even if incomplete)
- Question is about data we should have in database
- General movie/series information

Respond ONLY in JSON:
{{
  "should_search": true or false,
  "search_query": "optimized query for web" (if should_search is true),
  "reasoning": "why we should/shouldn't search"
}}"""
    
    response = llm.invoke(prompt)
    
    try:
        decision = json.loads(clean_json(response.content))
        
        if decision.get("should_search", False):
            web_result = web_search.invoke({"query": decision["search_query"]})
            urls = extract_urls(web_result)
            source = f"🌐 Web: [DuckDuckGo]({urls[0]})" if urls else "🌐 Web: DuckDuckGo"
            
            return {
                "sources_used": state.get("sources_used", []) + [source],
                "current_step": "web_searched",
                "messages": state["messages"] + [AIMessage(content=f"🌐 Web Results:\n{web_result}")]
            }
        else:
            return {
                "current_step": "web_skipped",
                "messages": state["messages"] + [AIMessage(content="✅ No web search needed")]
            }
    except Exception as e:
        return {
            "current_step": "web_error",
            "messages": state["messages"] + [AIMessage(content=f"⚠️ Web search error: {str(e)}")]
        }

def synthesize_node(state: AgentState) -> dict:
    """Synthesize all results into final answer"""
    sources = state.get("sources_used", [])
    sources_text = "\n".join([f"- {s}" for s in sources]) if sources else "- No external sources used"
    
    # Get original user question
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    # Get all collected data explicitly from state
    sql_result = state.get("sql_result", "[]")
    omdb_result = state.get("omdb_result", "{}")
    web_result = state.get("web_result", "{}")
    
    # Parse results to check what we have
    has_sql_data = False
    has_omdb_data = False
    has_web_data = False
    
    try:
        sql_data = json.loads(sql_result)
        has_sql_data = isinstance(sql_data, list) and len(sql_data) > 0 and "error" not in str(sql_data)
    except:
        pass
    
    try:
        omdb_data = json.loads(omdb_result)
        has_omdb_data = omdb_data and omdb_data.get("Response") != "False"
    except:
        pass
    
    try:
        web_data = json.loads(web_result)
        has_web_data = bool(web_data) and "error" not in str(web_data)
    except:
        pass
    
    prompt = f"""You are Albert, a friendly data assistant that helps users understand their data.

User's original question: "{user_question}"

Available data:
---
SQL Database Results: 
{sql_result if has_sql_data else "No data found in database"}

---
OMDB API Results:
{omdb_result if has_omdb_data else "No OMDB data available"}

---
WEB Results:
{web_result if has_web_data else "No WEB data available"}

---


Your task:
1. Provide a clear, concise answer in French
2. Use natural language, be conversational and friendly
3. If you have data from SQL or OMDB, present it naturally
4. If no data is available, explain politely what's missing
5. Don't mention technical terms like "SQL" or "API" - just answer naturally
6. Always add sources at the end

Format your response naturally, then add at the END:

**📚 Sources:**
{sources_text}

Be helpful, friendly, and human! Answer as if you're explaining to a friend."""
    
    response = llm.invoke(prompt)
    
    return {
        "messages": state["messages"] + [response],
        "current_step": "synthesis_complete"
    }

# === ROUTING ===

def route_after_clarify(state: AgentState) -> Literal["wait_user", "sql_agent"]:
    """Route based on clarification need"""
    return "wait_user" if state.get("needs_clarification") else "sql_agent"

def route_after_evaluator(state: AgentState) -> Literal["omdb_enrichment", "web_search"]:
    """Route to OMDB if we have SQL data, otherwise skip to web"""
    current_step = state.get("current_step", "")
    return "omdb_enrichment" if current_step == "evaluation_needs_omdb" else "web_search"

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
    """Build and compile the sequential workflow"""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("clarify_question", clarify_question_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("result_evaluator", result_evaluator_node)
    workflow.add_node("omdb_enrichment", omdb_enrichment_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("wait_user", lambda state: state)
    
    # Sequential flow: Schema → Clarify → SQL → Evaluator → OMDB → Web → Synthesize
    workflow.add_edge(START, "get_schema")
    workflow.add_edge("get_schema", "clarify_question")
    
    workflow.add_conditional_edges(
        "clarify_question",
        route_after_clarify,
        {"wait_user": END, "sql_agent": "sql_agent"}
    )
    
    workflow.add_edge("sql_agent", "result_evaluator")
    
    workflow.add_conditional_edges(
        "result_evaluator",
        route_after_evaluator,
        {"omdb_enrichment": "omdb_enrichment", "web_search": "web_search"}
    )
    
    workflow.add_edge("omdb_enrichment", "web_search")
    workflow.add_edge("web_search", "synthesize")
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
        welcome_msg += "- 🗄️ Requêtes SQL (toujours en premier)\n"
        welcome_msg += "- 🎬 Enrichissement OMDB automatique\n"
        welcome_msg += "- 🌐 Recherche web (si nécessaire)\n\n"
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
if prompt := st.chat_input("Pose ta question..."):
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
            "sql_result": "",
            "omdb_result": "",
            "needs_web": False,
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
                status_placeholder.info("📂 Chargement du catalogue...")
            elif current_step == "question_clear":
                status_placeholder.info("✅ Question analysée...")
            elif current_step == "sql_executed":
                status_placeholder.info("🗄️ Données SQL récupérées...")
            elif current_step == "omdb_enriched":
                status_placeholder.info("🎬 Enrichissement OMDB...")
            elif current_step == "web_searched":
                status_placeholder.info("🌐 Recherche web...")
            elif current_step == "synthesis_complete":
                status_placeholder.success("💬 Réponse prête !")
        
        if result:
            status_placeholder.empty()
            
            # Get the last message (should be from synthesize_node)
            final_message = result["messages"][-1]
            response_text = final_message.content
            
            response_placeholder.markdown(response_text)
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Update session state
            st.session_state.agent_messages = [result["messages"][0], result["messages"][-1]]  # Keep only user + final
            st.session_state.db_catalog = result.get("db_catalog", st.session_state.db_catalog)
            st.session_state.sources = result.get("sources_used", [])
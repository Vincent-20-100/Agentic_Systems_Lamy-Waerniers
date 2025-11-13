"""
SQL Agent with Sequential Flow: SQL → OMDB → Web → Synthesis
V3.1: STRUCTURAL FIXES ONLY (HITL flow + checkpointing)
Launch: streamlit run albert_v3.1.py
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
from langgraph.checkpoint.memory import MemorySaver  # ← STRUCTURAL FIX: Add checkpointing
from dotenv import load_dotenv
import pathlib

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="Albert Query v3.1", page_icon="🧙‍♂️", layout="wide")

# Custom CSS for chat styling (user messages on right)
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

# API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Absolute path to databases
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# ← STRUCTURAL FIX: Navigate to the actual repo location
PROJECT_ROOT = pathlib.Path("C:/Users/Vincent/GitHub/Vincent-20-100/Agentic_Systems_Project_Vlamy")
DB_FOLDER_PATH = str(PROJECT_ROOT / "data" / "databases")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === ENHANCED AGENT STATE ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    db_catalog: dict

    # Collected data
    sql_result: str
    omdb_result: str
    web_result: str

    # Conversation context
    original_question: str
    clarification_history: list
    user_intent: str

    # Control flags
    needs_clarification: bool
    clarification_question: str
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
            "current_step": "schema_error"
        }

    return {
        "db_catalog": catalog,
        "current_step": "schema_loaded"
    }

def clarify_question_node(state: AgentState) -> dict:
    """Analyze question and ask for clarification if needed"""
    # Get current question
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    current_question = user_messages[-1].content if user_messages else ""

    # Store original question if not set
    original_q = state.get("original_question", "")
    if not original_q:
        original_q = current_question

    # Check if this is a response to clarification
    is_clarification_response = state.get("needs_clarification", False) and len(user_messages) > 1

    if is_clarification_response:
        # User answered clarification, merge context
        clarif_history = state.get("clarification_history", [])
        clarif_history.append({
            "question": state.get("clarification_question", ""),
            "answer": current_question
        })

        # Build full context
        full_question = f"{original_q}\n\nAdditional context: {current_question}"

        return {
            "original_question": original_q,
            "clarification_history": clarif_history,
            "needs_clarification": False,
            "current_step": "clarification_answered"
        }

    catalog_info = format_catalog_for_llm(state["db_catalog"])

    prompt = f"""Analyze this question: "{current_question}"

Available resources:
{catalog_info}
- OMDb API (detailed movie/series info)
- Web search (news, recent releases)

Determine if the question is:
- CLEAR: Can be answered directly with available data
- AMBIGUOUS: Needs clarification (missing which movie? which genre? which year?)

Examples of AMBIGUOUS questions:
- "a movie" → which specific movie?
- "when was it released?" → which title are you referring to?
- "good series" → what genre? what year? which platform?
- "the actor" → which actor specifically?

Respond ONLY in valid JSON format:
{{
  "status": "clear" or "ambiguous",
  "clarification": "Specific question to ask user (if ambiguous)",
  "reasoning": "Why clarification is needed"
}}"""

    response = llm.invoke(prompt)

    try:
        decision = json.loads(clean_json(response.content))

        if decision["status"] == "ambiguous":
            return {
                "original_question": original_q,
                "needs_clarification": True,
                "clarification_question": decision["clarification"],
                "current_step": "waiting_clarification"
            }
        else:
            return {
                "original_question": original_q,
                "needs_clarification": False,
                "current_step": "question_clear"
            }
    except:
        return {
            "original_question": original_q,
            "needs_clarification": False,
            "current_step": "question_clear"
        }

def intent_analyzer_node(state: AgentState) -> dict:
    """Analyze user intent to decide which tools to use"""
    original_q = state.get("original_question", "")
    clarif_history = state.get("clarification_history", [])

    # Build full context
    context = original_q
    if clarif_history:
        context += "\n\nClarifications:\n"
        for item in clarif_history:
            context += f"Q: {item['question']}\nA: {item['answer']}\n"

    catalog_info = format_catalog_for_llm(state["db_catalog"])

    prompt = f"""You are an intent analyzer. Determine which data sources are needed to answer this question.

QUESTION CONTEXT:
{context}

AVAILABLE DATABASES:
{catalog_info}

AVAILABLE TOOLS:
1. SQL Database - Local data about movies/series
2. OMDb API - Detailed movie metadata (ratings, cast, plot, awards)
3. Web Search - Recent news, current events, trending topics

DECISION RULES:
- Always try SQL FIRST if the data might be in our database
- Use OMDb to ENRICH SQL results with details (ratings, cast, plot)
- Use Web ONLY for: recent news, current events, things not in database
- NEVER skip SQL if question is about data we should have

EXAMPLES:
- "movies from 2020" → SQL: yes, OMDb: yes (enrich), Web: no
- "who starred in Inception" → SQL: maybe, OMDb: yes (primary), Web: no
- "latest Netflix releases today" → SQL: no, OMDb: no, Web: yes
- "best rated action movies" → SQL: yes, OMDb: yes (enrich), Web: no

Respond in JSON:
{{
  "intent": "Brief description of what user wants",
  "needs_sql": true/false,
  "needs_omdb": true/false,
  "needs_web": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Why these tools are needed"
}}"""

    response = llm.invoke(prompt)

    try:
        intent = json.loads(clean_json(response.content))

        return {
            "user_intent": intent.get("intent", "Unknown"),
            "needs_sql": intent.get("needs_sql", True),
            "needs_omdb": intent.get("needs_omdb", False),
            "needs_web": intent.get("needs_web", False),
            "confidence_score": intent.get("confidence", 0.5),
            "current_step": "intent_analyzed"
        }
    except:
        # Default: try SQL first
        return {
            "user_intent": "Unknown",
            "needs_sql": True,
            "needs_omdb": False,
            "needs_web": False,
            "confidence_score": 0.3,
            "current_step": "intent_analysis_failed"
        }

def sql_agent_node(state: AgentState) -> dict:
    """Execute SQL with better prompting"""
    catalog_info = format_catalog_for_llm(state["db_catalog"])
    original_q = state.get("original_question", "")
    clarif_history = state.get("clarification_history", [])
    user_intent = state.get("user_intent", "")

    # Build full context
    context = f"Original question: {original_q}\n"
    if clarif_history:
        context += "Clarifications:\n"
        for item in clarif_history:
            context += f"- {item['question']}: {item['answer']}\n"

    prompt = f"""You are an expert SQL query generator for SQLite databases.

USER CONTEXT:
{context}
Detected intent: {user_intent}

AVAILABLE DATABASES:
{catalog_info}

TASK: Generate a SQL query to answer the user's question.

BEST PRACTICES:
1. Use LIKE with wildcards for text search: WHERE title LIKE '%keyword%'
2. Always use LIMIT to prevent huge results (default: 10)
3. Use ORDER BY for rankings (DESC for highest first)
4. Double-check column names against the catalog above
5. For multiple tables, use explicit JOINs with table aliases

GOOD QUERY EXAMPLES:
- Find movie: SELECT * FROM movies WHERE title LIKE '%Inception%' LIMIT 5
- Top rated: SELECT * FROM movies ORDER BY rating DESC LIMIT 10
- By year: SELECT * FROM movies WHERE release_year = 2020 LIMIT 10
- Search: SELECT * FROM movies WHERE title LIKE '%action%' OR genre LIKE '%action%'

Respond ONLY in valid JSON:
{{
  "can_answer_with_sql": true or false,
  "confidence": 0.0-1.0,
  "db_name": "database_name",
  "query": "SELECT ... FROM ... WHERE ... LIMIT ...",
  "expected_columns": ["column1", "column2"],
  "reasoning": "Why this query answers the question"
}}"""

    response = llm.invoke(prompt)

    try:
        decision = json.loads(clean_json(response.content))

        if decision.get("can_answer_with_sql", False):
            sql_result = execute_sql_query.invoke({
                "query": decision["query"],
                "db_name": decision["db_name"],
                "state_catalog": state["db_catalog"]
            })

            source = f"🗄️ Database: {decision['db_name']}"

            return {
                "sql_result": sql_result,
                "sources_used": state.get("sources_used", []) + [source],
                "current_step": "sql_executed"
            }
        else:
            return {
                "sql_result": "[]",
                "current_step": "sql_no_data"
            }
    except Exception as e:
        return {
            "sql_result": json.dumps({"error": str(e)}),
            "current_step": "sql_error"
        }

def result_evaluator_node(state: AgentState) -> dict:
    """Evaluate if we need OMDB enrichment"""
    sql_result = state.get("sql_result", "[]")
    needs_omdb_flag = state.get("needs_omdb", False)

    # Check if SQL returned data
    try:
        data = json.loads(sql_result)
        has_data = isinstance(data, list) and len(data) > 0 and "error" not in str(data)

        # Only enrich with OMDB if:
        # 1. We have SQL data OR
        # 2. Intent analyzer said we need OMDB
        if has_data or needs_omdb_flag:
            return {
                "needs_omdb": True,
                "current_step": "evaluation_needs_omdb"
            }
        else:
            return {
                "needs_omdb": False,
                "current_step": "evaluation_skip_omdb"
            }
    except:
        return {
            "needs_omdb": needs_omdb_flag,
            "current_step": "evaluation_skip_omdb"
        }

def omdb_enrichment_node(state: AgentState) -> dict:
    """Enrich results with OMDB data"""
    sql_result = state.get("sql_result", "[]")
    original_q = state.get("original_question", "")

    try:
        data = json.loads(sql_result)

        # Try to extract title from SQL results
        title_to_search = None

        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            title_to_search = (first_item.get("title") or
                             first_item.get("Title") or
                             first_item.get("show_id") or
                             first_item.get("name") or "")

        # If no title from SQL, try to extract from question
        if not title_to_search and original_q:
            # Simple extraction: look for quoted text or capitalize words
            match = re.search(r'"([^"]+)"', original_q)
            if match:
                title_to_search = match.group(1)

        if title_to_search:
            omdb_result = omdb_api.invoke({
                "by": "title",
                "t": title_to_search,
                "plot": "full"
            })

            source = f"🎬 OMDb API: {title_to_search}"

            return {
                "omdb_result": omdb_result,
                "sources_used": state.get("sources_used", []) + [source],
                "current_step": "omdb_enriched"
            }

        return {
            "omdb_result": "{}",
            "current_step": "omdb_skipped"
        }
    except Exception as e:
        return {
            "omdb_result": "{}",
            "current_step": "omdb_error"
        }

def web_search_node(state: AgentState) -> dict:
    """Search web only if explicitly needed"""
    needs_web_flag = state.get("needs_web", False)
    original_q = state.get("original_question", "")
    sql_result = state.get("sql_result", "[]")

    # Skip web if intent analyzer said not needed
    if not needs_web_flag:
        return {
            "web_result": "{}",
            "current_step": "web_skipped"
        }

    prompt = f"""Decide if we need web search for additional information.

User question: "{original_q}"
SQL data available: {sql_result != "[]" and "error" not in sql_result}

ONLY search web if question requires:
- Recent news or current events ("latest", "trending", "today", "this week")
- Real-time data not in our database
- Current information that changes frequently

DO NOT search if:
- We have data from SQL/OMDB
- Question is about historical/static data
- General movie/series information

Respond in JSON:
{{
  "should_search": true/false,
  "search_query": "optimized query" (if should_search=true),
  "reasoning": "why search is/isn't needed"
}}"""

    response = llm.invoke(prompt)

    try:
        decision = json.loads(clean_json(response.content))

        if decision.get("should_search", False):
            web_result = web_search.invoke({"query": decision["search_query"]})
            urls = extract_urls(web_result)
            source = f"🌐 Web: [DuckDuckGo]({urls[0]})" if urls else "🌐 Web: DuckDuckGo"

            return {
                "web_result": web_result,
                "sources_used": state.get("sources_used", []) + [source],
                "current_step": "web_searched"
            }
        else:
            return {
                "web_result": "{}",
                "current_step": "web_skipped"
            }
    except Exception as e:
        return {
            "web_result": "{}",
            "current_step": "web_error"
        }

def synthesize_node(state: AgentState) -> dict:
    """Synthesize with confidence scoring"""
    sources = state.get("sources_used", [])
    sources_text = "\n".join([f"- {s}" for s in sources]) if sources else "- No external sources used"

    original_q = state.get("original_question", "")
    user_intent = state.get("user_intent", "")

    # Get all collected data
    sql_result = state.get("sql_result", "[]")
    omdb_result = state.get("omdb_result", "{}")
    web_result = state.get("web_result", "{}")

    # Validate data availability
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
        web_data = json.loads(web_result) if web_result != "{}" else {}
        has_web_data = bool(web_data) and "error" not in str(web_data)
    except:
        pass

    prompt = f"""You are Albert, a friendly data assistant helping users understand their data.

USER'S QUESTION: "{original_q}"
DETECTED INTENT: {user_intent}

AVAILABLE DATA:
---
SQL Database:
{sql_result if has_sql_data else "No data found in database"}

---
OMDB API:
{omdb_result if has_omdb_data else "No OMDB data available"}

---
Web Search:
{web_result if has_web_data else "No web data available"}

---

YOUR TASK:
1. Provide a clear, natural answer in ENGLISH
2. Use conversational, friendly tone
3. If you have relevant data, present it naturally (don't mention "SQL" or "API")
4. If data is incomplete, say what you know + what's missing
5. If no data at all, explain why and suggest alternatives
6. NEVER invent information - only use provided data
7. Always cite sources at the end

CONFIDENCE SCORING:
- High (0.8-1.0): Complete, relevant data answering the question fully
- Medium (0.5-0.8): Partial data, some information available
- Low (0.0-0.5): Little or no relevant data

Respond in JSON:
{{
  "answer": "Your natural, conversational answer in English",
  "confidence": 0.0-1.0,
  "missing_info": ["info1", "info2"] or [],
  "sources": {sources_text}
}}"""

    response = llm.invoke(prompt)

    try:
        synthesis = json.loads(clean_json(response.content))

        answer = synthesis.get("answer", "I couldn't process the information properly.")
        confidence = synthesis.get("confidence", 0.3)

        # Format final answer
        final_answer = f"{answer}\n\n**📚 Sources:**\n{sources_text}"

        return {
            "messages": [AIMessage(content=final_answer)],
            "confidence_score": confidence,
            "current_step": "synthesis_complete"
        }
    except:
        # Fallback response
        fallback = "I encountered an issue processing the data. Please try rephrasing your question."
        return {
            "messages": [AIMessage(content=fallback)],
            "confidence_score": 0.2,
            "current_step": "synthesis_error"
        }

def quality_check_node(state: AgentState) -> dict:
    """Check if answer quality is sufficient"""
    confidence = state.get("confidence_score", 0.0)

    if confidence < 0.4:
        # Low confidence - ask for clarification
        original_q = state.get("original_question", "")

        clarification = (
            f"I'm not confident in my answer (confidence: {confidence:.0%}). "
            f"Could you provide more details about your question: '{original_q}'?"
        )

        return {
            "needs_clarification": True,
            "clarification_question": clarification,
            "current_step": "quality_check_failed"
        }

    return {
        "current_step": "quality_check_passed"
    }

# === ROUTING ===

def route_after_clarify(state: AgentState) -> Literal["wait_user", "intent_analyzer"]:
    """Route based on clarification need"""
    return "wait_user" if state.get("needs_clarification") else "intent_analyzer"

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

def route_after_quality(state: AgentState) -> Literal["clarify_question", "end"]:
    """Route back to clarification or end"""
    return "clarify_question" if state.get("needs_clarification") else "end"

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
    """Build and compile the enhanced workflow with checkpointing"""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("clarify_question", clarify_question_node)
    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("result_evaluator", result_evaluator_node)
    workflow.add_node("omdb_enrichment", omdb_enrichment_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("quality_check", quality_check_node)
    workflow.add_node("wait_user", lambda state: state)
    workflow.add_node("end", lambda state: state)

    # Enhanced flow with loops
    workflow.add_edge(START, "get_schema")
    workflow.add_edge("get_schema", "clarify_question")

    workflow.add_conditional_edges(
        "clarify_question",
        route_after_clarify,
        {"wait_user": "wait_user", "intent_analyzer": "intent_analyzer"}
    )

    # ← STRUCTURAL FIX: Remove wait_user → END, keep it open for continuation
    # workflow.add_edge("wait_user", END)  # ❌ REMOVED - This was terminating the conversation!

    workflow.add_conditional_edges(
        "intent_analyzer",
        route_after_intent,
        {
            "sql_agent": "sql_agent",
            "omdb_enrichment": "omdb_enrichment",
            "web_search": "web_search"
        }
    )

    workflow.add_edge("sql_agent", "result_evaluator")

    workflow.add_conditional_edges(
        "result_evaluator",
        route_after_evaluator,
        {"omdb_enrichment": "omdb_enrichment", "web_search": "web_search"}
    )

    workflow.add_edge("omdb_enrichment", "web_search")
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", "quality_check")

    workflow.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {"clarify_question": "clarify_question", "end": "end"}
    )

    workflow.add_edge("end", END)

    # ← STRUCTURAL FIX: Add checkpointing for state persistence
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

# ← STRUCTURAL FIX: Add thread_id for checkpointing
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_session"

# Show welcome message on first load
if not st.session_state.welcome_shown:
    catalog = build_db_catalog(DB_FOLDER_PATH)
    st.session_state.db_catalog = catalog

    if catalog.get("error"):
        welcome_msg = f"⚠️ **Loading Error:** {catalog['error']}"
    else:
        welcome_msg = "##### 👋 **Hey! I'm Albert v3.1**\n\n###### 🧐 I'll help you make sense of your data!\n\n"
        welcome_msg += "\n\n ###### 📊 Available Databases:\n\n"

        for db_name, db_info in catalog["databases"].items():
            if "error" in db_info:
                welcome_msg += f"❌ **{db_name}**: {db_info['error']}\n"
                continue

            welcome_msg += f"**{db_name}** ({db_info['file_name']})\n"
            for table_name, table_info in db_info["tables"].items():
                cols = ", ".join([f"`{col['name']}`" for col in table_info["columns"]])
                welcome_msg += f"  • Table `{table_name}`: {cols}\n"
            welcome_msg += "\n"

        welcome_msg += "###### 🔧 Available Tools:\n"
        welcome_msg += "- 🗄️ SQL Queries (always checked first)\n"
        welcome_msg += "- 🎬 OMDb API enrichment (automatic)\n"
        welcome_msg += "- 🌐 Web search (when needed)\n\n"
        welcome_msg += "💬 **Go ahead, ask me anything!**"

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
            "original_question": "",
            "clarification_history": [],
            "user_intent": "",
            "needs_clarification": False,
            "clarification_question": "",
            "needs_sql": False,
            "needs_omdb": False,
            "needs_web": False,
            "sources_used": [],
            "confidence_score": 0.0,
            "current_step": ""
        }

        # ← STRUCTURAL FIX: Use config with thread_id for checkpointing
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        result = None

        for step in app.stream(inputs, config=config, stream_mode="values"):
            result = step
            current_step = step.get("current_step", "")

            # Show progress
            if current_step == "schema_loaded":
                status_placeholder.info("📂 Loading catalog...")
            elif current_step == "question_clear":
                status_placeholder.info("✅ Question analyzed...")
            elif current_step == "intent_analyzed":
                status_placeholder.info("🎯 Intent detected...")
            elif current_step == "sql_executed":
                status_placeholder.info("🗄️ SQL data retrieved...")
            elif current_step == "omdb_enriched":
                status_placeholder.info("🎬 OMDB enrichment...")
            elif current_step == "web_searched":
                status_placeholder.info("🌐 Web search...")
            elif current_step == "synthesis_complete":
                status_placeholder.success("💬 Answer ready!")
            elif current_step == "waiting_clarification":
                # Handle clarification request
                clarif_q = step.get("clarification_question", "")
                status_placeholder.empty()
                response_placeholder.markdown(f"🤔 {clarif_q}")

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"🤔 {clarif_q}"
                })

                # ← STRUCTURAL FIX: Don't break - the graph is now interrupted, not terminated
                break

        if result and result.get("current_step") != "waiting_clarification":
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

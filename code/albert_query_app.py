import streamlit as st
import os
import json
from typing import TypedDict, Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import pathlib
from utils import clean_json, build_db_catalog, format_catalog_for_llm, execute_sql_query, web_search, omdb_api


# Load environment
load_dotenv()

# Configuration
st.set_page_config(page_title="Albert query", page_icon="🧙‍♂️", layout="wide")
st.title("🧙‍♂️ Albert Query - Le magicien des données")
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
PROJECT_ROOT = pathlib.Path("C:/Users/Vincent/GitHub/Vincent-20-100/Agentic_Systems_Project_Vlamy")
DB_FOLDER_PATH = str(PROJECT_ROOT / "data" / "databases")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# === STRUCTURED OUTPUTS ===

class PlannerOutput(BaseModel):
    """Planner decision output"""
    resolved_query: str = Field(..., description="Query reformulated with context from history")
    planning_reasoning: str = Field(..., description="Why these tools are needed")
    needs_sql: bool = Field(default=False, description="Whether SQL query is needed")
    needs_omdb: bool = Field(default=False, description="Whether OMDB enrichment is needed")
    needs_web: bool = Field(default=False, description="Whether web search is needed")
    sql_query: Optional[str] = Field(None, description="Prepared SQL query if needed")
    omdb_query: Optional[str] = Field(None, description="Title to search in OMDB if needed")
    web_query: Optional[str] = Field(None, description="Web search query if needed")

class SQLOutput(BaseModel):
    """SQL execution decision"""
    can_answer: bool = Field(..., description="Whether SQL can answer the query")
    db_name: Optional[str] = Field(None, description="Database to query")
    query: Optional[str] = Field(None, description="SQL query to execute")
    reasoning: str = Field(..., description="Why this query or why SQL cannot answer")

# === AGENT STATE ===

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    db_catalog: dict
    # Planning
    original_question: str
    resolved_query: str
    planning_reasoning: str
    # Tool queries
    sql_query: str
    omdb_query: str
    web_query: str
    # Tool flags
    needs_sql: bool
    needs_omdb: bool
    needs_web: bool
    # Tool results
    sql_result: str
    omdb_result: str
    web_result: str
    # Metadata
    sources_used: list
    sources_detailed: list
    current_step: str

# === WORKFLOW NODES ===

def planner_node(state: AgentState) -> dict:
      """Analyze query and plan tool usage"""
      question = state.get("original_question", "")
      history = state.get("messages", [])
      catalog = state.get("db_catalog", {})

      # Format tool responses clearly
      sql_res = state.get("sql_result", "")
      omdb_res = state.get("omdb_result", "")
      web_res = state.get("web_result", "")
      catalog_info = format_catalog_for_llm(catalog)

      prompt = f"""You are a planning agent you have to interpet user questions and decide which tools to use to answer them.
  Your goal is to help the user explore movie/series data across multiple SQL databases, an external movie API (OMDB), and web search.
  You must decide which tools are needed, prepare queries for them, and resolve any ambiguous references from the conversation history.

  ORIGINAL QUESTION: "{question}"

  CONVERSATION HISTORY (last 5 messages):
  {json.dumps([{"role": m.type, "content": m.content} for m in history[-5:]], indent=2)}

  {catalog_info}

  AVAILABLE DATA:
  --- SQL Results ---
  {sql_res}

  --- OMDB Data ---
  {omdb_res}

  --- Web Results ---
  {web_res}

  AVAILABLE TOOLS:
  1. SQL Database - Query movies/series data (titles, ratings, years, genres, cast)
  2. OMDB API - Get posters, detailed plot, actors, awards (requires movie TITLE from SQL!)
  3. Web Search - Recent news, trending topics only

  DECISION LOGIC:
  - If question needs movie data → SQL first to get titles
  - If SQL returned titles AND user wants poster/details → Use OMDB with exact title from SQL
  - If question is about schema/structure → NO tools needed (catalog has everything)
  - If you have enough data to answer OR iteration >= 3 → set all needs_* to False

  INSTRUCTIONS:
  - Resolve references from history (e.g., "that movie" → actual movie name)
  - If OMDB is needed, extract the EXACT movie title from SQL results and put it in omdb_query
  - For poster requests: SQL first (iteration 1), then OMDB (iteration 2)
  - Only set one tool to True per iteration (sequential execution)
  - If you already have sufficient data to answer → set all to False

  OUTPUT: Structured decision with resolved query and tool flags"""

      structured_llm = llm.with_structured_output(PlannerOutput)

      try:
          plan = structured_llm.invoke(prompt)

          return {
              "resolved_query": plan.resolved_query,
              "planning_reasoning": plan.planning_reasoning,
              "needs_sql": plan.needs_sql,
              "needs_omdb": plan.needs_omdb,
              "needs_web": plan.needs_web,
              "sql_query": plan.sql_query or "",
              "omdb_query": plan.omdb_query or "",
              "web_query": plan.web_query or "",
              "current_step": "planned",
          }
      except Exception as e:
          return {
              "resolved_query": question,
              "planning_reasoning": f"Planning error: {str(e)}",
              "needs_sql": False,
              "needs_omdb": False,
              "needs_web": False,
              "sql_query": "",
              "omdb_query": "",
              "web_query": "",
              "current_step": "planned"
          }

def sql_node(state: AgentState) -> dict:
    """Execute SQL query"""
    catalog = state.get("db_catalog", {})
    resolved_query = state.get("resolved_query", "")
    catalog_info = format_catalog_for_llm(catalog)

    prompt = f"""Generate a precise SQL query to answer: "{resolved_query}"

{catalog_info}

CRITICAL INSTRUCTIONS:
1. **Table Names**: Use the EXACT table name from the catalog (e.g., 'shows', NOT 'movies' or 'netflix')
2. **Type Filtering**:
   - For movies: WHERE type = 'Movie'
   - For TV shows: WHERE type = 'TV Show'
   - Check the 'type' column unique values in the catalog above
3. **Genre Search**:
   - Genres are in 'listed_in' column as comma-separated strings
   - Use: WHERE listed_in LIKE '%Action%' or '%Action & Adventure%'
   - Check the 'ALL INDIVIDUAL GENRES' list in the catalog to use exact genre names
4. **Year Filtering**:
   - Column: 'release_year' (INTEGER)
   - For 2000s: WHERE release_year >= 2000 AND release_year <= 2009
   - Check the unique values range in the catalog
5. **Text Search**:
   - Title: WHERE title LIKE '%keyword%'
   - Description: WHERE description LIKE '%keyword%'
6. **Always use LIMIT**: Default 10, maximum 50
7. **ORDER BY**: Use DESC for highest first (ratings, year)

EXAMPLE QUERIES:
- "action movies from 2000s":
  SELECT * FROM shows WHERE type = 'Movie' AND listed_in LIKE '%Action%' AND release_year BETWEEN 2000 AND 2009 LIMIT 10

- "top rated comedies":
  SELECT * FROM shows WHERE type = 'Movie' AND listed_in LIKE '%Comed%' ORDER BY rating DESC LIMIT 10

OUTPUT: SQL decision with database name and query"""

    structured_llm = llm.with_structured_output(SQLOutput)
    
    try:
        decision = structured_llm.invoke(prompt)
        
        if decision.can_answer and decision.query and decision.db_name:
            result = execute_sql_query.invoke({
                "query": decision.query,
                "db_name": decision.db_name,
                "state_catalog": catalog
            })

            # Extract table name from query
            table_name = "unknown"
            if "FROM" in decision.query.upper():
                try:
                    from_clause = decision.query.upper().split("FROM")[1].split()[0]
                    table_name = from_clause.strip()
                except:
                    pass

            # Create detailed source
            detailed_source = {
                "type": "database",
                "name": decision.db_name.replace("_", " ").title(),
                "details": f"Table: {table_name}"
            }

            return {
                "sql_result": result,
                "sources_used": state.get("sources_used", []) + [f"DB: {decision.db_name}"],
                "sources_detailed": state.get("sources_detailed", []) + [detailed_source],
                "current_step": "sql_executed"
            }
        else:
            return {
                "sql_result": json.dumps({"info": decision.reasoning}),
                "current_step": "sql_skipped"
            }
    except Exception as e:
        return {
            "sql_result": json.dumps({"error": str(e)}),
            "current_step": "sql_error"
        }

def omdb_node(state: AgentState) -> dict:
    """Execute OMDB query"""
    omdb_query = state.get("omdb_query", "")
    sql_result = state.get("sql_result", "[]")
    
    # Try to get title from OMDB query or SQL result
    title = omdb_query
    
    if not title:
        try:
            data = json.loads(sql_result)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                title = item.get("title") or item.get("Title") or item.get("name")
        except:
            pass
    
    if title:
        try:
            result = omdb_api.invoke({"by": "title", "t": title, "plot": "full"})

            # Try to extract IMDb ID for clickable link
            imdb_url = None
            try:
                result_data = json.loads(result)
                if "imdbID" in result_data:
                    imdb_id = result_data["imdbID"]
                    imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
            except:
                pass

            # Create detailed source
            detailed_source = {
                "type": "omdb",
                "name": f"OMDB: {title}",
                "url": imdb_url
            }

            return {
                "omdb_result": result,
                "sources_used": state.get("sources_used", []) + [f"OMDB: {title}"],
                "sources_detailed": state.get("sources_detailed", []) + [detailed_source],
                "current_step": "omdb_executed"
            }
        except Exception as e:
            return {
                "omdb_result": json.dumps({"error": str(e)}),
                "current_step": "omdb_error"
            }
    
    return {
        "omdb_result": "{}",
        "current_step": "omdb_skipped"
    }

def web_node(state: AgentState) -> dict:
    """Execute web search"""
    web_query = state.get("web_query", "")
    
    if not web_query:
        web_query = state.get("resolved_query", "")
    
    try:
        result = web_search.invoke({"query": web_query, "num_results": 5})

        # Create DuckDuckGo search URL
        import urllib.parse
        search_url = f"https://duckduckgo.com/?q={urllib.parse.quote(web_query)}"

        # Create detailed source
        detailed_source = {
            "type": "web",
            "name": "Web Search",
            "url": search_url
        }

        return {
            "web_result": result,
            "sources_used": state.get("sources_used", []) + ["Web Search"],
            "sources_detailed": state.get("sources_detailed", []) + [detailed_source],
            "current_step": "web_executed"
        }
    except Exception as e:
        return {
            "web_result": json.dumps({"error": str(e)}),
            "current_step": "web_error"
        }

def synthesizer_node(state: AgentState) -> dict:
      """Generate final response"""
      question = state.get("original_question", "")
      resolved = state.get("resolved_query", "")
      reasoning = state.get("planning_reasoning", "")
      sql = state.get("sql_result", "[]")
      omdb = state.get("omdb_result", "{}")
      web = state.get("web_result", "{}")
      sources = state.get("sources_used", [])
      catalog = state.get("db_catalog", {})

      # Include catalog info for schema-related questions
      catalog_info = format_catalog_for_llm(catalog)

      prompt = f"""Generate a natural, helpful response in the user language using all available data.

  ORIGINAL QUESTION: "{question}"
  RESOLVED QUERY: "{resolved}"
  PLANNING CONTEXT: {reasoning}

  DATABASE SCHEMA (use this to answer questions about database structure):
  {catalog_info}

  AVAILABLE DATA:
  --- SQL Results ---
  {sql}

  --- OMDB Data ---
  {omdb}

  --- Web Results ---
  {web}

  SOURCES: {', '.join(sources)}

  GENERAL INSTRUCTIONS:
  - Answer naturally and clearly in French
  - Use DATABASE SCHEMA for structure questions (no SQL needed)
  - Integrate all source data seamlessly
  - Cite sources when mentioning facts
  - Keep responses concise but complete
  - Use natural language, not raw JSON dumps"""

      response = llm.invoke(prompt)

      return {
          "messages": [AIMessage(content=response.content)],
          "current_step": "complete"
      }

# === ROUTING ===

def should_run_sql(state: AgentState) -> bool:
    """Check if SQL should run"""
    return state.get("needs_sql", False)

def should_run_omdb(state: AgentState) -> bool:
    """Check if OMDB should run"""
    return state.get("needs_omdb", False)

def should_run_web(state: AgentState) -> bool:
    """Check if web search should run"""
    return state.get("needs_web", False)

def route_from_planner(state: AgentState) -> str:
    """Route from planner to first tool or synthesizer"""
    iteration = state.get("iteration_count", 0)
    if iteration > 5:
          return "synthesize"    # Prevent infinite loops (max 5 iterations)
    if state.get("needs_sql"):
        return "sql"
    elif state.get("needs_omdb"):
        return "omdb"
    elif state.get("needs_web"):
        return "web"
    else:
        return "synthesize"

def route_from_sql(state: AgentState) -> str:
    """Route from SQL to next tool"""
    if state.get("needs_omdb"):
        return "omdb"
    elif state.get("needs_web"):
        return "web"
    else:
        return "synthesize"

def route_from_omdb(state: AgentState) -> str:
    """Route from OMDB to next tool"""
    if state.get("needs_sql"):
        return "sql"
    if state.get("needs_web"):
        return "web"
    else:
        return "synthesize"
    
def route_from_web(state: AgentState) -> str:
    """Route from Web to next tool"""
    if state.get("needs_sql"):
        return "sql"
    if state.get("needs_omdb"):
        return "omdb"
    else:
        return "synthesize"

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
      """Build workflow with iteration loop"""
      workflow = StateGraph(AgentState)

      workflow.add_node("planner", planner_node)
      workflow.add_node("sql", sql_node)
      workflow.add_node("omdb", omdb_node)
      workflow.add_node("web", web_node)
      workflow.add_node("synthesize", synthesizer_node)

      workflow.add_edge(START, "planner")

      workflow.add_conditional_edges("planner", route_from_planner, ["sql", "omdb", "web", "synthesize"])
      workflow.add_conditional_edges("sql", route_from_sql, ["omdb", "web", "synthesize"])
      workflow.add_conditional_edges("omdb", route_from_omdb, ["sql", "web", "synthesize"])
      workflow.add_conditional_edges("web", route_from_web, ["sql", "omdb", "synthesize"])

      workflow.add_edge("synthesize", END)

      checkpointer = MemorySaver()
      return workflow.compile(checkpointer=checkpointer)

app = build_agent()

# === STREAMLIT INTERFACE ===

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "db_catalog" not in st.session_state:
    catalog = build_db_catalog(DB_FOLDER_PATH)
    st.session_state.db_catalog = catalog
    
    # Welcome message
    if catalog.get("error"):
        welcome = f"❌ Error: {catalog['error']}"
    else:
        welcome = "##### 👋 **Salut, moi c'est Albert Query**\n\n"
        welcome += "##### Je suis là pour t'aider à explorer tes bases de données !\n\n"
        welcome += "\n\n"
        welcome += "**Bases de données disponibles:**\n"
        for db_name, db_info in catalog["databases"].items():
            if "error" not in db_info:
                welcome += f" {db_name} -\n"
        welcome += "**Outils disponibles**: Requête de base de données SQL / Requête d'API OMDB / Recherche Web\n\n**Demande-moi quelque chose pour commencer !**\n\nPar exemple :\n- Quelles tables et colonnes sont disponibles dans les bases de données ?\n- Combien de genres différents sont disponibles dans toutes les bases ?\n- Trouve-moi 5 films d'action des années 2000 dans Netflix.\n- Quels sont les films avec les meilleures notes sur Disney Plus ?\n- Donne-moi des comédies récentes sur Amazon Prime."
    
    st.session_state.chat_messages.append({"role": "assistant", "content": welcome})

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"

# Display chat
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            st.markdown("---")
            st.caption("📚 Sources utilisées:")
            cols = st.columns(len(msg["sources"]))
            for idx, source in enumerate(msg["sources"]):
                with cols[idx]:
                    if source.get("type") == "database":
                        st.markdown(f"🗄️ **{source['name']}**")
                        if "details" in source:
                            st.caption(source["details"])
                    elif source.get("type") == "omdb":
                        if source.get("url"):
                            st.markdown(f"🎬 [{source['name']}]({source['url']})")
                        else:
                            st.markdown(f"🎬 **{source['name']}**")
                    elif source.get("type") == "web":
                        if source.get("url"):
                            st.markdown(f"🌐 [Web Search]({source['url']})")
                        else:
                            st.markdown(f"🌐 **Web Search**")

# User input
if prompt := st.chat_input("Your question..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        status = st.empty()
        response_placeholder = st.empty()
        
        inputs = {
            "messages": st.session_state.agent_messages,
            "db_catalog": st.session_state.db_catalog,
            "original_question": prompt,
            "resolved_query": "",
            "planning_reasoning": "",
            "sql_query": "",
            "omdb_query": "",
            "web_query": "",
            "needs_sql": False,
            "needs_omdb": False,
            "needs_web": False,
            "sql_result": "[]",
            "omdb_result": "{}",
            "web_result": "{}",
            "sources_used": [],
            "sources_detailed": [],
            "current_step": ""
        }
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        result = None
        for step in app.stream(inputs, config=config, stream_mode="values"):
            result = step
            current = step.get("current_step", "")
            
            if current == "planned":
                status.info("🧠 Albert réfléchit...")
            elif current == "sql_executed":
                status.info("💾 Albert interroge la base de données SQL...")
            elif current == "omdb_executed":
                status.info("🎬 Albert interroge l'API OMDB...")
            elif current == "web_executed":
                status.info("🌐 Albert recherche sur le web...")
            elif current == "complete":
                status.success("✅ Terminé !")
        
        if result:
            status.empty()

            final_msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
            if final_msgs:
                response = final_msgs[-1].content
                response_placeholder.markdown(response)

                # Get detailed sources
                sources_detailed = result.get("sources_detailed", [])

                # Display sources below response
                if sources_detailed:
                    st.markdown("---")
                    st.caption("📚 Sources utilisées:")
                    cols = st.columns(len(sources_detailed))
                    for idx, source in enumerate(sources_detailed):
                        with cols[idx]:
                            if source.get("type") == "database":
                                st.markdown(f"🗄️ **{source['name']}**")
                                if "details" in source:
                                    st.caption(source["details"])
                            elif source.get("type") == "omdb":
                                if source.get("url"):
                                    st.markdown(f"🎬 [{source['name']}]({source['url']})")
                                else:
                                    st.markdown(f"🎬 **{source['name']}**")
                            elif source.get("type") == "web":
                                if source.get("url"):
                                    st.markdown(f"🌐 [Recherche Web]({source['url']})")
                                else:
                                    st.markdown(f"🌐 **Recherche Web**")

                # Save message with sources
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources_detailed
                })

                # Keep last user + assistant in agent messages
                user_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
                if user_msgs:
                    st.session_state.agent_messages = [user_msgs[-1], final_msgs[-1]]
import streamlit as st
import os
import json
import urllib.parse
import pathlib
from typing import TypedDict, Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import operator
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

@st.cache_resource
def get_db_catalog():
    return build_db_catalog(DB_FOLDER_PATH)

# === STRUCTURED OUTPUTS ===

class PlannerOutput(BaseModel):
    """Planner decision output"""
    resolved_query: str = Field(..., description="Query reformulated with context from history")
    planning_reasoning: str = Field(..., description="Why these tools are needed")
    needs_sql: bool = Field(default=False, description="Whether SQL query is needed")
    needs_omdb: bool = Field(default=False, description="Whether OMDB enrichment is needed")
    needs_web: bool = Field(default=False, description="Whether web search is needed")
    instructions_for_sql: str = Field(default="", description="Instructions pour le node SQL")
    omdb_title: str = Field(default="", description="Exact movie title for the OMDB API call search by title")
    instructions_for_web: str = Field(default="", description="Instructions pour le node Web")

class SQLOutput(BaseModel):
    """SQL execution decision"""
    can_answer: bool = Field(..., description="Whether SQL can answer the query")
    db_name: Optional[str] = Field(None, description="Database to query")
    query: Optional[str] = Field(None, description="SQL query to execute")
    reasoning: str = Field(..., description="Why this query or why SQL cannot answer")

class OMDBOutput(BaseModel):
    """OMDB query decision"""
    title: Optional[str] = Field(None, description="Exact movie title or None")
    plot: str = Field("full", description="short or full")
    reasoning: str = Field(..., description="Why these choices")

class WebOutput(BaseModel):
    """Web search query construction"""
    search_query: str = Field(..., description="Optimized search query for DuckDuckGo (3-8 words)")
    reasoning: str = Field(..., description="Why this search query was chosen")

class SynthesizerOutput(BaseModel):
    """Synthesizer response and completeness evaluation"""
    response: str = Field(..., description="Natural language response, integrating all available data")
    needs_more_data: bool = Field(False, description="Whether additional data collection is needed")
    missing_info: Optional[str] = Field(None, description="What specific information is missing (if needs_more_data=True)")
    reasoning: str = Field(..., description="Evaluation of data completeness and response quality")

# === AGENT STATE ===

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
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
    sources_detailed: Annotated[list, operator.add]
    current_step: str

# === WORKFLOW NODES ===

def planner_node(state: AgentState) -> dict:
    """Decide which tools to use and provide instructions"""
    question = state.get("original_question", "")
    history = state.get("messages", [])
    previous_synthesis = state.get("synthesis", "")

    sql_res = state.get("sql_result", "")
    omdb_res = state.get("omdb_result", "")
    web_res = state.get("web_result", "")
    catalog = get_db_catalog()
    catalog_info = format_catalog_for_llm(catalog)

    prompt = f"""You are a planning agent. Decide which tools to use and provide instructions.

  PREVIOUS SYNTHESIS (none is first iteration): "{previous_synthesis}"

  QUESTION: "{question}"

  HISTORY (last 3): {json.dumps([{"role": m.type, "content": m.content[:100]} for m in history[-3:]], indent=2)}

  DATABASE CATALOG:{catalog_info}

  PREVIOUS DATA COLLECTED:
  - SQL: {('✅ ' + sql_res[:100] + '...') if sql_res else '❌ Not retrieved'}
  - OMDB: {('✅ ' + omdb_res[:100] + '...') if omdb_res else '❌ Not retrieved'}
  - Web: {('✅ ' + web_res[:100] + '...') if web_res else '❌ Not retrieved'}

  AVAILABLE TOOLS:
  1. SQL Database - Query movies/series (titles, ratings, cast, genres)
     Available DBs: {', '.join(get_db_catalog()['databases'].keys())}
  2. OMDB API - Get posters, detailed plots, awards (needs exact title)
  3. Web Search - Recent news, reviews, trending info

  DECISION LOGIC:
  - Structured movie data → needs_sql=True
  - Have title, need poster/details → needs_omdb=True
  - Current events/reviews → needs_web=True

  EXAMPLES:
  Q: "Gladiator poster" → needs_sql=True ("Find 'Gladiator' exact title"), needs_omdb=True ("Get poster for Gladiator")
  Q: "10 films in 2020" → needs_sql=True only ("Find 10 random movies")
  Q: "Latest Marvel news" → needs_web=True only ("Search Marvel news 2024")

  FOR OMDB OUTPUT: Just extract title for the API call search by title.
  
  Provide clear, actionable instructions for each enabled tool."""

    structured_llm = llm.with_structured_output(PlannerOutput)

    try:
        plan = structured_llm.invoke(prompt)

        return {
            "resolved_query": plan.resolved_query,
            "planning_reasoning": plan.planning_reasoning,
            "needs_sql": plan.needs_sql,
            "needs_omdb": plan.needs_omdb,
            "needs_web": plan.needs_web,
            "instructions_for_sql": plan.instructions_for_sql,
            "omdb_query": plan.omdb_title,
            "instructions_for_web": plan.instructions_for_web,
            "current_step": "planned"
        }
    except Exception as e:
        return {
            "resolved_query": question,
            "planning_reasoning": f"Planning error: {str(e)}",
            "needs_sql": False,
            "needs_omdb": False,
            "needs_web": False,
            "instructions_for_sql": "",
            "omdb_query": "",
            "instructions_for_web": "",
            "current_step": "planned"
        }

def sql_node(state: AgentState) -> dict:
    """Execute SQL query - constructs query based on planner instructions"""
    instructions = state.get("instructions_for_sql", "")
    resolved_query = state.get("resolved_query", "")
    catalog = get_db_catalog()
    catalog_info = format_catalog_for_llm(catalog)

    prompt = f"""Generate SQL query from planner instructions.

  INSTRUCTIONS: "{instructions}"
  CONTEXT: "{resolved_query}"

  {catalog_info}

  SQL RULES:
  1. Use EXACT table/column names from catalog above
  2. Text search: LIKE '%keyword%' (case-insensitive)
  3. Comma-separated fields (genres, cast): LIKE '%value%'
  4. Top/best queries: ORDER BY rating/score DESC
  5. Always add LIMIT (default 10, max 50)
  6. Check unique values in catalog to validate filters

  EXAMPLES:
  - "Find Inception" → SELECT * FROM movies WHERE title LIKE '%Inception%' LIMIT 1
  - "5 action films" → SELECT * FROM movies WHERE listed_in LIKE '%Action%' LIMIT 5

  Provide database name and query."""

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

            detailed_source = {
                "type": "database",
                "name": decision.db_name.replace("_", " ").title(),
                "details": f"Table: {table_name}"
            }

            return {
                "sql_result": result,
                "sources_detailed": [detailed_source]
            }
        else:
            return {
                "sql_result": json.dumps({"info": decision.reasoning}),
                "current_step": "sql_skipped"
            }
    except Exception as e:
        return {"sql_result": json.dumps({"error": str(e)})}

def omdb_node(state: AgentState) -> dict:
      """Execute OMDB query - simple and robust"""
      omdb_query = state.get("omdb_query", "")
      sql_result = state.get("sql_result", "[]")

      # Try to get title from omdb_query (from planner) or SQL result (fallback)
      title = omdb_query

      # Fallback: extract from SQL if omdb_query is empty
      # Note: In parallel execution, sql_result might be empty, that's OK
      if not title and sql_result != "[]":
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

              # Extract IMDb ID for clickable link
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
                  "sources_used": [f"OMDB: {title}"],
                  "sources_detailed": [detailed_source],
              }
          except Exception as e:
              return {
                  "omdb_result": json.dumps({"error": str(e)}),
              }

      return {
          "omdb_result": "{}",
      }
    
def web_node(state: AgentState) -> dict:
    """Execute web search - constructs query based on instructions"""
    instructions = state.get("instructions_for_web", "")
    resolved_query = state.get("resolved_query", "")
    
    prompt = f"""Create optimized web search query for DuckDuckGo.

  PLANNER INSTRUCTIONS: "{instructions}"
  USER CONTEXT: "{resolved_query}"

  SEARCH STRATEGIES:
  - Movie news → "[title] latest news 2024"
  - Reviews/critics → "[title] reviews ratings"
  - General info → "[title] plot cast summary"
  - Trending topics → "[keyword] trending discussion"

  RULES:
  - 3-8 words maximum
  - No quotes, no special characters
  - Year context if relevant (2024, 2025)

  Output concise search query with reasoning."""

    try:
        structured_llm = llm.with_structured_output(WebOutput)
        decision = structured_llm.invoke(prompt)

        result = web_search.invoke({
            "query": decision.search_query,
            "num_results": 5
        })

        search_url = f"https://duckduckgo.com/?q={urllib.parse.quote(decision.search_query)}"

        detailed_source = {
            "type": "web",
            "name": "Web Search",
            "url": search_url
        }

        return {
            "web_result": result,
            "sources_detailed": [detailed_source]
        }
    except Exception as e:
        return {"web_result": json.dumps({"error": str(e)})}

def synthesizer_node(state: AgentState) -> dict:
    """Generate final response and decide if more data is needed"""
    question = state.get("original_question", "")
    reasoning = state.get("planning_reasoning", "")
    sql = state.get("sql_result", "[]")
    omdb = state.get("omdb_result", "{}")
    web = state.get("web_result", "{}")

    prompt = f"""Generate natural response using all available data.

  QUESTION: "{question}"
  PLANNING: {reasoning}

  DATA SOURCES:
  --- SQL Database ---
  {sql if sql != '[]' else '❌ Aucune donnée'}

  --- OMDB API ---
  {omdb if omdb != '{{}}' else '❌ Aucune donnée'}

  --- Web Search ---
  {web if web != '{{}}' else '❌ Aucune donnée'}

  FORMATTING RULES:
  1. **OMDB Posters**: If "Poster" field present → "Voici l'affiche : [URL]"
  2. **Data integration**: Seamlessly combine sources, don't dump JSON
  3. **Citations**: "Selon OMDB...", "Dans la base de données..."
  4. **Conciseness**: Max 200 words unless detailed analysis requested

  COMPLETENESS EVALUATION:
  - Missing critical data (poster requested but no OMDB)? → needs_more_data=True, missing_info="OMDB poster data"
  - All essential info present? → needs_more_data=False
  - Partial data but sufficient for answer? → needs_more_data=False

  Provide response, needs_more_data flag, and reasoning."""

    try:
      structured_llm = llm.with_structured_output(SynthesizerOutput)
      output = structured_llm.invoke(prompt)

      return {
          "synthesis": output.response,
          "needs_more_data": output.needs_more_data,
          "messages": [AIMessage(content=output.response)],
          "missing_info": output.missing_info,
          "reasoning": output.reasoning,
          "current_step": "synthesized"
      }
    except Exception as e:
        return {
          "synthesis": f"Erreur lors de la génération de la réponse: {str(e)}",
          "needs_more_data": False,
          "messages": [AIMessage(content=f"Erreur: {str(e)}")],
          "missing_info": "",
          "reasoning": "",
          "current_step": "synthesize_error"
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

def route_from_planner(state: AgentState) -> list:
      """Route from planner to tools in parallel using Send"""
      iteration = state.get("iteration_count", 0)
      if iteration > 5:
          return [Send("synthesize", state)]

      # Collect all tools to run in parallel
      sends = []
      if state.get("needs_sql"):
          sends.append(Send("sql", state))
      if state.get("needs_omdb"):
          sends.append(Send("omdb", state))
      if state.get("needs_web"):
          sends.append(Send("web", state))

      # If no tools needed, go to synthesizer
      if not sends:
          sends.append(Send("synthesize", state))

      return sends

def route_from_synthesizer(state: AgentState) -> str:
    """Decide if we need more data or can end"""
    needs_more = state.get("needs_more_data", False)
    
    if needs_more:
        return "planner"
    else:
        return END

# === BUILD GRAPH ===

@st.cache_resource
def build_agent():
    """Build workflow with parallel tool execution"""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("omdb", omdb_node)
    workflow.add_node("web", web_node)
    workflow.add_node("synthesize", synthesizer_node)

    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges("planner", route_from_planner, ["sql", "omdb", "web", "synthesize"])
    workflow.add_edge("sql", "synthesize")
    workflow.add_edge("omdb", "synthesize")
    workflow.add_edge("web", "synthesize")
    workflow.add_conditional_edges("synthesize", route_from_synthesizer, ["planner", END])

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

app = build_agent()

# === STREAMLIT INTERFACE ===

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
    
    catalog = get_db_catalog()

    # Welcome message
    welcome = "##### 👋 **Salut, moi c'est Albert Query**\n\n"
    welcome += "##### Je suis là pour t'aider à explorer tes bases de données !\n\n"
    welcome += "\n\n"
    welcome += "**Bases de données disponibles:**\n"
    for db_name, db_info in catalog["databases"].items():
        if "error" not in db_info:
            welcome += f" {db_name} -\n"
    welcome += "**Outils disponibles**: Requête de base de données SQL / Requête d'API OMDB / Recherche Web\n\n"
    welcome += "**Demande-moi quelque chose pour commencer !**\n\n"
    welcome += "Par exemple :\n"
    welcome += "- Quelles tables et colonnes sont disponibles dans les bases de données ?\n"
    welcome += "- Combien de genres différents sont disponibles dans toutes les bases ?\n"
    welcome += "- Trouve-moi 5 films d'action des années 2000 dans Netflix.\n"
    welcome += "- Quels sont les films avec les meilleures notes sur Disney Plus ?\n"
    welcome += "- Donne-moi des comédies récentes sur Amazon Prime.\n"
        
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
            "sources_detailed": [],
            "current_step": ""
        }
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        result = None
        for step in app.stream(inputs, config=config, stream_mode="values"):
            result = step
            current = step.get("current_step", "")
            
            if current == "planned" or current == "synthesized":
                status.info("🧠 Albert réfléchit...")
            elif current == "collected":
                status.info("🔄 Albert collecte les données...")
        
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
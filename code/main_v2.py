# === IMPORT ===
import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tool_v2 import execute_sql_query, web_search, omdb_api, get_db_schema

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
OMDB_BASE_URL = "http://www.omdbapi.com/"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manquant")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === STATE ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    schema: str  # Data base schema
    next_tool: str  # Tool to call ("sql", "web", "omdb", "none")

# === LLM WITH TOOLS ===
tools = [execute_sql_query, web_search, omdb_api]
llm_with_tools = llm.bind_tools(tools)

# === WORKFLOW NODES ===

def get_schema_node(state: AgentState) -> dict:
    """Charge le schéma de la base de données."""
    schema = get_db_schema.invoke({})
    return {
        "schema": schema,
        "messages": [AIMessage(content=f"✅ Schéma chargé")]
    }

def chief_agent_node(state: AgentState) -> dict:
    """Le chief analyse la requête et appelle l'outil approprié."""
    prompt = f"""Tu es un assistant SQL/données. Tu as accès à :
1. execute_sql_query : pour interroger netflix.db
2. web_search : pour recherches web générales
3. omdb_api : pour infos précises sur films/séries (IMDb, réalisateur, etc.)

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
        
        return {
            "next_tool": next_tool,
            "messages": [response]
        }
    else:
        return {
            "next_tool": "none",
            "messages": [response]
        }

def tool_executor_node(state: AgentState) -> dict:
    """Exécute l'outil choisi par le chief."""
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
    """Synthétise les résultats en réponse finale."""
    prompt = """Tu es un assistant qui synthétise les résultats.
Fournis une réponse claire, concise et en français à l'utilisateur.
Utilise les données disponibles dans l'historique des messages."""
    
    messages = [{"role": "system", "content": prompt}] + [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# === ROUTING ===

def route_after_chief(state: AgentState) -> Literal["tool_executor", "synthesize"]:
    """Détermine si on exécute un outil ou si on synthétise directement."""
    return "tool_executor" if state["next_tool"] != "none" else "synthesize"

# === GRAPH BUILD ===

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("get_schema", get_schema_node)
workflow.add_node("chief_agent", chief_agent_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("synthesize", synthesize_node)

# Edges
workflow.add_edge(START, "get_schema")
workflow.add_edge("get_schema", "chief_agent")
workflow.add_conditional_edges(
    "chief_agent",
    route_after_chief,
    {"tool_executor": "tool_executor", "synthesize": "synthesize"}
)
workflow.add_edge("tool_executor", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()

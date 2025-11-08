#####################################################################################
#                                     IMPORTS                                       #
#####################################################################################

import os
import json
import sqlite3
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages


#####################################################################################
#                                      TOOLS                                        #
#####################################################################################

# tool 1: execute_sql_query
@tool
def execute_sql_query(sql_query: str, db_path: str) -> str:
    """
    Executes a SQL query on a local SQLite database.
    Args:
        sql_query (str): SQL query to execute.
        db_path (str): Path to the SQLite database file.
    Returns:
        str: Results formatted as Markdown table or error message.
    """
    try:
        connect = sqlite3.connect(db_path)
        cursor = connect.cursor()
        cursor.execute(sql_query)

        # Handle SELECT queries
        if sql_query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            if not rows:
                return "No results found."

            # Get column names
            column_names = [description[0] for description in cursor.description]
            # Format as Markdown table
            markdown_table = "  " + " | ".join(column_names) + " |\n"
            markdown_table += "|" + "|".join(["---"] * len(column_names)) + "|\n"
            for row in rows:
                markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            return markdown_table

        # Handle INSERT/UPDATE/DELETE
        else:
            connect.commit()
            return f"Query executed successfully. {cursor.rowcount} rows affected."

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    finally:
        connect.close()

# tool 2: get_db_schema
@tool
def get_db_schema(db_path: str = "./data") -> str:
    """
    Retrieves simplified schema information for all SQLite databases in JSON format.
    Args:
        db_path (str): Path to directory containing SQLite database files. Defaults to "../data".
    Returns:
        str: JSON string containing simplified schema information.
    """
    result = {
        "databases": [],
        "error": None
    }

    # Find all database files
    db_files = []
    for file in os.listdir(db_path):
        if file.endswith(('.db', '.sqlite', '.sqlite3')):
            db_files.append({
                "path": os.path.join(db_path, file),
                "name": os.path.basename(file)
            })

    if not db_files:
        result["error"] = "No SQLite database files found in the specified directory."
        return json.dumps(result, indent=2)

    for db_file in db_files:
        try:
            conn = sqlite3.connect(db_file["path"])
            cursor = conn.cursor()

            database = {
                "name": db_file["name"],
                "path": db_file["path"],
                "tables": []
            }

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]

            for table in tables:
                table_info = {
                    "name": table,
                    "columns": []    
                }

                # Get column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                for col in columns:
                    table_info["columns"].append({
                        "name": col[1],
                        "type": col[2]
                    })

                database["tables"].append(table_info)

            result["databases"].append(database)
            conn.close()

        except Exception as e:
            result["databases"].append({
                "name": db_file["name"],
                "error": str(e)
            })

    return json.dumps(result, indent=2)


#####################################################################################
#                                       APP                                         #
#####################################################################################

# Load environment variables from .env file
load_dotenv()

# Access the API keys
api_key = os.getenv("OPENAI_API_KEY")

# 1. Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], add_messages]

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
tools = [get_db_schema, execute_sql_query]
llm_with_tools = llm.bind_tools(tools)

# 3. Define nodes
def call_model(state: AgentState):
    """Call LLM with tools"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Execute tools"""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        # Compatibilité : dict ou objet
        name = getattr(tool_call, "name", tool_call.get("name"))
        args = getattr(tool_call, "args", tool_call.get("args"))
        call_id = getattr(tool_call, "id", tool_call.get("id"))

        if name == "get_db_schema":
            result = get_db_schema.invoke({"db_path": "../data"})
        else:
            result = execute_sql_query.invoke(args)

        outputs.append(ToolMessage(
            content=result,
            name=name,
            tool_call_id=call_id
        ))
    return {"messages": outputs}

def should_continue(state: AgentState) -> str:
    """Decide whether to continue"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"
    return "end"

# 4. Build graph
workflow = StateGraph(AgentState)
workflow.add_node("model", call_model)
workflow.add_node("tool", call_tool)
workflow.add_edge(START, "model")
workflow.add_conditional_edges(
    "model",
    should_continue,
    {"tool": "tool", "end": END}
)
workflow.add_edge("tool", "model")
app = workflow.compile()

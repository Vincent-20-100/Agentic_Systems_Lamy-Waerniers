import math
import re
import sqlite3
import csv
import os
import json
from datetime import datetime
from langchain_core.tools import tool
from datetime import datetime, timedelta
from IPython.display import Image, display


@tool
def execute_sql_query(query: str, db_path: str = "../data/netflix.db") -> str:
    """
    Executes a SQL query on a local SQLite database.
    Args:
        query (str): SQL query to execute.
    Returns:
        str: Results formatted as Markdown table or error message.
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(query)

        # Handle SELECT queries
        if query.strip().upper().startswith("SELECT"):
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
            connection.commit()
            return f"Query executed successfully. {cursor.rowcount} rows affected."

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    finally:
        connection.close()


@tool
def get_db_schema(db_path: str = "../data") -> str:
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
                "total_size_mb": round(os.path.getsize(db_file["path"]) / (1024 * 1024), 2),
                "tables": []
            }

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [table[0] for table in cursor.fetchall()]

            for table in tables:
                table_info = {
                    "name": table,
                    "columns": [],
                    "indexes": []
                }

                # Get column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                for col in columns:
                    table_info["columns"].append({
                        "name": col[1],
                        "type": col[2],
                        "nullable": col[3] == 0
                    })

                # Get indexes
                cursor.execute(f"PRAGMA index_list({table})")
                indexes = cursor.fetchall()
                for idx in indexes:
                    index_info = {
                        "name": idx[1],
                        "unique": idx[2] == 1,
                        "columns": []
                    }
                    cursor.execute(f"PRAGMA index_info({idx[1]})")
                    idx_cols = cursor.fetchall()
                    index_info["columns"] = [col[2] for col in idx_cols]
                    table_info["indexes"].append(index_info)

                database["tables"].append(table_info)

            result["databases"].append(database)
            conn.close()

        except Exception as e:
            result["databases"].append({
                "name": db_file["name"],
                "error": str(e)
            })

    return json.dumps(result, indent=2)



def display_graph(graph: any):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass

# Helper function for formatting the stream nicely
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
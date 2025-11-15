import os
import json
import sqlite3
import requests
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


# === HELPER FUNCTIONS ===

def clean_json(text: str) -> str:
    """Clean JSON from markdown"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def build_db_catalog(folder_path: str) -> dict:
    """Build database catalog with unique values for all columns"""
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
        catalog["error"] = "No databases found"
        return catalog

    for db_file in db_files:
        db_path = os.path.join(folder_path, db_file)
        db_name = os.path.splitext(db_file)[0]

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            db_info = {
                "file_name": db_file,
                "full_path": db_path,
                "tables": {}
            }

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                table_info = {
                    "columns": [
                        {"name": col[1], "type": col[2], "primary_key": bool(col[5])}
                        for col in columns
                    ],
                    "column_names": [col[1] for col in columns],
                    "unique_values": {}
                }

                # Get row count
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    table_info["row_count"] = cursor.fetchone()[0]
                except:
                    table_info["row_count"] = None

                # Get unique values for ALL columns
                for col_info in columns:
                    col_name = col_info[1]

                    try:
                        # Get distinct count first
                        cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL")
                        distinct_count = cursor.fetchone()[0]

                        # If reasonable number of unique values, get them
                        if distinct_count <= 50:
                            cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 50")
                            unique_vals = [row[0] for row in cursor.fetchall()]
                            table_info["unique_values"][col_name] = {
                                "count": distinct_count,
                                "values": unique_vals
                            }
                        else:
                            # For columns with many values, get sample + range
                            cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 20")
                            sample_vals = [row[0] for row in cursor.fetchall()]
                            table_info["unique_values"][col_name] = {
                                "count": distinct_count,
                                "sample": sample_vals
                            }
                    except:
                        pass

                # Special handling for genre columns (comma-separated)
                if "listed_in" in table_info["column_names"]:
                    try:
                        cursor.execute(f"SELECT DISTINCT listed_in FROM {table_name} WHERE listed_in IS NOT NULL")
                        all_genres = set()
                        for (genre_string,) in cursor.fetchall():
                            if genre_string:
                                genres = [g.strip() for g in str(genre_string).split(',')]
                                all_genres.update(genres)
                        table_info["unique_values"]["all_genres"] = sorted(list(all_genres))
                    except:
                        pass

                db_info["tables"][table_name] = table_info

            catalog["databases"][db_name] = db_info
            conn.close()

        except Exception as e:
            catalog["databases"][db_name] = {"error": str(e)}

    return catalog

def format_catalog_for_llm(catalog: dict) -> str:
    """Format catalog for LLM with all unique values"""
    if catalog.get("error"):
        return f"ERROR: {catalog['error']}"

    output = "DATABASE CATALOG:\n\n"

    for db_name, db_info in catalog["databases"].items():
        if "error" in db_info:
            output += f"❌ {db_name}: {db_info['error']}\n"
            continue

        output += f"═══ Database: {db_name} ═══\n\n"

        for table_name, table_info in db_info["tables"].items():
            output += f"TABLE: {table_name}\n"
            output += f"Total rows: {table_info.get('row_count', 'unknown')}\n\n"

            output += "COLUMNS:\n"
            for col in table_info["columns"]:
                pk_marker = " [PRIMARY KEY]" if col["primary_key"] else ""
                output += f"  • {col['name']} ({col['type']}){pk_marker}\n"

                # Add unique values info
                col_name = col['name']
                unique_info = table_info.get("unique_values", {}).get(col_name)

                if unique_info:
                    if "values" in unique_info:
                        # Full list of unique values
                        output += f"    → {unique_info['count']} unique values: "
                        vals_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in unique_info['values'][:20]])
                        output += vals_str
                        if len(unique_info['values']) > 20:
                            output += ", ..."
                        output += "\n"
                    elif "sample" in unique_info:
                        # Sample for columns with many values
                        output += f"    → {unique_info['count']} unique values (sample): "
                        vals_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in unique_info['sample'][:10]])
                        output += vals_str + ", ...\n"

            # Add genre breakdown if available
            if "all_genres" in table_info.get("unique_values", {}):
                output += f"\nALL INDIVIDUAL GENRES (from listed_in column):\n"
                genres = table_info["unique_values"]["all_genres"]
                output += f"  {', '.join(genres)}\n"

            output += "\n"

    return output

# === TOOLS ===

@tool
def execute_sql_query(query: str, db_name: str, state_catalog: dict) -> str:
    """Execute SQL query"""
    catalog = state_catalog
    
    if db_name not in catalog["databases"]:
        return json.dumps({"error": f"Database '{db_name}' not found"})
    
    db_info = catalog["databases"][db_name]
    
    if "error" in db_info:
        return json.dumps({"error": db_info["error"]})
    
    db_path = db_info["full_path"]
    
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
        return json.dumps({"error": str(e)})

@tool
def omdb_api(by: str = "title", t: str = None, plot: str = "full") -> str:
    """Query OMDb API"""
    if not OMDB_API_KEY:
        return json.dumps({"error": "OMDB_API_KEY missing"})
    
    params = {"apikey": OMDB_API_KEY, "plot": plot}
    
    if by == "title" and t:
        params["t"] = t
    else:
        return json.dumps({"error": "Title required"})
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})

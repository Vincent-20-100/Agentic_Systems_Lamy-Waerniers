# === TOOLS ===
@tool
def get_db_schema(db_path: str = "../data") -> str:
    """Récupère le schéma de toutes les bases SQLite du dossier."""
    result = {"databases": [], "error": None}
    
    try:
        db_files = [f for f in os.listdir(db_path) if f.endswith(('.db', '.sqlite', '.sqlite3'))]
    except FileNotFoundError:
        return json.dumps({"error": f"Dossier {db_path} introuvable"})
    
    if not db_files:
        return json.dumps({"error": "Aucune base SQLite trouvée"})
    
    for db_file in db_files:
        db_path_full = os.path.join(db_path, db_file)
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
def execute_sql_query(query: str, db_path: str = "../data/netflix.db") -> str:
    """Exécute une requête SQL sur netflix.db."""
    if not os.path.exists(db_path):
        return json.dumps({"error": f"Base {db_path} introuvable"})
    
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
        return json.dumps({"error": f"Erreur SQL: {str(e)}"})

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Recherche web via DuckDuckGo."""
    try:
        search = DuckDuckGoSearchResults(num_results=num_results)
        return search.run(query)
    except Exception as e:
        return json.dumps({"error": f"Erreur web search: {str(e)}"})

@tool
def omdb_api(by: str = "search", i: str = None, t: str = None, 
             s: str = None, y: str = None, plot: str = "short") -> str:
    """
    Interroge l'API OMDb pour infos films/séries.
    - by="id" + i="tt123" → recherche par IMDb ID
    - by="title" + t="Inception" → recherche par titre exact
    - by="search" + s="matrix" → recherche par mot-clé
    """
    if not OMDB_API_KEY:
        return json.dumps({"error": "Clé OMDB_API_KEY manquante"})
    
    params = {"apikey": OMDB_API_KEY, "plot": plot}
    
    if by == "id" and i:
        params["i"] = i
    elif by == "title" and t:
        params["t"] = t
    elif by == "search" and s:
        params["s"] = s
    else:
        return json.dumps({"error": "Paramètres manquants (i/t/s selon 'by')"})
    
    if y:
        params["y"] = y
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Erreur API OMDb: {str(e)}"})

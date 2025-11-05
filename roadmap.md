Agent SQL Basique avec LangGraph
Objectif MVP
Créer un chatbot qui :

Se connecte à une base de données SQLite locale (fichier .db dans le repo).
Traduire le langage naturel en requêtes SQL (ex: "Montre-moi tous les clients de Paris" → SELECT * FROM clients WHERE ville = 'Paris').
Exécute la requête et retourne les résultats sous forme de tableau.
Gère les erreurs (syntaxe SQL invalide, tables inexistantes).


## 1. Setup
### Dépendances
```bash
python -m venv .venv
source .venv/bin/activate
pip install langgraph langchain langchain-openai sqlite3 pandas
```

### Structure
sql_agent/ 
├── data/ 
│   └── sample.db/ 
├── main.py/ 
├── tools.py/ 
└── requirements.txt

## 2. Base de Données (SQLite)
Initialisation (data/init_db.py)
```python Copierimport sqlite3

conn = sqlite3.connect("data/sample.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY,
    nom TEXT NOT NULL,
    ville TEXT,
    age INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS commandes (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    produit TEXT,
    montant REAL,
    date TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(id)
)
""")

cursor.executemany(
    "INSERT INTO clients (nom, ville, age) VALUES (?, ?, ?)",
    [("Alice", "Paris", 30), ("Bob", "Lyon", 25), ("Charlie", "Paris", 35)]
)

cursor.executemany(
    "INSERT INTO commandes (client_id, produit, montant, date) VALUES (?, ?, ?, ?)",
    [(1, "Livre", 29.99, "2023-01-15"), (1, "Ordinateur", 999.99, "2023-02-20"), (2, "Téléphone", 699.99, "2023-03-10")]
)

conn.commit()
conn.close()
```

## 3. Outils (tools.py)
```python Copierimport sqlite3
from langchain.tools import tool

@tool
def execute_sql_query(query: str) -> str:
    try:
        conn = sqlite3.connect("data/sample.db")
        cursor = conn.cursor()
        cursor.execute(query)

        if query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            if not rows:
                return "Aucun résultat."

            column_names = [d[0] for d in cursor.description]
            markdown_table = "| " + " | ".join(column_names) + " |\n"
            markdown_table += "|" + "|".join(["---"] * len(column_names)) + "|\n"
            for row in rows:
                markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            return markdown_table
        else:
            conn.commit()
            return f"Requête exécutée. {cursor.rowcount} lignes affectées."

    except sqlite3.Error as e:
        return f"Erreur SQL: {str(e)}"
    finally:
        conn.close()
```

## 4. Agent LangGraph (main.py)
```python Copierfrom langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from tools import execute_sql_query

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], add_messages]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
tools = [execute_sql_query]
llm_with_tools = llm.bind_tools(tools)

def call_llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = execute_sql_query.invoke(tool_call["args"])
        outputs.append(ToolMessage(content=tool_result, name=tool_call["name"], tool_call_id=tool_call["id"]))
    return {"messages": outputs}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm_node)
workflow.add_node("tool", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tool", "end": END})
workflow.add_edge("tool", "agent")
graph = workflow.compile()

if __name__ == "__main__":
    system_prompt = """
    Tu es un assistant SQL pour une base avec les tables :
    - clients(id, nom, ville, age)
    - commandes(client_id, produit, montant, date)
    Utilise `execute_sql_query` pour répondre. Demande des précisions si nécessaire.
    """

    user_query = "Montre-moi tous les clients de Paris avec leurs commandes."
    response = graph.invoke({
        "messages": [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
    })
    print(response["messages"][-1].content)
```

## 5. Interface Streamlit (app.py)
```python Copierimport streamlit as st
from main import graph

st.title("Agent SQL Local")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question SQL en langage naturel..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = """
    Assistant SQL pour une base locale avec tables :
    - clients(id, nom, ville, age)
    - commandes(client_id, produit, montant, date)
    Utilise `execute_sql_query`.
    """
    response = graph.invoke({
        "messages": [
            {"type": "system", "content": system_prompt},
            {"type": "human", "content": prompt}
        ]
    })

    with st.chat_message("assistant"):
        st.markdown(response["messages"][-1].content)
    st.session_state.messages.append({"role": "assistant", "content": response["messages"][-1].content})
```

## 6. Exécution
```bash 
# Initialiser la base de données
python data/init_db.py

# Lancer l'agent en CLI
python main.py

# Lancer l'interface web
streamlit run app.py
```

## 7. Exemples de Requêtes
Requête en Langage NaturelRésultat Attendu"Montre-moi tous les clients de Paris."Tableau avec Alice et Charlie."Quels produits a achetés Alice ?"Livre, Ordinateur."Quel est le montant total des commandes ?"1729.97"Qui a dépensé le plus ?"Alice (1029.98)

## 8. Extensions Futures

Ajouter INSERT/UPDATE/DELETE (avec confirmation).
Support PostgreSQL/MySQL.
Visualisation de données (graphiques).
Historique des requêtes.
Mode expert (voir/modifier le SQL généré).


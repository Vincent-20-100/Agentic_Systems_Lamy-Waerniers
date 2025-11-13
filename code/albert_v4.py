"""
Albert V4 - Production-Ready SQL Agent with RAG
Author: Vincent Lamy
Date: 2025-11-10

Architecture: Multi-Agent System with:
- Safety Guardrails (Pre/Post)
- Dual Memory System (Short-term + Long-term with JSON persistence)
- Structured Output (Pydantic)
- Intent Classification
- SQL + Semantic RAG + Web Search
- Smart Routing with LangGraph

Based on: LangChain Advanced Solution Best Practices
"""

import os
import json
import sqlite3
import pathlib
import re
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Pydantic for structured data
from pydantic import BaseModel, Field

# Environment
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Paths
PROJECT_ROOT = pathlib.Path.cwd()
if PROJECT_ROOT.name == "code":
    PROJECT_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = str(PROJECT_ROOT / "code")
DB_FOLDER_PATH = str(PROJECT_ROOT / "data" / "databases")
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
MEMORY_PATH = str(PROJECT_ROOT / "data" / "memory")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# ============================================================================
# ENUMS AND STRUCTURED DATA MODELS
# ============================================================================

class QueryIntent(str, Enum):
    """Query intent types"""
    SQL_QUERY = "sql_query"
    SEMANTIC_SEARCH = "semantic_search"
    WEB_SEARCH = "web_search"
    CLARIFICATION_NEEDED = "clarification_needed"
    GENERAL_CHAT = "general_chat"

class DataSource(str, Enum):
    """Available data sources"""
    SQL_DATABASE = "sql_database"
    VECTOR_STORE = "vector_store"
    OMDB_API = "omdb_api"
    WEB_SEARCH = "web_search"

class QueryContext(BaseModel):
    """Rich context for each query"""
    query: str = Field(..., description="User's query")
    intent: QueryIntent = Field(..., description="Detected intent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters")
    semantic_keywords: List[str] = Field(default_factory=list)

class SearchResult(BaseModel):
    """Structured search result"""
    source: DataSource
    data: Dict[str, Any]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class AgentState(BaseModel):
    """Complete agent state with Pydantic validation"""
    # Session management
    session_id: str
    thread_id: str

    # Query context
    original_query: str
    query_context: Optional[QueryContext] = None

    # Conversation history (managed separately)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # User profile (loaded from JSON)
    user_profile: Dict[str, Any] = Field(default_factory=dict)

    # Search results
    sql_results: List[SearchResult] = Field(default_factory=list)
    semantic_results: List[SearchResult] = Field(default_factory=list)
    web_results: List[SearchResult] = Field(default_factory=list)

    # Clarification handling
    needs_clarification: bool = False
    clarification_context: Optional[str] = None
    clarification_history: List[Dict[str, str]] = Field(default_factory=list)

    # Final output
    final_answer: Optional[str] = None
    sources_used: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0

    # Metadata
    current_step: str = "initialized"
    errors: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

# Patterns for pre-guard validation
DISALLOWED_PATTERNS = re.compile(
    r"(?i)(credit card|ssn|social security|password|api[_\s]?key|"
    r"violent threat|malicious code|sql injection)"
)

INJECTION_PATTERNS = re.compile(
    r"(?i)(ignore previous|disregard instructions|system prompt|"
    r"you are now|forget everything|override)"
)

class SafetyGuard:
    """Production-ready safety system"""

    @staticmethod
    def pre_guard(user_input: str) -> tuple[bool, Optional[str]]:
        """
        Validate input before processing.
        Returns: (is_safe, error_message)
        """
        # Check for disallowed content
        if DISALLOWED_PATTERNS.search(user_input):
            return False, "Input contains prohibited content"

        # Check for prompt injection
        if INJECTION_PATTERNS.search(user_input):
            return False, "Input appears to be a prompt injection attempt"

        # Check length
        if len(user_input) > 2000:
            return False, "Input exceeds maximum length (2000 characters)"

        return True, None

    @staticmethod
    def post_guard(output: str, max_length: int = 2000) -> str:
        """Sanitize output before returning to user"""
        # Truncate if too long
        if len(output) > max_length:
            output = output[:max_length] + "... [truncated]"

        # Remove any leaked system prompts
        output = re.sub(r"(?i)(system prompt:|internal note:).*", "", output)

        return output

    @staticmethod
    def validate_sql_query(query: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for safety"""
        dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
        query_upper = query.upper()

        for keyword in dangerous:
            if keyword in query_upper:
                return False, f"SQL query contains dangerous keyword: {keyword}"

        return True, None

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class ConversationMemory:
    """Manages short-term conversation history with JSON persistence"""

    def __init__(self, storage_path: str = MEMORY_PATH):
        self.storage_path = pathlib.Path(storage_path) / "conversations"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.in_memory_store = {}  # For InMemoryChatMessageHistory

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create history for session"""
        if session_id not in self.in_memory_store:
            self.in_memory_store[session_id] = InMemoryChatMessageHistory()
            # Load from disk if exists
            self._load_from_disk(session_id)
        return self.in_memory_store[session_id]

    def add_turn(self, session_id: str, user_msg: str, ai_msg: str):
        """Add conversation turn and save to disk"""
        history = self.get_history(session_id)
        history.add_user_message(user_msg)
        history.add_ai_message(ai_msg)
        # Save to disk
        self._save_to_disk(session_id)

    def get_recent_context(self, session_id: str, n: int = 5) -> List[Dict[str, str]]:
        """Get last N conversation turns"""
        history = self.get_history(session_id)
        messages = history.messages[-n*2:] if len(history.messages) > n*2 else history.messages
        return [{"role": m.type, "content": m.content} for m in messages]

    def _save_to_disk(self, session_id: str):
        """Save conversation history to JSON"""
        file_path = self.storage_path / f"{session_id}.json"
        history = self.get_history(session_id)

        conversations = [
            {
                "role": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            for msg in history.messages
        ]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session_id,
                "last_updated": datetime.now().isoformat(),
                "messages": conversations
            }, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self, session_id: str):
        """Load conversation history from JSON"""
        file_path = self.storage_path / f"{session_id}.json"
        if not file_path.exists():
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            history = self.in_memory_store[session_id]
            for msg in data.get("messages", []):
                if msg["role"] == "human":
                    history.add_user_message(msg["content"])
                elif msg["role"] == "ai":
                    history.add_ai_message(msg["content"])
        except Exception as e:
            print(f"Error loading conversation history: {e}")

    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        if session_id in self.in_memory_store:
            del self.in_memory_store[session_id]

        file_path = self.storage_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()

class UserProfileMemory:
    """Manages long-term user preferences and profiles with JSON persistence"""

    def __init__(self, storage_path: str = MEMORY_PATH):
        self.storage_path = pathlib.Path(storage_path) / "user_profiles"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile from disk"""
        profile_file = self.storage_path / f"{user_id}.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user profile: {e}")

        # Default profile
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {
                "favorite_genres": [],
                "favorite_platforms": [],
                "preferred_content_type": None  # "Movie" or "TV Show"
            },
            "search_history": [],
            "interaction_count": 0,
            "last_active": datetime.now().isoformat()
        }

    def save_profile(self, user_id: str, profile: Dict[str, Any]):
        """Save user profile to disk"""
        profile_file = self.storage_path / f"{user_id}.json"
        profile["last_active"] = datetime.now().isoformat()

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def update_preferences(self, user_id: str, query: str, results: List[SearchResult]):
        """Update profile based on interaction"""
        profile = self.load_profile(user_id)
        profile["interaction_count"] += 1

        # Add to search history (keep last 50)
        profile["search_history"].append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(results)
        })
        profile["search_history"] = profile["search_history"][-50:]

        # Extract preferences from results
        for result in results:
            if result.source == DataSource.SQL_DATABASE or result.source == DataSource.VECTOR_STORE:
                data = result.data.get("results", [])
                if isinstance(data, list):
                    for item in data:
                        # Extract genres
                        if "genres" in item or "listed_in" in item:
                            genres = item.get("genres") or item.get("listed_in", "")
                            if genres:
                                genre_list = [g.strip() for g in genres.split(",")]
                                for genre in genre_list:
                                    if genre and genre not in profile["preferences"]["favorite_genres"]:
                                        profile["preferences"]["favorite_genres"].append(genre)

                        # Extract platforms
                        if "platform" in item:
                            platform = item["platform"]
                            if platform and platform not in profile["preferences"]["favorite_platforms"]:
                                profile["preferences"]["favorite_platforms"].append(platform)

        # Limit lists
        profile["preferences"]["favorite_genres"] = profile["preferences"]["favorite_genres"][-10:]
        profile["preferences"]["favorite_platforms"] = profile["preferences"]["favorite_platforms"][-5:]

        self.save_profile(user_id, profile)

    def get_personalization_context(self, user_id: str) -> str:
        """Get formatted context for personalization"""
        profile = self.load_profile(user_id)

        if profile["interaction_count"] == 0:
            return ""

        context_parts = []
        prefs = profile["preferences"]

        if prefs["favorite_genres"]:
            context_parts.append(f"User frequently searches for: {', '.join(prefs['favorite_genres'][:5])}")
        if prefs["favorite_platforms"]:
            context_parts.append(f"Preferred platforms: {', '.join(prefs['favorite_platforms'])}")
        if prefs["preferred_content_type"]:
            context_parts.append(f"Prefers: {prefs['preferred_content_type']}")

        return "\n".join(context_parts) if context_parts else ""

# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def build_db_catalog(folder_path: str) -> Dict[str, Any]:
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

def format_catalog_for_llm(catalog: Dict[str, Any]) -> str:
    """Format catalog for LLM"""
    if catalog.get("error"):
        return f"ERROR: {catalog['error']}"

    formatted = " Available Databases:\n\n"

    for db_name, db_info in catalog["databases"].items():
        if "error" in db_info:
            formatted += f" {db_name}: {db_info['error']}\n"
            continue

        formatted += f"**Database: {db_name}** (file: {db_info['file_name']})\n"

        for table_name, table_info in db_info["tables"].items():
            cols = ", ".join([f"{col['name']} ({col['type']})"
                            for col in table_info["columns"][:5]])  # Limit for brevity
            formatted += f"  • Table `{table_name}`: {cols}\n"

        formatted += "\n"

    return formatted

# ============================================================================
# INTENT CLASSIFICATION AGENT
# ============================================================================

class IntentClassification(BaseModel):
    """Structured intent classification"""
    intent: QueryIntent
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    requires_clarification: bool
    clarification_question: Optional[str] = None
    extracted_filters: Optional[Dict[str, Any]] = None

class IntentClassifierAgent:
    """Classifies user intent with structured output"""

    def __init__(self, llm):
        self.llm = llm.with_structured_output(IntentClassification)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for a movie/TV show database assistant.

Available databases:
- Netflix (~8,800 titles)
- Amazon Prime (~9,600 titles)
- Disney+ (~1,400 titles)

Classify user queries into these intents:

1. SQL_QUERY: Structured queries (year, platform, genre filters)
   Examples: "movies from 2020", "Netflix shows", "PG-rated films"

2. SEMANTIC_SEARCH: Mood/style/similarity queries
   Examples: "dark thrillers", "movies like Inception", "heartwarming films"

3. WEB_SEARCH: Current events, recent releases
   Examples: "latest Netflix releases", "trending movies today"

4. CLARIFICATION_NEEDED: Ambiguous queries
   Examples: "a movie", "good show", "something to watch"

5. GENERAL_CHAT: Greetings, off-topic
   Examples: "hello", "how are you", "thank you"

Extract structured filters when possible:
- platform: netflix, amazon_prime, disney_plus
- release_year: integer
- type: Movie, TV Show
- rating: PG, PG-13, R, TV-MA, etc.
- genre: Action, Drama, Comedy, etc.
"""),
            ("user", "{query}")
        ])
        self.chain = self.prompt | self.llm

    def classify(self, query: str) -> IntentClassification:
        """Classify query and return structured intent"""
        return self.chain.invoke({"query": query})

# ============================================================================
# SQL AGENT
# ============================================================================

class SQLQueryResult(BaseModel):
    """Structured SQL query result"""
    can_answer_with_sql: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    database: Optional[str] = None
    query: Optional[str] = None
    expected_columns: List[str] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str

class SQLAgent:
    """SQL query generation and execution with safety"""

    def __init__(self, llm, db_catalog: Dict[str, Any]):
        self.llm = llm.with_structured_output(SQLQueryResult)
        self.db_catalog = db_catalog
        self.guard = SafetyGuard()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator for movie/TV databases.

{db_catalog}

Generate SAFE, READ-ONLY queries:
- Only SELECT statements
- Always use LIMIT (max 50)
- Use LIKE with wildcards for text search: WHERE title LIKE '%keyword%'
- Apply filters from user context
- Use proper column names from schema

Return structured query plan."""),
            ("user", "Query: {query}\nUser Context: {context}")
        ])

        self.chain = self.prompt | self.llm

    def generate_query(self, query_context: QueryContext, user_context: str = "") -> SQLQueryResult:
        """Generate SQL query with structured output"""
        return self.chain.invoke({
            "query": query_context.query,
            "context": user_context,
            "db_catalog": format_catalog_for_llm(self.db_catalog)
        })

    def execute_query(self, sql_result: SQLQueryResult) -> SearchResult:
        """Execute SQL query with safety checks"""
        # Validate query
        is_safe, error = self.guard.validate_sql_query(sql_result.query)
        if not is_safe:
            return SearchResult(
                source=DataSource.SQL_DATABASE,
                data={"error": error, "results": []},
                confidence=0.0
            )

        # Execute query
        try:
            db_path = self.db_catalog["databases"][sql_result.database]["full_path"]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_result.query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
            conn.close()

            return SearchResult(
                source=DataSource.SQL_DATABASE,
                data={"results": results, "count": len(results)},
                confidence=sql_result.confidence,
                metadata={"database": sql_result.database, "filters": sql_result.filters_applied}
            )
        except Exception as e:
            return SearchResult(
                source=DataSource.SQL_DATABASE,
                data={"error": str(e), "results": []},
                confidence=0.0
            )

# ============================================================================
# SEMANTIC RAG AGENT
# ============================================================================

class SemanticRAGAgent:
    """Semantic search using vector embeddings"""

    def __init__(self, llm, vectorstore_path: str = CHROMA_PATH):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

        # Load vector store (create if doesn't exist)
        try:
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings,
                collection_name="movies_shows"
            )
            print(f" Loaded vector store from {vectorstore_path}")
        except Exception as e:
            print(f" Vector store not found: {e}")
            print("   Run create_embeddings.ipynb first!")
            self.vectorstore = None

    def search(self, query_context: QueryContext, k: int = 10) -> SearchResult:
        """Perform semantic search"""
        if not self.vectorstore:
            return SearchResult(
                source=DataSource.VECTOR_STORE,
                data={"error": "Vector store not initialized", "results": []},
                confidence=0.0
            )

        try:
            # Build metadata filters
            filters = {}
            if query_context.filters:
                if query_context.filters.get("platform"):
                    filters["platform"] = query_context.filters["platform"]
                if query_context.filters.get("release_year"):
                    filters["release_year"] = query_context.filters["release_year"]
                if query_context.filters.get("type"):
                    filters["type"] = query_context.filters["type"]

            # Execute search
            if filters:
                results = self.vectorstore.similarity_search(
                    query_context.query,
                    k=k,
                    filter=filters
                )
            else:
                results = self.vectorstore.similarity_search(
                    query_context.query,
                    k=k
                )

            formatted_results = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "platform": doc.metadata.get("platform", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown"),
                    "year": doc.metadata.get("release_year", 0),
                    "genres": doc.metadata.get("genres", ""),
                    "description": doc.page_content[:200] if hasattr(doc, 'page_content') else ""
                }
                for doc in results
            ]

            return SearchResult(
                source=DataSource.VECTOR_STORE,
                data={"results": formatted_results, "count": len(formatted_results)},
                confidence=0.8,
                metadata={"filters": filters}
            )
        except Exception as e:
            return SearchResult(
                source=DataSource.VECTOR_STORE,
                data={"error": str(e), "results": []},
                confidence=0.0
            )

# ============================================================================
# SYNTHESIZER AGENT
# ============================================================================

class FinalAnswer(BaseModel):
    """Structured final answer"""
    answer: str = Field(..., description="Natural language answer")
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str]
    recommendations: Optional[List[Dict[str, str]]] = None
    missing_info: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)

class SynthesizerAgent:
    """Synthesizes results from multiple sources into coherent answer"""

    def __init__(self, llm):
        self.llm = llm.with_structured_output(FinalAnswer)
        self.guard = SafetyGuard()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Albert, a friendly movie/TV show assistant.

Synthesize information from multiple sources into a natural, conversational answer.

Guidelines:
- Be concise but informative (2-4 sentences for simple queries)
- Cite sources naturally (e.g., "According to our database...")
- If data is incomplete, acknowledge it professionally
- Suggest related queries when appropriate
- Use friendly, conversational tone
- Provide specific recommendations when data is available

{user_profile}"""),
            ("user", """Original query: {query}

Available data:

SQL Results: {sql_results}

Semantic Results: {semantic_results}

Web Results: {web_results}

Create a comprehensive, natural answer.""")
        ])

        self.chain = self.prompt | self.llm

    def synthesize(
        self,
        query: str,
        sql_results: List[SearchResult],
        semantic_results: List[SearchResult],
        web_results: List[SearchResult],
        user_profile: str = ""
    ) -> FinalAnswer:
        """Synthesize all results into final answer"""
        answer = self.chain.invoke({
            "query": query,
            "sql_results": json.dumps([r.data for r in sql_results], default=str),
            "semantic_results": json.dumps([r.data for r in semantic_results], default=str),
            "web_results": json.dumps([r.data for r in web_results], default=str),
            "user_profile": user_profile
        })

        # Apply post-guard
        answer.answer = self.guard.post_guard(answer.answer)

        return answer

# ============================================================================
# MASTER ORCHESTRATOR (Albert V4)
# ============================================================================

class AlbertV4:
    """Main orchestrator for Albert V4"""

    def __init__(self, db_catalog: Dict[str, Any], vectorstore_path: str = CHROMA_PATH):
        self.llm = llm
        self.guard = SafetyGuard()
        self.db_catalog = db_catalog

        # Initialize agents
        self.intent_classifier = IntentClassifierAgent(llm)
        self.sql_agent = SQLAgent(llm, db_catalog)
        self.semantic_agent = SemanticRAGAgent(llm, vectorstore_path)
        self.synthesizer = SynthesizerAgent(llm)

        # Initialize memory
        self.conv_memory = ConversationMemory()
        self.user_memory = UserProfileMemory()

        # Build graph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("safety_check", self._safety_check_node)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("handle_clarification", self._handle_clarification_node)
        workflow.add_node("sql_search", self._sql_search_node)
        workflow.add_node("semantic_search", self._semantic_search_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("post_guard", self._post_guard_node)

        # Define flow
        workflow.add_edge(START, "safety_check")

        workflow.add_conditional_edges(
            "safety_check",
            self._route_after_safety,
            {"safe": "classify_intent", "unsafe": "post_guard"}
        )

        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "sql": "sql_search",
                "semantic": "semantic_search",
                "clarify": "handle_clarification",
                "chat": "synthesize"
            }
        )

        workflow.add_edge("sql_search", "synthesize")
        workflow.add_edge("semantic_search", "synthesize")
        workflow.add_edge("handle_clarification", "post_guard")
        workflow.add_edge("synthesize", "post_guard")
        workflow.add_edge("post_guard", END)

        # Compile with checkpointing
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    def _safety_check_node(self, state: AgentState) -> AgentState:
        """Pre-guard validation"""
        is_safe, error = self.guard.pre_guard(state.original_query)

        if not is_safe:
            state.final_answer = f"I'm sorry, but I cannot process this request. {error}"
            state.current_step = "safety_blocked"
        else:
            state.current_step = "safety_passed"

        return state

    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify user intent"""
        # Get user profile for personalization
        user_profile = self.user_memory.get_personalization_context(state.session_id)

        # Classify intent
        intent_result = self.intent_classifier.classify(state.original_query)

        state.query_context = QueryContext(
            query=state.original_query,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            filters=intent_result.extracted_filters,
            semantic_keywords=[]
        )

        state.needs_clarification = intent_result.requires_clarification
        state.clarification_context = intent_result.clarification_question
        state.current_step = f"intent_{intent_result.intent.value}"

        return state

    def _handle_clarification_node(self, state: AgentState) -> AgentState:
        """Handle clarification request"""
        if state.clarification_context:
            state.final_answer = state.clarification_context
            state.current_step = "awaiting_clarification"
        else:
            state.final_answer = "I need more information. Could you please be more specific?"
            state.current_step = "clarification_generic"

        return state

    def _sql_search_node(self, state: AgentState) -> AgentState:
        """Execute SQL search"""
        user_context = self.user_memory.get_personalization_context(state.session_id)

        # Generate query
        sql_plan = self.sql_agent.generate_query(state.query_context, user_context)

        # Execute if valid
        if sql_plan.can_answer_with_sql and sql_plan.query:
            result = self.sql_agent.execute_query(sql_plan)
            state.sql_results.append(result)
            if sql_plan.database:
                state.sources_used.append(f"SQL: {sql_plan.database}")

        state.current_step = "sql_complete"
        return state

    def _semantic_search_node(self, state: AgentState) -> AgentState:
        """Execute semantic search"""
        result = self.semantic_agent.search(state.query_context)
        state.semantic_results.append(result)
        state.sources_used.append("Semantic Search (Vector DB)")
        state.current_step = "semantic_complete"
        return state

    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer"""
        user_profile = self.user_memory.get_personalization_context(state.session_id)

        final = self.synthesizer.synthesize(
            state.original_query,
            state.sql_results,
            state.semantic_results,
            state.web_results,
            user_profile
        )

        state.final_answer = final.answer
        state.confidence_score = final.confidence
        state.sources_used.extend(final.sources)
        state.current_step = "synthesized"

        return state

    def _post_guard_node(self, state: AgentState) -> AgentState:
        """Final output validation"""
        if state.final_answer:
            state.final_answer = self.guard.post_guard(state.final_answer)

        state.current_step = "complete"
        return state

    def _route_after_safety(self, state: AgentState) -> str:
        """Route based on safety check"""
        return "safe" if state.current_step == "safety_passed" else "unsafe"

    def _route_after_intent(self, state: AgentState) -> str:
        """Route based on intent"""
        if state.needs_clarification:
            return "clarify"

        if not state.query_context:
            return "chat"

        intent = state.query_context.intent
        if intent == QueryIntent.SQL_QUERY:
            return "sql"
        elif intent == QueryIntent.SEMANTIC_SEARCH:
            return "semantic"
        elif intent == QueryIntent.WEB_SEARCH:
            return "semantic"  # Fallback to semantic for now
        else:
            return "chat"

    def query(self, user_query: str, session_id: str = "default") -> tuple[str, Dict[str, Any]]:
        """
        Main entry point for queries.
        Returns: (answer, metadata)
        """
        # Load user profile
        user_profile = self.user_memory.load_profile(session_id)

        # Get recent conversation context
        recent_context = self.conv_memory.get_recent_context(session_id, n=3)

        # Create initial state
        initial_state = AgentState(
            session_id=session_id,
            thread_id=session_id,
            original_query=user_query,
            user_profile=user_profile,
            chat_history=recent_context
        )

        # Execute workflow
        config = {"configurable": {"thread_id": session_id}}
        result = self.app.invoke(initial_state, config)

        # Update conversation memory
        if result.final_answer:
            self.conv_memory.add_turn(session_id, user_query, result.final_answer)

        # Update user profile
        all_results = result.sql_results + result.semantic_results + result.web_results
        self.user_memory.update_preferences(session_id, user_query, all_results)

        # Return answer and metadata
        metadata = {
            "confidence": result.confidence_score,
            "sources": result.sources_used,
            "intent": result.query_context.intent.value if result.query_context else "unknown",
            "clarification_needed": result.needs_clarification
        }

        return result.final_answer or "I couldn't generate a response.", metadata

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(" Initializing Albert V4...")

    # Build database catalog
    print(" Loading database catalog...")
    db_catalog = build_db_catalog(DB_FOLDER_PATH)

    if db_catalog.get("error"):
        print(f" Error: {db_catalog['error']}")
        print(f"   Please check DB_FOLDER_PATH: {DB_FOLDER_PATH}")
        exit(1)

    print(f" Loaded {len(db_catalog['databases'])} databases")

    # Initialize Albert V4
    albert = AlbertV4(db_catalog)

    print("\n" + "="*60)
    print(" Albert V4 Ready!")
    print("="*60)
    print("\nMemory Features:")
    print("- Short-term: Conversation history (saved to JSON)")
    print("- Long-term: User profiles & preferences (saved to JSON)")
    print(f"- Storage: {MEMORY_PATH}")
    print("\nTry queries like:")
    print('  - "movies from 2020"')
    print('  - "dark psychological thrillers"')
    print('  - "show me something similar to Inception"')
    print("\nType 'quit' to exit\n")

    session_id = "default_session"

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break

            if not user_input:
                continue

            # Query Albert
            answer, metadata = albert.query(user_input, session_id)

            # Display response
            print(f"\nAlbert: {answer}")
            print(f"\n Metadata:")
            print(f"   Confidence: {metadata['confidence']:.2f}")
            print(f"   Intent: {metadata['intent']}")
            print(f"   Sources: {', '.join(metadata['sources']) if metadata['sources'] else 'None'}")
            print()

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print()

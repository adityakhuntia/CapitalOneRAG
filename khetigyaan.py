from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import warnings
import psycopg2
from transformers import pipeline
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(title="AgriBot API", description="API for an AI-powered agricultural chatbot")

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gemini_key)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_key,
    temperature=0,
)

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Initialize intent classifier
labels = ["weather advisory", "crop health", "price", "government support", 
          "fertilizer and pesticides", "irrigation", "miscellaneous"]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_intent(query: str) -> str:
    pred = classifier(query, labels, multi_label=False)
    return pred["labels"][0]

# SimplePostgresChatHistory class
class SimplePostgresChatHistory:
    def __init__(self, connection_string: str, session_id: str, table_name: str = "chat_messages"):
        self.connection_string = connection_string
        self.session_id = session_id
        self.table_name = table_name
        self.messages = []
        self.use_postgres = False
        
        if connection_string:
            self._init_postgres()
        else:
            print("No DATABASE_URL provided. Using in-memory storage.")
    
    def _init_postgres(self):
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                message_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            create_index_query = f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session_id 
            ON {self.table_name}(session_id)
            """
            cursor.execute(create_index_query)
            conn.commit()
            cursor.close()
            conn.close()
            self.use_postgres = True
            print(f"✅ PostgreSQL chat history initialized. Table: {self.table_name}")
        except Exception as e:
            print(f"❌ PostgreSQL initialization failed: {e}")
            self.use_postgres = False
    
    def add_user_message(self, message: str):
        if self.use_postgres:
            self._add_to_postgres("user", message)
        else:
            self.messages.append({
                "type": "user", 
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
    
    def add_ai_message(self, message: str):
        if self.use_postgres:
            self._add_to_postgres("assistant", message)
        else:
            self.messages.append({
                "type": "assistant", 
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
    
    def _add_to_postgres(self, message_type: str, content: str):
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            insert_query = f"""
            INSERT INTO {self.table_name} (session_id, message_type, content) 
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (self.session_id, message_type, content))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
            self.messages.append({
                "type": message_type, 
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self.use_postgres:
            return self._get_from_postgres(limit)
        else:
            return self.messages[-limit:] if self.messages else []
    
    def _get_from_postgres(self, limit: int) -> List[Dict[str, Any]]:
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            select_query = f"""
            SELECT message_type, content, created_at 
            FROM {self.table_name} 
            WHERE session_id = %s 
            ORDER BY created_at ASC
            LIMIT %s
            """
            cursor.execute(select_query, (self.session_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            messages = []
            for row in rows:
                messages.append({
                    "type": row[0],
                    "content": row[1],
                    "timestamp": row[2].isoformat() if row[2] else None
                })
            return messages
        except Exception as e:
            print(f"Error getting from PostgreSQL: {e}")
            return self.messages[-limit:] if self.messages else []
    
    def format_conversation_context(self, max_messages: int = 6) -> str:
        messages = self.get_recent_messages(max_messages)
        if not messages:
            return "Previous Conversation: This is a new conversation."
        context_lines = ["Recent Conversation Context:"]
        for msg in messages:
            role = "User" if msg["type"] == "user" else "Assistant"
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            context_lines.append(f"{role}: {content}")
        return "\n".join(context_lines)

    def clear(self):
        if self.use_postgres:
            try:
                conn = psycopg2.connect(self.connection_string)
                cursor = conn.cursor()
                delete_query = f"DELETE FROM {self.table_name} WHERE session_id = %s"
                cursor.execute(delete_query, (self.session_id,))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"Error clearing PostgreSQL history: {e}")
        self.messages = []

# Initialize chat history
pg_url = os.getenv("DATABASE_URL")
chat_history = SimplePostgresChatHistory(pg_url, "farmer01")

# Helper functions
def get_season():
    now = datetime.now()
    month = now.month
    if 6 <= month <= 10:
        season = "Kharif"
    elif 11 <= month or month <= 4:
        season = "Rabi"
    else:
        season = "Zaid/Transition"
    return f"Today is {now.strftime('%d-%m-%Y %H:%M:%S')} and it is the {season} season."

# RAG function
def rag_function_working(query: str, language: str = "English", region: str = "Haryana") -> str:
    try:
        conversation_context = chat_history.format_conversation_context(max_messages=6)
        predicted_intent = classify_intent(query)
        print(f"Detected intent: {predicted_intent}")
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"intent": predicted_intent}})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs[:3]])
        prompt_template = """You are AI KhetiGyaan, an agricultural advisor for India.

User Details:
- Language: {language}
- Region: {region}
- Season/Date: {date}
- Available Tools: {tools}

{chat_history}

Knowledge Context:
{context}

Current Question: {query}

Instructions:
- Provide practical, region-specific agricultural advice
- Use simple language appropriate for farmers
- Reference previous conversation when relevant
- Include actionable steps and safety warnings
- Cite sources when available

Response:"""
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            chat_history=conversation_context,
            language=language,
            region=region,
            date=get_season(),
            tools="RAG + Search + Wikipedia"
        )
        response = llm.invoke(formatted_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        return result
    except Exception as e:
        error_msg = f"RAG Error: {str(e)}"
        print(error_msg)
        return f"I apologize, but I encountered an error processing your question: {error_msg}"

# Initialize tools
tools = [
    Tool(
        name="Agriculture_RAG",
        func=lambda query: rag_function_working(query),  # Defaults will be used if not overridden
        description="Get agriculture advice from knowledge base with PostgreSQL chat history."
    ),
    Tool(
        name="Web_Search",
        func=TavilySearchResults(search_depth="advanced").run,
        description="Search the web for current agricultural information."
    ),
    Tool(
        name="Wikipedia_Info",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Get scientific agricultural information from Wikipedia."
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# WorkingAgricultureBot class (added for RAG-based queries)
class WorkingAgricultureBot:
    def __init__(self, chat_history):
        self.chat_history = chat_history

    def ask(self, query: str, language: str = "English", region: str = "Haryana") -> str:
        return rag_function_working(query, language, region)

# Hybrid bot-agent class
class HybridAgricultureBot:
    def __init__(self, chat_history):
        self.chat_history = chat_history
        self.bot = WorkingAgricultureBot(chat_history)
        self.agent = agent

    def ask(self, query: str, language: str = "English", region: str = "Haryana") -> str:
        """Route query to bot or agent based on intent."""
        try:
            intent = classify_intent(query)
            print(f"Routing query with intent: {intent}")
            
            # Route to bot for context-sensitive or history-dependent queries
            if intent in ["price", "crop health", "government support", "fertilizer and pesticides", "irrigation"] or "what about" in query.lower():
                response = self.bot.ask(query, language=language, region=region)
            else:
                # Route to agent for search or general knowledge queries, incorporating language and region
                agent_input = f"Provide answer in {language} for region {region}: {query}"
                response = self.agent.invoke({"input": agent_input})['output']
            
            # Save to chat history
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(response)
            
            return response
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error processing your question: {error_msg}"

    def show_history(self, limit: int = 5):
        """Show recent conversation history."""
        messages = self.chat_history.get_recent_messages(limit)
        print(f"\n=== Last {len(messages)} Messages ===")
        for i, msg in enumerate(messages, 1):
            role = "🧑‍🌾 User" if msg["type"] == "user" else "🤖 Bot"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"{i}. {role}: {content}")
        print("=" * 40)

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history.clear()
        print("✅ Conversation history cleared.")

# Initialize hybrid bot
hybrid_bot = HybridAgricultureBot(chat_history)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: str = "farmer01"
    language: str = "English"
    region: str = "Haryana"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    session_id: str

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        global chat_history
        if request.session_id != chat_history.session_id:
            chat_history = SimplePostgresChatHistory(pg_url, request.session_id)
        
        start_time = time.time()
        response = hybrid_bot.ask(request.query, language=request.language, region=request.region)
        end_time = time.time()
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, limit: int = 5):
    try:
        global chat_history
        if session_id != chat_history.session_id:
            chat_history = SimplePostgresChatHistory(pg_url, session_id)
        
        history = chat_history.get_recent_messages(limit)
        return HistoryResponse(
            history=history,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    try:
        global chat_history
        if session_id != chat_history.session_id:
            chat_history = SimplePostgresChatHistory(pg_url, session_id)
        
        chat_history.clear()
        return {"message": "History cleared successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
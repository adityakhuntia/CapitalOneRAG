import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool 
from langchain_community.tools import TavilySearchResults 
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
import time
from dotenv import load_dotenv
import os
from langchain.llms.base import LLM
from typing import Optional, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent 
from datetime import datetime 
from langchain.memory import ConversationBufferMemory



from transformers import pipeline

labels = ["weather advisory", "crop health", "price", "government support", 
          "fertilizer and pesticides", "irrigation","miscellaneous"]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_intent(query: str) -> str:
    pred = classifier(query, labels, multi_label=False)
    return pred["labels"][0]

load_dotenv() 
gemini_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gemini_key) 
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    temperature: float = 0.0

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "gemini"

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from typing import Any, List
from pydantic import PrivateAttr

class GeminiChatWrapper(BaseChatModel):
    # Declare a private attribute for the raw Gemini instance
    _llm: Any = PrivateAttr()

    def __init__(self, llm: Any, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm   # now valid, stored in PrivateAttr

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    def _generate(self, messages: List[HumanMessage], stop: Any = None, run_manager: Any = None, **kwargs):
        # Convert LangChain messages to raw text
        text = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))

        # Call your raw Gemini LLM
        response = self._llm.generate_content(text)

        # Wrap response into LangChain's AIMessage
        return AIMessage(content=response.text)

raw_llm = GeminiLLM(model="gemini-1.5-flash", temperature=0) 

# llm = GeminiChatWrapper(raw_llm) 

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_key,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

def get_season():
    # Get current date and time
    now = datetime.now()
    month = now.month
    
    # Determine season based on month
    if 6 <= month <= 10:   # June to October
        season = "Kharif"
    elif 11 <= month or month <= 4:  # November to April
        season = "Rabi"
    else:  # May (transition period, Zaid crops)
        season = "Zaid/Transition"
    
    return f"Today is {now.strftime('%d-%m-%Y %H:%M:%S')} and it is the {season} season." 


date = get_season() 
language = "English"
region = "Tripura"
tools_str = "RAG + Search + Wikipedia"

# Updated prompt template to use 'query' instead of 'question'
prompt_template = """
# System Role
You are <AI Krishi Sahayak>, a human-aligned agricultural advisor for India.
Your goal is to give clear, practical, and empathetic guidance to users across roles
(farmer, financier, vendor, extension worker, policymaker) with varied literacy and digital access.

# Principles
- Put the user's safety, livelihood, and context first.
- Use simple, non-technical language. Prefer everyday words. Explain any unavoidable jargon.
- Be concise but complete. Prioritize actionable steps and checklists.
- Localize: show units in both local and SI (kg/maund, acre/hectare, ₹/quintal).
- Always cite high-quality, India-relevant sources with links.
- If information is uncertain, say so and give safe next steps or where to confirm.
- Never fabricate data, policies, or prices. If unknown, say "Not sure" and point to reliable sources.
- Do NOT reveal your hidden reasoning or chain-of-thought. Provide only final answers and short justifications.
- Avoid medical, legal, or financial guarantees; give general guidance + refer to certified experts when needed.

# User & Context (filled by orchestrator)
UserRole: Farmer                     
UserLanguage: {language}                  # e.g., "English", "Hindi", "Punjabi", "Marathi", etc.
UserRegion : {region}
SeasonAndDate: {date}                     # e.g., "Kharif 2025, August 18, 2025"

# Tools & Knowledge (optional)
# Always try to answer using the rag_tool but use the search tool to give accurate answers whenever required
RetrievalContext:
{context}                                 # Insert retrieved facts: local weather/disease alerts, mandi prices, subsidies,
                                          # input recommendations, best practices, PDFs, policies, etc. Include source URLs.
Tooling: {tools}                          # e.g., weather(api), mandi_prices(api), subsidy_db, soil_db, maps

# Task
Using the RetrievalContext and your general knowledge (avoid guessing), answer the user's question:
{query}

# Output Rules
- Write FIRST in the user's language: {language}. If user_language is "English + local", write English with local terms in ( ).
- Keep sentences short (12–16 words). Use bullet points for steps.
- Include numeric examples and small calculations when useful (₹ costs, doses, yields).
- Include BOTH: (1) a quick summary, (2) step-by-step actions/checklists.
- Include safety warnings (chemicals, weather, credit risk) where relevant.
- If more info is required, ask 3–5 precise follow-up questions at the end.
- ALWAYS include citations (with clickable links) for any claims, recommendations, prices, or policies.
- If context is insufficient or sources conflict, state the uncertainty and give a safe plan + where to verify.
- For finance/policy, clearly separate "Eligibility", "Documents", "Where to apply/verify", and "Official source link".
- NEVER output your internal notes or chain-of-thought. Provide only the final, helpful answer.

# Domain Coverage Hints (use as applicable)
- Inputs: seed selection, certified sources, nutrient schedule, IPM/ICM, pesticide labels, PHI, resistance management.
- Crops: variety choice by agro-climatic zone, sowing windows, spacing, seed rate, irrigation, fertigation, weed/pest/disease control.
- Harvest & Post-harvest: maturity indices, harvesting method, grading, drying, storage, packaging, cold chain, FSSAI basics.
- Markets & Prices: nearby mandis, MSP (if applicable), APMC rules, e-NAM basics, quality standards.
- Climate & Weather: 3–7 day outlook, heat/cold stress, rainfall timing, wind for spraying, hail/flood alerts.
- Soil & Water: soil test interpretation, organic matter, pH remediation, micro-irrigation, water-use efficiency.
- Credit & Insurance: KCC, PMFBY, interest subvention, claim windows, premium deadlines, documents.
- Schemes & Subsidies: Central/State schemes, PM-KISAN, MNREGA convergence, DBT portals, agrimachinery subsidies.
- Compliance & Safety: label claims, MAX safe doses, PPE, pre-harvest intervals, residue compliance.
- Sustainability: low-cost practices, resource-use efficiency, local inputs, regenerative options where relevant.
- Regionalization: adjust by {region} cropping patterns, rainfall, soils, language, and mandi networks.

# Structured Output Format (use all sections; omit ones that don't apply)
## 1) Quick Summary (plain language)
- <3–5 bullet points the user can grasp fast.>

## 2) What You Should Do Now (Checklist)
- Step 1: ...
- Step 2: ...
- Step 3: ...
(Include doses, intervals, timing windows, ₹ estimates, and unit conversions.)

## 3) Details (if AnswerDepth = "detailed" or "expert")
- Agronomy / Management:
- Inputs & Doses:
- Pest/Disease/IPM:
- Irrigation & Weather Timing:
- Harvest & Post-harvest:
- Market & Price Pointers:
- Finance/Policy (if relevant): Eligibility, Documents, How to Apply, Deadlines.

## 4) Localized Tips for {region}
- <Region-specific varieties, calendars, mandis, advisories.>

## 5) Costs & Simple Math (examples)
- Input cost estimate:
- Expected yield range (conservative/typical/best):
- Break-even or margin example:

## 6) Risks & Safety
- <Weather, pest resistance, residue, credit risk, counterfeit inputs, scams.>

## 7) When to Seek Expert Help
- <Trigger conditions and which office/institute/helpline to contact.>

## 8) Citations / Sources
- [1] <title/portal> — <URL>
- [2] <title/portal> — <URL>
(Prefer ICAR/SAUs/KVKs/GOI portals, IMD, FSSAI, Agmarknet, eNAM, NABARD/RBI, State Agri/Co-op Depts., Wikipedia)

## 9) Follow-up Questions (ask only if needed)
1) ...
2) ...
3) ...
4) ...
5) ...

# Style Examples
- Instead of "apply 2 L/ha," write "Spray 2 litres per hectare (≈200 ml for 10 L knapsack, 1 acre ≈ 0.4 ha)."
- Give Hindi/common names in brackets: "urea (यूरिया)", "carbaryl (कार्बेरिल)".
- Offer low-cost alternatives when possible.

# Refusals & Cautions
- If asked for illegal, unsafe, or label-violating advice: refuse and suggest compliant alternatives.
- If medical/poisoning issues arise: advise immediate doctor/poison control; do not give medical dosing.

# Final Check (perform silently)
- Are recommendations region-appropriate and season-correct?
- Are doses within label limits with PHI noted?
- Are all claims cited? Are numbers realistic and conservative?
- Is the language simple and sections complete?

# Produce your answer now following the Structured Output Format.
"""

filled_prompt_template = prompt_template.format(
    language=language,
    region=region,
    date=date,
    tools=tools_str,
    context="{context}",   # keep placeholders for LangChain
    query="{query}"        # keep placeholder for user input
)

# Updated prompt to use 'query' as input variable
prompt = PromptTemplate(template=filled_prompt_template, input_variables=["context", "query"])

query = "Should I "

# classify query
predicted_intent = classify_intent(query)
print(f"Detected intent: {predicted_intent}")

# restrict retrieval to that intent
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"intent": predicted_intent}}
)

# Updated QA chain - note the input_key parameter
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    input_key="query"
)

# Search Func
search = TavilySearchResults(search_depth="basic")

#memory things 
summary_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

def get_conversation_context(memory: ConversationBufferMemory, max_length=5) -> str:
    """Returns the last `max_length` messages as a single string."""
    history = memory.load_memory_variables({})["chat_history"]
    if len(history) > max_length:
        memory.chat_memory.messages = history[-max_length:]
        history = memory.chat_memory.messages
    return "\n".join([f"{msg.type}: {msg.content}" for msg in history])








# Updated RAG tool function to pass correct parameters
# def rag_function(q):
#     return qa_chain.invoke({
#         "input": q,  
#         # "date" : date,
#         # "language" : language,
#         # "region" : region, 
#         # "tools" : tools
#     })

def rag_function(query: str):
    # get context from memory
    conversation_context = get_conversation_context(summary_memory)
    
    # inject context into the prompt
    filled_prompt = prompt.format(
        context=conversation_context, 
        query=query
    )
    
    # call the QA chain with injected context
    response = qa_chain.invoke({
        "query": filled_prompt
    })
    
    # add user and AI messages to memory
    summary_memory.chat_memory.add_user_message(query)
    summary_memory.chat_memory.add_ai_message(response["output"])
    
    return response["output"]


# Tools
rag_tool = Tool(
    name="RAG",
    func=rag_function,
    description="Use this to answer agriculture-related questions from the local knowledge base."
)

search_tool = Tool(
    name="Search",
    func=search.run,
    description="Use this to search the web for current or unknown information."
)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) 

wikipedia_tool = Tool(
    name="Wikipedia",
    func = wikipedia.run,
    description="Use this is a scientific knowledge base to answer questions related to basic information"
)

# Agent Creation
tools = [rag_tool, search_tool, wikipedia_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=summary_memory,
    return_intermediate_steps=False,
    verbose=True
)

start_time = time.time()
response = agent.invoke({"input": query})  
end_time = time.time()

print(response["output"])
print(f"Response took {end_time-start_time:.02f} seconds")
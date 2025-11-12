from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# --- Load environment variables ---
load_dotenv()



MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")



# ------------------- DATABASE CONNECTION -------------------
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)



db = init_database(
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    database=MYSQL_DATABASE
)

if "db" not in st.session_state:
    st.session_state.db = db
# ------------------- SQL GENERATION CHAIN -------------------
def get_sql_chain(db):
    template = """
     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
  
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
     SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
 
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY missing from environment. Locally use a .env file.")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def get_schema(_):
        schema = db.get_table_info()
        return schema[:6000]  # Limit length for safety

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# ------------------- RESPONSE LOGIC -------------------
def get_response(user_query: str, db: SQLDatabase):
    # Step 1: Generate SQL query
    sql_chain = get_sql_chain(db)
    query = sql_chain.invoke({
        "question": user_query,
    })

    # Step 2: Explain SQL query
    explain_prompt = ChatPromptTemplate.from_template("""
    Explain clearly and simply what the following SQL query does and what data it returns:
    {query}
    """)
    explain_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    explain_chain = explain_prompt | explain_llm | StrOutputParser()
    explanation = explain_chain.invoke({"query": query})

    # Step 3: Run query and handle errors
    try:
        response_data = db.run(query)
    except Exception as e:
        response_data = f"Error running query: {e}"

    # Step 4: Create concise natural language answer
    nl_prompt = ChatPromptTemplate.from_template("""
    You are a helpful SQL assistant. Based on the schema, question, SQL query, and SQL response,
    write a **short, factual answer** to the user's question.
    **Do not add explanations, speculations, or troubleshooting.**
    Only return the direct answer.

    <SCHEMA>{schema}</SCHEMA>
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    """)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    schema_text = db.get_table_info()[:4000]
    final_chain = nl_prompt | llm | StrOutputParser()
    final_answer = final_chain.invoke({
        "schema": schema_text,
        "question": user_query,
        "query": query,
        "response": str(response_data)[:3000]
    })

    return {"query": query, "explanation": explanation, "answer": final_answer}

# ------------------- STREAMLIT APP -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your SQL assistant. Ask me anything about your database.")
    ]
st.image(
    "p.jpg",
    width=100
)
st.title("Learn SQL by Asking Questions")
st.write("Type a question and discover how SQL works. You'll see the generated query, a step-by-step explanation, and the query results in real time.")




 


# --- Chat Display ---
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI",avatar="assistance.png"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human",avatar="laptop.png"):
            st.markdown(message.content)

# --- Chat Input ---
user_query = st.chat_input("Type your question...")
if user_query and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human",avatar="laptop.png"):
        st.markdown(user_query)

    with st.chat_message("AI",avatar="assistance.png"):
        outputs = get_response(user_query, st.session_state.db)

        st.markdown("**Generated SQL Query:**")
        st.code(outputs["query"], language="sql")

        st.markdown("**Explanation of SQL Query:**")
        st.info(outputs["explanation"])

        st.markdown("**Final Answer:**")
        st.success(outputs["answer"])

    # Save only the final answer text to chat history
    chat_text = outputs.get("answer") or outputs.get("explanation") or outputs["query"]
    st.session_state.chat_history.append(AIMessage(content=chat_text))

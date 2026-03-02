import streamlit as st
from PyPDF2 import PdfReader, errors as PyPDF2errors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuration and Initialization ---

st.set_page_config(
    page_title="PDF Q&A ChatBot",
    page_icon="https://img.icons8.com/bubbles/100/pdf-2.png",
    layout="wide",
    initial_sidebar_state="auto"
)

# Define the logo image URL
IMAGE_URL = "https://img.icons8.com/bubbles/100/pdf-2.png"

# --- Custom UI Style ---
st.markdown("""
<style>
/* Adjust top padding to pull content higher */
.main .block-container {
    padding-top: 0.5rem; 
}
footer {visibility: hidden;}

/* Inner container for the Icon and H1 */
.header-container {
    display: flex; 
    align-items: center; 
    gap: 10px;
}

/* Ensure the H1 inside the container has zero margin for alignment */
.header-container h1 {
    margin: 0px !important;
    font-size: 2.25rem !important;
    line-height: 1.2 !important;
}
/* Adjust the button size and alignment */
.stButton>button {
    height: 38px; 
    margin-top: 5px; 
}

/* Remove margin above the chat content area */
.stApp > div:nth-child(1) > div:nth-child(1) > div.main > div.block-container > div:nth-child(1) > div {
    margin-top: 0 !important;
}

/* Standard chat and social link styling */
.stChatMessage {
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.social-links {
    display: flex;
    gap: 15px;
    margin-top: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_TEMPERATURE = 0.3

if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
if 'show_info' not in st.session_state: 
    st.session_state.show_info = False

try:
    # Get the API Key from Streamlit secrets
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 **Configuration Error:** `GEMINI_API_KEY` not found in `.streamlit/secrets.toml`. Please set it up to run the RAG pipeline.")
    st.stop()

# --- Core RAG Functions (Cached to save time on re-runs) ---

@st.cache_resource
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf.seek(0) 
            pdf_reader_obj = PdfReader(pdf)
            for page in pdf_reader_obj.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            st.toast(f"✅ Successfully read: {pdf.name}", icon="📄")
        except PyPDF2errors.PdfReadError as e:
            st.warning(f"⚠️ **Skipped:** Could not read **'{pdf.name}'**. (Error: {e})")
            continue
        except Exception as e:
            st.error(f"❌ **Error processing '{pdf.name}'**: {e}")
            continue
    return text

@st.cache_resource
def get_text_chunks(text):
    # Split the massive text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_conversational_chain(_vector_store):
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_LLM_MODEL,
        google_api_key=API_KEY,
        temperature=DEFAULT_TEMPERATURE
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# -------------------------------------------------------------------------
# --- UI Helper Functions ---
# -------------------------------------------------------------------------

def display_chat_history():
    """Loops through and displays all messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(user_question):
    """Handles new user questions, calls the RAG chain, and updates history."""
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first!")
        return

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response = st.session_state.conversation.invoke(
                    {"question": user_question}
                )
                assistant_response = response["answer"]
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
                return

        st.markdown(assistant_response) 
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.rerun()

# -------------------------------------------------------------------------
# --- MAIN INTERFACE EXECUTION ---
# -------------------------------------------------------------------------

def toggle_info():
    st.session_state.show_info = not st.session_state.show_info

# --- Title and Info Button Layout ---
col_title, col_button = st.columns([0.85, 0.15])

with col_title:
    st.markdown(
        f"""
        <div class="header-container">
            <img src='{IMAGE_URL}' alt='PDF Q&A ChatBot Logo' width='40' height='40'>
            <h1>PDF Q&A ChatBot</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_button:
    button_label = "⬅️ Back to Chat" if st.session_state.show_info else "❓ App Info"

  
    st.button(
        button_label, 
        use_container_width=True,
        on_click=toggle_info, 
        key="app_info_toggle_button"
    )

# ---App Info Display ---

if st.session_state.show_info:
    
    st.markdown("---") 
    
    # 1. How to Use
    st.subheader("🤖 How to Use This Chatbot")
    st.markdown("""
    **Upload Documents:** Go to the **Sidebar**, upload one or more PDF files using the **'Choose your PDF files'** uploader.
    **Process:** Click the **'🚀 Process Documents'** button. The app will extract text, create vector embeddings, and build a conversational chain.
    **Ask Questions:** Once you see the success message, you can ask questions about the content of your uploaded PDFs in the chat box below.
    """)
    
    # 2. Use Case
    st.subheader("💡 Use Case")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) ChatBot is perfect for analyzing **large, complex documents**.
    
    * **For Professionals:** Use it to quickly find specific clauses in **legal contracts**, extract key data from **financial reports**, or summarize long **product manuals**.
    * **For Students:** Upload multiple **research papers** or a large **textbook chapter** to quickly locate definitions, summarize key theories, or find relevant examples for your study notes and exam preparation.
    """)
    
    # 3. Developer Info
    st.subheader("👨‍💻 Developer Info")
    st.markdown(f"""
    **Developer:** Sanket Kshirsagar  
    **Contact:** sanketkshirsagar05@gmail.com
    
    <div class="social-links">
        <a href="https://www.linkedin.com/in/sanket-kshirsagar-0a416820b/" target="_blank">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn" width="24" height="24"> LinkedIn
        </a>
        <a href="https://github.com/Sanketkshirsagar05" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" width="24" height="24"> GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # --- SHOW CHAT INTERFACE ---
    st.markdown("---") 
    
    # Welcome Message
    if not st.session_state.conversation and not st.session_state.messages:
         st.info("👋 **Welcome!** Use the **Sidebar** to upload your files and begin the conversation.")

    # chat history
    display_chat_history()
    
    # The chat input box
    if prompt := st.chat_input("Ask a question about your processed documents..."):
        handle_user_input(prompt)


# 1. Sidebar
with st.sidebar:
    st.header("🗂️ Document Manager")

    with st.expander("📄 **Upload & Process PDFs**", expanded=True):

        FILE_UPLOADER_KEY = "pdf_uploader_key"

        pdf_docs = st.file_uploader(
            "**Choose your PDF files**",
            accept_multiple_files=True,
            type="pdf",
            help="Upload one or multiple PDF files.",
            key=FILE_UPLOADER_KEY
        )

        process_button = st.button("🚀 Process Documents", use_container_width=True, type="primary")

        # Display the green success box 
        if st.session_state.documents_processed:
            st.success(f"✅ Documents processed! {st.session_state.chunk_count} chunks ready.")

        if process_button:
            if pdf_docs:

                with st.spinner("Processing..."):

                    # Clear all caches and reset processing state
                    get_pdf_text.clear()
                    get_text_chunks.clear()
                    get_vector_store.clear()
                    get_conversational_chain.clear()
                    st.session_state.documents_processed = False

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text:
                        st.error("🚨 Failed to extract text. Check file integrity.")
                        st.session_state.conversation = None
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.conversation = get_conversational_chain(vector_store)

                        
                        st.session_state.chunk_count = len(text_chunks)
                        st.session_state.documents_processed = True
                        st.session_state.messages = []
                        st.session_state.messages.append({"role": "assistant", "content": "Documents processed successfully! Ask your first question."})

                        st.rerun()
            else:
                st.error("Please upload at least one PDF file.")

    st.markdown("---")

    with st.expander("🗑️ **Chat & App Reset**"):
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []

            # Add appropriate starting message
            if st.session_state.documents_processed:
                st.session_state.messages.append({"role": "assistant", "content": "Documents processed successfully! Ask your first question."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Upload and process your PDF(s) to start chatting!"})

            st.success("Chat history cleared! Ready to chat on current documents.")
            st.rerun()

        if st.button("Reset (Clear Cache)", use_container_width=True):

            # Force reset
            if FILE_UPLOADER_KEY in st.session_state:
                del st.session_state[FILE_UPLOADER_KEY]

            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Upload and process your PDF(s) to start chatting!"})

            # Delete key session variables
            if 'conversation' in st.session_state:
                del st.session_state['conversation']

            # Clear all cached
            get_pdf_text.clear()
            get_text_chunks.clear()
            get_vector_store.clear()
            get_conversational_chain.clear()

            # Reset document state flags
            st.session_state.documents_processed = False
            st.session_state.chunk_count = 0

            st.success("App cache and history cleared!")
            st.rerun()

    # --- Author/Created By Section ---
    st.markdown("---")
    st.subheader("📢 Get Help")
    st.markdown("Click the **'❓ App Info'** button at the top for usage and developer details.")

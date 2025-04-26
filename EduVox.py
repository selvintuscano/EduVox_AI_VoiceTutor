import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from fpdf import FPDF
from io import BytesIO
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# --- Document Processor Class ---
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()

    def load_documents(self, directory):
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }
        documents = []
        for loader in loaders.values():
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading documents: {str(e)}")
        return documents

    def process_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents, persist_directory):
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        vector_store = Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=persist_directory)
        vector_store.persist()
        return vector_store

# --- AI Voice Tutor Class ---
class AIVoiceTutor:
    def __init__(self, elevenlabs_api_key):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.voice_generator = ElevenLabs(api_key=elevenlabs_api_key)
        self.sample_rate = 44100
        self.qa_chain = None

    def setup_vector_store(self, vector_store):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=False,
        )

    def record_audio(self, duration=5):
        if not hasattr(self, 'whisper_model'):
            self.whisper_model = whisper.load_model("base")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_array, self.sample_rate)
            result = self.whisper_model.transcribe(temp_audio.name)
            os.unlink(temp_audio.name)
        return result["text"]

    def translate(self, text, target_language):
        if target_language == "English":
            return text
        prompt = f"Translate this to {target_language}: {text}"
        translated = self.llm.invoke(prompt)
        return translated.content

    def generate_response(self, query, mode="default", language="English"):
        if self.qa_chain is None:
            return "Error: Knowledge base not initialized."
        prompts = {
            "summarize": f"Summarize this topic: {query}",
            "quiz": f"Generate a short quiz (5 multiple-choice questions) on: {query}. Format: Q1, options (A, B, C), and 'Answer:'.",
            "simplify": f"Explain in simple terms: {query}",
            "default": query
        }
        prompt = prompts.get(mode, query)
        response = self.qa_chain.invoke({"question": prompt})["answer"]
        if language == "Hindi":
            response = self.translate(response, "Hindi")
        return response

    def generate_voice(self, text, voice_name):
        try:
            audio_gen = self.voice_generator.generate(text=text, voice=voice_name, model="eleven_multilingual_v2")
            audio_bytes = b"".join(audio_gen)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name
        except:
            return None

    def export_to_pdf(self, content):
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('DejaVu', '', fname='fonts/DejaVuSans.ttf', uni=True)
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(0, 10, content)
        pdf_bytes = pdf.output(dest='S').encode('utf-8')
        return BytesIO(pdf_bytes)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Voice Tutor", layout="wide")
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "AI Voice Tutor"])

elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
doc_processor = DocumentProcessor()

# --- Setup Knowledge Base ---
def setup_kb():
    st.title("📚 Setup Knowledge Base")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "md"])
    if uploaded_files and st.button("⚡ Process Documents"):
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        docs = doc_processor.load_documents(temp_dir)
        processed = doc_processor.process_documents(docs)
        vector_store = doc_processor.create_vector_store(processed, "knowledge_base")
        st.session_state.vector_store = vector_store
        st.success(f"✅ Processed {len(processed)} document chunks!")
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

# --- Tutor Mode ---
def tutor_mode():
    st.title("🎓 AI Voice Tutor")
    st.markdown("Welcome to your personalized AI learning assistant. Ask questions via voice or text, get summaries, quizzes, and more!")

    if "vector_store" not in st.session_state:
        st.error("🚨 Please setup the Knowledge Base first in the sidebar!")
        return

    if "tutor_instance" not in st.session_state:
        st.session_state.tutor_instance = AIVoiceTutor(elevenlabs_api_key)
        st.session_state.tutor_instance.setup_vector_store(st.session_state.vector_store)

    tutor = st.session_state.tutor_instance

    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("Language & Mode")
    language = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"])
    mode = st.sidebar.selectbox("🎯 Mode", ["default", "summarize", "quiz", "simplify"])

    st.sidebar.subheader("Voice Options")
    voice_name = st.sidebar.selectbox("🗣️ Voice", ["Rachel", "Domi", "Bella", "Antoni", "Elli", "Josh", "Arnold", "Adam", "Sam"])
    duration = st.sidebar.slider("🎤 Recording Duration (sec)", 1, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Voice Tutor v1.0 | Powered by OpenAI, LangChain, ElevenLabs")

    st.subheader("💬 Ask Your Question")
    input_mode = st.radio("Choose Input Method", ["Voice", "Text"])

    if input_mode == "Voice":
        if st.button("🎤 Start Voice Query"):
            with st.spinner(f"Recording for {duration} seconds..."):
                audio = tutor.record_audio(duration)
            with st.spinner("Transcribing your audio..."):
                query = tutor.transcribe_audio(audio)
            st.chat_message("user").markdown(f"**You said:** {query}")
            with st.spinner("Generating response..."):
                response = tutor.generate_response(query, mode=mode, language=language)
            display_response(response, tutor, voice_name)
    else:
        query = st.text_input("Type your question here:")
        if st.button("⚡ Process Text Query") and query:
            st.chat_message("user").markdown(f"**You:** {query}")
            with st.spinner("Generating response..."):
                response = tutor.generate_response(query, mode=mode, language=language)
            display_response(response, tutor, voice_name)

# --- Display Response Helper ---
def display_response(response, tutor, voice_name):
    if not response:
        st.error("No response generated. Please try again.")
        return

    if "Q1" in response and "Answer:" in response:
        render_quiz(response)
    else:
        if len(response) > 300:
            with st.expander("📄 View Full Response"):
                st.write(response)
        else:
            st.chat_message("assistant").markdown(f"**Tutor:** {response}")

    audio_file = tutor.generate_voice(response, voice_name)
    if audio_file:
        st.audio(audio_file)
        os.remove(audio_file)

    pdf = tutor.export_to_pdf(response)
    st.download_button("📄 Download Response as PDF", data=pdf, file_name="response.pdf", mime="application/pdf")

# --- Interactive Quiz Renderer ---
def render_quiz(quiz_text):
    st.subheader("📝 Interactive Quiz")
    questions = quiz_text.split("Q")[1:]

    score = 0
    total = len(questions)

    for idx, q in enumerate(questions):
        lines = q.strip().split("\n")
        question_text = lines[0]
        options = [line for line in lines if line.strip().startswith(('A', 'B', 'C', 'D'))]
        answer_line = [line for line in lines if "Answer:" in line][0]
        correct_answer = answer_line.split(":")[1].strip()

        st.write(f"**Q{idx+1}: {question_text}**")
        user_choice = st.radio("Choose an option:", options, key=question_text)

        if st.button(f"Submit Answer for Q{idx+1}", key=f"submit_{idx}"):
            if user_choice.startswith(correct_answer):
                st.success("✅ Correct!")
                score += 1
            else:
                st.error(f"❌ Incorrect! Correct answer: {correct_answer}")

    st.info(f"🏆 Final Score: {score} / {total}")

# --- Page Routing ---
if page == "Setup Knowledge Base":
    setup_kb()
else:
    tutor_mode()

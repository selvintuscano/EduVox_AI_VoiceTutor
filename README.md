# EduVox: AI Voice Tutor

<p align="center">
  <em>"Your Voice-Enabled Learning Companion"</em>
</p>

## Project Resources
GitHub Repository: https://github.com/selvintuscano/EduVox_AI_VoiceTutor

### Presentations:
1.	PowerPoint Project Presentation: 
https://drive.google.com/file/d/1H-iwAAAZsFp0_RJ_pFPG0nw-ldDV5r78/view?usp=drivesdk

2.	Code and Streamlit App Walkthrough: 
https://drive.google.com/file/d/1HXs68IqLxASY7ypSj7peMQnk0Ix_9FpB/view?usp=drivesdk
Note: Haven’t deployed Streamlit App as it contains OpenAI API key and eleven labs keys. Have provided screenshots of Streamlit App below
![image](https://github.com/user-attachments/assets/c2ad298d-58ef-4f15-a46e-3e5cf217d04c)


## 📋 Overview

EduVox is an innovative AI-powered voice tutoring system designed to transform traditional learning into an interactive, personalized experience. By leveraging advanced technologies in natural language processing, speech recognition, and retrieval-augmented generation, EduVox creates a dynamic learning environment that responds intelligently to user queries through both voice and text interfaces.

## ✨ Key Features

- **Voice & Text Interaction**: Communicate naturally using voice or text input
- **Personalized, Context-Aware Responses**: Get tailored answers based on uploaded learning materials
- **Multiple Learning Modes**: Default, Summarize, Quiz, and Simplify modes
- **Multilingual Support**: Currently supports English and Hindi
- **PDF Export**: Save responses for offline reference
- **Interactive Quiz Generation**: Automatically creates quizzes based on content

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **RAG Implementation**: LangChain + ChromaDB
- **Language Model**: GPT-4o-mini (OpenAI)
- **Speech-to-Text**: Whisper
- **Text-to-Speech**: ElevenLabs
- **Document Processing**: PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader

## 🏗️ System Architecture

EduVox combines multiple AI technologies through a unified LangChain framework:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangChain Orchestrator                      │
└─────┬───────────────────┬────────────────────┬──────────────────┘
      │                   │                    │
      ▼                   ▼                    ▼
┌───────────┐     ┌─────────────┐     ┌──────────────────┐
│   GPT-4o  │     │ ChromaDB    │     │ ElevenLabs       │
│ Language  │     │ Vector      │     │ Text-to-Speech   │
│   Model   │     │ Storage     │     │                  │
└───────────┘     └─────────────┘     └──────────────────┘
      ▲                   ▲                    ▲
      │                   │                    │
┌───────────┐     ┌─────────────┐     ┌──────────────────┐
│  Whisper  │     │ Document    │     │ PDF Export       │
│Speech-to- │     │ Processor   │     │ Service          │
│   Text    │     │             │     │                  │
└───────────┘     └─────────────┘     └──────────────────┘
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/selvintuscano/EduVox_AI_VoiceTutor.git
   cd EduVox_AI_VoiceTutor
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
   ```

4. Run the application:
   ```bash
   streamlit run EduVox.py
   ```

## 💡 How It Works

1. **Knowledge Base Setup**:
   - Upload educational documents (PDF, TXT, MD)
   - System processes and stores them as vector embeddings in ChromaDB

2. **Query Processing**:
   - Voice inputs are transcribed using Whisper
   - Text inputs are processed directly
   - System retrieves relevant information from the knowledge base

3. **Response Generation**:
   - LangChain orchestrates the retrieval and generation process
   - GPT-4o creates contextually relevant responses
   - Responses are delivered as text, voice, or downloadable PDF

## 📊 Workflow Diagram

```
[ User Question 🗣️ ]
         │
         ▼
🔍 Retrieve Relevant Info
         │
         ▼
[ Vector Database (ChromaDB) ]
         │
         ▼
📄 Pass Data to GPT via LangChain
         │
         ▼
🤖 Generate Answer (LLM)
         │
         ▼
[ Response: Text 📝 | Voice 🔊 | PDF 📃 ]
```

## 🧩 Components

### Document Processing

The `DocumentProcessor` class handles content ingestion:
- Supports multiple file formats
- Chunks text with configurable size and overlap
- Creates vector embeddings for semantic search

### Voice Integration

Two-way voice communication:
- Speech-to-Text via Whisper
- Text-to-Speech via ElevenLabs with multiple voice options

### LangChain Framework

Serves as the orchestration layer:
- Manages conversation context and history
- Connects retrieval system with language model
- Handles prompt management for different modes

### Interactive Quiz Generation

Automatically creates quizzes from content:
- Parses questions and answer options
- Tracks user scores
- Provides immediate feedback

## 🧠 Retrieval-Augmented Generation (RAG)

EduVox implements RAG to enhance response quality:
- **Enhanced factuality**: Responses grounded in uploaded documents
- **Improved accuracy**: Contextually relevant information retrieval
- **Reduced hallucination**: AI doesn't guess—it responds based on your documents
- **Adaptability**: No need for model retraining

## 🛣️ Future Roadmap

- Expanded language support
- Enhanced UI design
- Adaptive learning features
- Mobile application deployment
- Advanced RAG techniques

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- [Selvin Tuscano](https://github.com/selvintuscano)

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI](https://openai.com/)
- [ElevenLabs](https://elevenlabs.io/)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)

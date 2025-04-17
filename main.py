import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Configuración de la página
st.set_page_config(page_title="PDF Q&A App")
st.header("📄 Pregunta a tu documento PDF")

# Entrada de la API Key de OpenAI
st.markdown("## 🔑 Ingresa tu OpenAI API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.warning('Por favor, ingresa tu OpenAI API Key para continuar.', icon="⚠️")
    st.stop()

# Cargador de PDF
st.markdown("## 📂 Sube tu archivo PDF")
pdf_file = st.file_uploader("Selecciona un archivo PDF", type=["pdf"])

if pdf_file:
    # Guardar temporalmente el PDF
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Cargar y dividir el documento
    loader = PyPDFLoader("temp.pdf")
    documentos = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos)

    # Crear embeddings y base vectorial
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(fragmentos, embeddings)

    # Definir prompt para la cadena de QA
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres un asistente útil. Utiliza la siguiente información extraída del documento para responder la pregunta.

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""
    )

    # Crear la cadena de Retrieval QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}
    )

    # Entrada de consulta del usuario
    pregunta = st.text_input("❓ Escribe tu pregunta sobre el documento PDF")
    if pregunta:
        respuesta = qa_chain.run(pregunta)
        st.markdown("### 💡 Respuesta:")
        st.write(respuesta)

# Nota: El resto de la estructura de la app permanece igual según tu petición.

        tone=option_tone, 
        dialect=option_dialect, 
        draft=draft_input
    )

    improved_redaction = llm(prompt_with_draft)

    st.write(improved_redaction)

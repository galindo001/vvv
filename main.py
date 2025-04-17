import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Página de configuración
st.set_page_config(page_title="Multi-Tool AI App")
st.header("✍️ Reescribe tu texto & 📄 Pregunta a un PDF")

# Entrada de la API Key de OpenAI
st.markdown("## 🔑 Ingresa tu OpenAI API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.warning('Por favor, ingresa tu OpenAI API Key.', icon="⚠️")
    st.stop()

# ===== Sección 1: Re-writing de texto =====
st.markdown("---")
st.subheader("1. Re-write your text")

template = """
Below is a draft text that may be poorly worded.
Your goal is to:
- Properly redact the draft text
- Convert the draft text to a specified tone
- Convert the draft text to a specified dialect

DRAFT: {draft}
TONE: {tone}
DIALECT: {dialect}

YOUR {dialect} RESPONSE:
"""

prompt_rewrite = PromptTemplate(
    input_variables=["tone", "dialect", "draft"],
    template=template,
)

# Input de texto
draft_input = st.text_area("Escribe el texto a reescribir (máx. 700 palabras)", height=150)
if draft_input and len(draft_input.split()) <= 700:
    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox('Tono', ['Formal','Informal'], key='tone')
    with col2:
        dialect = st.selectbox('Dialect', ['American','British'], key='dialect')

    if draft_input:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        prompt_fmt = prompt_rewrite.format(draft=draft_input, tone=tone, dialect=dialect)
        result = llm(prompt_fmt)
        st.markdown("### Resultado:")
        st.write(result)
elif draft_input:
    st.error("Por favor, ingresa un texto de hasta 700 palabras.")

# ===== Sección 2: QA sobre PDF =====
st.markdown("---")
st.subheader("2. Pregunta a tu documento PDF")

pdf_file = st.file_uploader("Sube un PDF para hacer preguntas", type=['pdf'])
if pdf_file:
    # Guardar temporalmente el PDF
    path = "temp.pdf"
    with open(path, "wb") as f:
        f.write(pdf_file.read())

    # Cargar y dividir texto
    loader = PDFMinerLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Crear embeddings e índice
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = FAISS.from_documents(chunks, embeddings)

    # Prompt de QA
    qa_prompt = PromptTemplate(
        input_variables=["context","question"],
        template=(
            "Eres un asistente útil. Usa la información del documento para responder.\n\n"
            "Contexto:\n{context}\n\nPregunta:\n{question}\n\nRespuesta:"
        )
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        chain_type='stuff',
        retriever=index.as_retriever(),
        chain_type_kwargs={'prompt': qa_prompt}
    )

    pregunta = st.text_input("Escribe tu pregunta sobre el PDF")
    if pregunta:
        answer = qa.run(pregunta)
        st.markdown("### Respuesta:")
        st.write(answer)

# ===== Fin de la app =====

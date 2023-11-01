import os
import streamlit as st
import pickle

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from youtube_transcript_api import YouTubeTranscriptApi
import tiktoken

# import openai api key
from dotenv import load_dotenv
load_dotenv()  

# Initialize OpenAI model
llm = OpenAI(temperature=0.9, max_tokens=500)
compressor = LLMChainExtractor.from_llm(llm)
faiss_path = "faiss_store_openai.pkl"

# youtube transcription function
def yt_transcribe(url):

    # get video id
    id = url.split("v=")[1]

    transcript_path = "yt-transcript.txt"

    responses = YouTubeTranscriptApi.get_transcript(video_id=id)
    texts = " ".join(r.get("text") for r in responses)

    with open(transcript_path, "w+") as f:
        f.write(texts)

    return transcript_path

# create the length function
tokenizer = tiktoken.get_encoding('cl100k_base') # for gpt-3.5 turbo or gpt-4

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Streamlit title
st.title("Youtube Video Q&A")
url = st.sidebar.text_input("YouTube URL")
main_placeholder = st.empty()
process_url_clicked = st.sidebar.button("Process URL")

# Read the files, do word embedding, and store it in vector store index FAISS as a pickle file
if process_url_clicked:
    # transcribe youtube and save transcript
    main_placeholder.text("⏳ Processing YouTube Video...")
    transcript_path = yt_transcribe(url)
    
    # load transcript
    transcript = TextLoader(transcript_path).load()

    # Split the data into chunks
    main_placeholder.text("✅⏳ Processing the transcript...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.'], 
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len
        )
    chunks = text_splitter.split_documents(transcript)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Store in FAISS index
    vectorstore_openai = FAISS.from_documents(chunks, embeddings)

    # Save as pickle file
    with open(faiss_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    main_placeholder.text("✅✅ Transcript processed and embeddings saved!")

st.text("Ask me anything about the video👇")
query = st.text_input("Question:")


# Answer according to the content of the uploaded text files
chat_history = []
if query:
    # Check if the FAISS index pickle file exists
    if os.path.exists(faiss_path):
        with open(faiss_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # compression retreiver to optimize 
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
                )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                chain_type="map_reduce",
                retriever=compression_retriever
            )
            
            result = chain(
                {"question": query, "chat_history": chat_history}, return_only_outputs=True
                )
            
            st.header("Answer")
            st.write(result["answer"])

            chat_history.append(query)
            chat_history.append(result["answer"])


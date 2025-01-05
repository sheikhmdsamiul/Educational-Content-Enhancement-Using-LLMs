import streamlit as st
import os
import re
import fitz  
import json
from gtts import gTTS
from dotenv import load_dotenv
from summa import summarizer
import math
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.combine_documents import create_stuff_documents_chain
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq for chat pdf
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.sidebar.error("GROQ_API_KEY is not set. Please set it in the .env file.")
    st.stop()

model = 'llama-3.1-70b-versatile'

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name=model
)

# Initialize Groq for quiz
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# Define functions for file handling and processing
def save_uploaded_files(uploaded_files, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for uploaded_file in uploaded_files:
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return save_dir

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def pdf_read(pdf_directory):
    text_content = []
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, file_name)
            text = extract_text_from_pdf(file_path)
            text_content.append(text)
    return text_content

def pdf_read2(pdf_directory):
    loader = PyPDFDirectoryLoader(pdf_directory)
    data = loader.load()
    return data

# Return vectorstore for the documents
def get_vector_store(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(data)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"

def fetch_corrected_text(text_content):

    PROMPT_TEMPLATE = """
    Text: {text_content}
    You are an expert in regenerating provided content in a readable format. 
    Given the above text, regenerate it by making it readable. Make every header text visible to the user.
    Do not change anything, not a single word or information should be missed in your output from the provided content.
    Make sure to keep every word same as it is written in the provided content, you can ignore texts like Copyright or All rights reserved.
    """

    formatted_template = PROMPT_TEMPLATE.format(text_content=text_content)

    # Make API request
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": formatted_template
            }
        ]
    )

# Extract response content
    extracted_response = response.choices[0].message.content
    return extracted_response

@st.cache_data
def fetch_summary(text_content, PROMPT_TEMPLATE):

    formatted_template = PROMPT_TEMPLATE.format(text_content=text_content)

    # Make API request
    response = client.chat.completions.create(
        model="gemma-7b-it",
        messages=[
            {
                "role": "user",
                "content": formatted_template
            }
        ]
    )

    # Extract response content
    extracted_response = response.choices[0].message.content
    return extracted_response


# Returns history_retriever_chain
def get_retriever_chain(vector_store):
    llm = groq_chat
    retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    #compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", """Based on the above conversation, generate a search query that retrieves the most relevant and up-to-date information for the user. Focus on key topics, entities, or concepts that are directly related to the user's query. 
        Make sure the search query is specific and targets the most relevant sources of information.""")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, compression_retriever, prompt)

    return history_retriever_chain

# Returns conversational rag
def get_conversational_rag(history_retriever_chain):
    llm = groq_chat
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a highly knowledgeable assistant, your task is to answer any task or query of the user, using information retrieved from provided PDF documents.
        Your goal is to provide clear and accurate answers based on the retrieved context. 
        If the answer is not directly available, say: "I couldn't find this information in the provided documents."
        Be concise, but thorough.
        \n\nContext snippets used in response:\n\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    # Create final retrieval chain
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

    return conversational_retrieval_chain

# Returns the final response
def get_response(user_input):
    history_retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]

@st.cache_data
def fetch_questions(text_content, quiz_level):
    RESPONSE_JSON = '''{
    "mcqs": [
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4"
            },
            "correct": "a"
        },
        {
            "mcq": "multiple choice question2",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4"
            },
            "correct": "b"
        },
        {
            "mcq": "multiple choice question3",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4"
            },
            "correct": "c"
        }
    ]
}'''

    PROMPT_TEMPLATE = """
    Text: {text_content}
    You are an expert in generating MCQ type quiz on the basis of provided content. 
    Given the above text, create a quiz of 10 multiple choice questions keeping difficulty level as {quiz_level}. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to keep the format of your response like RESPONSE_JSON below and use it as a guide, do not do anything extra. 
    Ensure to make an array of 3 MCQs referring the following response json.
    Here is the RESPONSE_JSON: 

    {RESPONSE_JSON}
    """

    formatted_template = PROMPT_TEMPLATE.format(text_content=text_content, quiz_level=quiz_level, RESPONSE_JSON=RESPONSE_JSON)

    # Make API request
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": formatted_template
            }
        ]
    )

    # Extract response content
    extracted_response = response.choices[0].message.content
    print("Raw API response:", repr(extracted_response))  # Use print for console logging

    # Remove backticks and extra whitespace
    extracted_response = extracted_response.strip()

    # Use regex to extract JSON content between curly braces
    json_match = re.search(r"\{.*\}", extracted_response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0).strip()  # Get matched JSON string
    else:
        print("No valid JSON found in the response.")
        st.write("No valid JSON found in the response.")
        return []

    # Attempt to parse the cleaned JSON string
    try:
        json_data = json.loads(json_str)
        print("JSON is valid.")
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
        st.write("Invalid JSON response received.")
        return []

    return json_data.get("mcqs", [])


def fetch_ques_activity_list(text_content):

    sample_res="""  Exercise 1: sample question 1 

                    Activity List:

                    Activity 1: sample activity 1

                    Activity 2: sample activity 2

                    Activity 3: sample activity 3 and so on

"""

    PROMPT_TEMPLATE = """
    Text: {text_content}

    You are an expert in designing Active Learning Exercises specifically aligned to this text. Ensure each exercise is directly tied to the main ideas and topics in the provided text. 

    Avoid any speculative, unrelated, or off-topic exercises. Only generate exercises that are explicitly supported by the text. 

    The structure should follow this format:

    {sample_res}

    Please make sure the exercises cover various complexity levels from simple to advanced, and are directly tied to the provided content. Double-check each exercise for contextual relevance.
    """

    formatted_template = PROMPT_TEMPLATE.format(text_content=text_content, sample_res=sample_res )

    # Make API request
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": formatted_template
            }
        ]
    )

# Extract response content
    extracted_response = response.choices[0].message.content
    return extracted_response

def summarize_textrank(text):
    # Generate summary as a list of sentences
    summary = summarizer.summarize(text, split=True)
    return summary

def add_alpha(sentences, summary_sentences, highlight_color="green"):
    def transform(score):
        return math.exp(score * 5)
    
    scores = []
    highlighted_text = ""

    for sentence in sentences:
        if sentence in summary_sentences:
            score = transform(1.0)  # Assign a high score to sentences in the summary
        else:
            score = transform(0.0)  # Assign a low score to non-summary sentences
        
        scores.append(score)

    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score + 1
    
    for i, sentence in enumerate(sentences):
        alpha = round((scores[i] - min_score + 1) / span, 4) * 50
        if alpha > 25:  # Threshold for highlighting
            highlighted_text += f'<span style="background-color:{highlight_color};">{sentence}</span> '
        else:
            highlighted_text += f"{sentence} "
    
    return highlighted_text

# Main app
def main():
    st.set_page_config("Study With AI 📖")
    st.header("Study With AI 📖")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        save_dir = "uploaded_pdfs"

    if pdf_files:
        save_uploaded_files(pdf_files, save_dir)
        st.sidebar.success("PDF Uploaded and Processed")

        # Dropdown for user to select an option
        option = st.sidebar.selectbox("Choose an option", ["Summary with Question Suggestion", "Key Sentence Highlighter","AI Assistant", "Active Learning Exercises","Give a Quiz"])
        

        # Clear session state based on the option selected
        if option == "AI Assistant":
            st.session_state.quiz_generated = False  # Reset quiz-related session state

        if option == "Summary with Question Suggestion":
            st.subheader("Summary with Question Suggestion📝")
            # Dropdown for selecting quiz level
            sum_style = st.selectbox("Summary Style:", ["Infographic-Style Summary", "Detailed Summary"])

            # Initialize session_state
            session_state = st.session_state
            
            save_uploaded_files(pdf_files, save_dir)
            raw_text = pdf_read(save_dir)

            if sum_style == "Infographic-Style Summary":

                PROMPT_TEMPLATE = """
                Text: {text_content}
                -Given the provided content, briefly summarize it by selecting the most critical sentences or phrases directly from the text. -Ensure the summary captures the key points and main ideas without including unnecessary details. -The summary should closely mirror the original wording and structure for accurate evaluation. -The final summary should be no longer than 6 sentences.
                """
        
                # Storing raw text in session_state
                if 'raw_text' not in session_state:
                    session_state.raw_text = raw_text

                summary = fetch_summary(text_content=session_state.raw_text, PROMPT_TEMPLATE=PROMPT_TEMPLATE)

                st.subheader(summary)

            if sum_style == "Detailed Summary":

                PROMPT_TEMPLATE = """
                Text: {text_content}
                - Provide a comprehensive, in-depth summary of the provided text.
   - Include detailed explanations, examples, and descriptions.
   - Organize the content with clear headings and subheadings.
   - Ensure that the summary covers all major points thoroughly. At the end provide some question suggestions. 
                """
        
                # Storing raw text in session_state
                if 'raw_text' not in session_state:
                    session_state.raw_text = raw_text

                summary = fetch_summary(text_content=session_state.raw_text, PROMPT_TEMPLATE=PROMPT_TEMPLATE)

                st.subheader(summary)

            # Button to convert text to speech
            if st.button("🔊"):
                audio_file = text_to_speech(summary)
                st.audio(audio_file, format="audio/mp3")

        if option =="Key Sentence Highlighter":
            st.subheader("Key Sentence Highlighter📝")

            texts = pdf_read(save_dir)

            # Storing raw text in session_state
            if 'texts' not in st.session_state:
                st.session_state.texts = texts

            text=fetch_corrected_text(st.session_state.texts)

            if text:
                # Split the text into sentences
                sentences = text.split('. ')
    
                # Get summary sentences using TextRank
            summary_sentences = summarize_textrank(text)

            #st.write(text)
    
            # Highlight key sentences
            highlighted_text = add_alpha(sentences, summary_sentences)

            # Display highlighted text
            st.markdown(highlighted_text, unsafe_allow_html=True)
            # Button to convert text to speech
            if st.button("🔊"):
                audio_file = text_to_speech(text)
                st.audio(audio_file, format="audio/mp3")

        if option == "AI Assistant":
            st.subheader("AI Assistant📝")
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            raw_text = pdf_read2(save_dir)

            if "vector_store" not in st.session_state:
                st.session_state.vector_store = get_vector_store(raw_text)
                st.session_state.chat_history = [AIMessage(content="Hi, how can I help you?")]       
            
            # User input through chat interface
            user_input = st.chat_input("Type your message here...")
            if user_input is not None and user_input.strip() != "":
                response = get_response(user_input)
        
                # Update chat history
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))
            

            # Display chat history
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                        audio_file = text_to_speech(message.content)
                        st.audio(audio_file, format="audio/mp3")
                        
                else:
                    with st.chat_message("Human"):
                        st.write(message.content)
                        
                        
            
        if option == "Active Learning Exercises":
            st.subheader("Active Learning Exercises📝")

            # Initialize session_state
            session_state = st.session_state

            # Define questions and options
            save_uploaded_files(pdf_files, save_dir)
            raw_text = pdf_read(save_dir)
        
            # Storing raw text in session_state
            if 'raw_text' not in session_state:
                session_state.raw_text = raw_text

            k_ques = fetch_ques_activity_list(text_content=session_state.raw_text)

            st.write(k_ques)


        elif option == "Give a Quiz":
            st.subheader("MCQ quiz📝")
            raw_text = pdf_read(save_dir)
            # Dropdown for selecting quiz level
            quiz_level = st.selectbox("Select quiz level:", ["Easy", "Medium", "Hard"])

            # Convert quiz level to lower casing
            quiz_level_lower = quiz_level.lower()

            # Check if quiz_generated flag exists in session_state, if not initialize it
            if 'quiz_generated' not in st.session_state:
                st.session_state.quiz_generated = False

            # Track if Generate Quiz button is clicked
            if not st.session_state.quiz_generated:
                st.session_state.quiz_generated = st.button("Generate Quiz")

            if st.session_state.quiz_generated:
                # Define questions and options
                save_uploaded_files(pdf_files, save_dir)
                raw_text = pdf_read(save_dir)
                
                # Storing raw text in session_state
                if 'raw_text' not in st.session_state:
                    st.session_state.raw_text = raw_text

                questions = fetch_questions(text_content=st.session_state.raw_text, quiz_level=quiz_level_lower)

                # Display questions and radio buttons
                selected_options = []
                correct_answers = []
                for question in questions:
                    options = list(question["options"].values())
                    selected_option = st.radio(question["mcq"], options, index=None)
                    selected_options.append(selected_option)
                    correct_answers.append(question["options"][question["correct"]])

                # Submit button
                if st.button("Submit"):
                    # Display selected options
                    marks = 0
                    st.header("Quiz Result:")
                    for i, question in enumerate(questions):
                        selected_option = selected_options[i]
                        correct_option = correct_answers[i]
                        st.subheader(f"{question['mcq']}")
                        st.write(f"You selected: {selected_option}")
                        st.write(f"Correct answer: {correct_option}")
                        if selected_option == correct_option:
                            marks += 1
                    st.subheader(f"You scored {marks} out of {len(questions)}")

if __name__ == "__main__":
    main()

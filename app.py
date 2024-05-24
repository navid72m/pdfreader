import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import fitz
import io
from transformers import pipeline
import requests
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


my_token = os.getenv('my_repo_token')
def find_most_relevant_context(contexts, question, max_features=10000):
    # Vectorize contexts and question with limited features
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform([question] + contexts)
    
    # Compute cosine similarity between question and contexts
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get index of context with highest similarity
    most_relevant_index = similarity_scores.argmax()
    
    return contexts[most_relevant_index]








API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
headers = {"Authorization": f"Bearer {my_token}"}
pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")

def query(payload):
	# response = requests.post(API_URL, headers=headers, json=payload)
    
	# return response.json()
    return pipe(payload)
	



# Mock function for answering questions from the PDF
# Replace this with your actual backend function
def answer_question_from_pdf(pdf_text, question):
    # This function should return the answer to the question based on the PDF content
    # Here we just return a mock response
 
    return query(   {"inputs": "Based on this content: " + pdf_text+" The Question is: "+ question + " Provide the answer with max lenghth of about 100",})

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the PDF file
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    pdf_arr = []
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Get the page as an image
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # Perform OCR on the image
        pdf_text = pytesseract.image_to_string(img)
        pdf_arr.append(pdf_text)
    
    return pdf_arr
# Streamlit app
st.title("PDF Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from uploaded PDF
    pdf_arr = extract_text_from_pdf(uploaded_file)
    
    st.write("PDF Uploaded Successfully.")
    
    # Text area for entering a question
    question = st.text_input("Ask a question about the PDF")
    pdf_text = find_most_relevant_context(pdf_arr,question)
    
    if st.button("Get Answer"):
        if question:
            # Get the answer from the backend
            answer = answer_question_from_pdf(pdf_text, question)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file.")

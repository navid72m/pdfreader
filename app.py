import streamlit as st
from PyPDF2 import PdfReader
import io

# Mock function for answering questions from the PDF
# Replace this with your actual backend function
def answer_question_from_pdf(pdf_text, question):
    # This function should return the answer to the question based on the PDF content
    # Here we just return a mock response
    return f"Answer to your question: '{question}' based on the PDF content."

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text

# Streamlit app
st.title("PDF Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    st.write("PDF Uploaded Successfully.")
    
    # Text area for entering a question
    question = st.text_input("Ask a question about the PDF")
    
    if st.button("Get Answer"):
        if question:
            # Get the answer from the backend
            answer = answer_question_from_pdf(pdf_text, question)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file.")

import streamlit as st
import fitz  # this is PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# step 1
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# step 2

def calculate_similarity(resume_text, job_description):
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)  # percentage


# Streamlit UI
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matcher App")

uploaded_file = st.file_uploader("🔼 Upload your Resume (PDF only)", type="pdf")
job_description = st.text_area("💼 Paste the Job Description here")

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    score = calculate_similarity(resume_text, job_description)

    # Keyword comparison
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())

    matched_keywords = resume_words & jd_words
    missing_keywords = jd_words - resume_words

    # Display results
    st.subheader("📊 Match Score:")
    st.success(f"✅ Your resume matches **{score}%** with the job description!")

    st.subheader("✅ Matched Keywords:")
    st.write(", ".join(list(matched_keywords)[:20]) if matched_keywords else "No significant matches found.")

    st.subheader("❌ Missing Keywords:")
    st.write(", ".join(list(missing_keywords)[:20]) if missing_keywords else "No missing keywords — Great!")

    # Optional: Show full resume text (for debugging or transparency)
    with st.expander("📃 View Extracted Resume Text"):
        st.write(resume_text)

else:
    st.info("📌 Please upload a PDF resume and enter a job description to get started.")
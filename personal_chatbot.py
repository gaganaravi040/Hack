import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import fitz  # PyMuPDF

# Optional heavy imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_QA_AVAILABLE = True
except:
    HF_QA_AVAILABLE = False

try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    WATSON_AVAILABLE = True
except:
    WATSON_AVAILABLE = False

# ---- CONFIG ----
st.set_page_config(page_title="Personal Finance Chatbot", layout="wide", page_icon="💸")
APP_TITLE = "Personal Finance Chatbot"

# HuggingFace Q&A model
if HF_QA_AVAILABLE:
    hf_qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
else:
    hf_qa_model = None

# Optional Granite model (disabled by default)
USE_GRANITE = False
if HF_QA_AVAILABLE and USE_GRANITE:
    granite_model_name = "ibm-granite/granite-3.3-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(granite_model_name)
    granite_model = AutoModelForCausalLM.from_pretrained(granite_model_name, device_map="auto")
    granite_model.eval()

# Optional Watson NLU
if WATSON_AVAILABLE:
    WATSON_APIKEY = "YOUR_WATSON_APIKEY"
    WATSON_URL = "YOUR_WATSON_URL"
    authenticator = IAMAuthenticator(WATSON_APIKEY)
    watson_nlp = NaturalLanguageUnderstandingV1(version="2023-08-01", authenticator=authenticator)
    watson_nlp.set_service_url(WATSON_URL)
else:
    watson_nlp = None

# -------------------- LOGIN SYSTEM --------------------
USER_CREDENTIALS = {
    "admin": "1234",
    "rachana": "hack2025"
}

def login_page():
    st.title("🔑 Login to Personal Finance Chatbot")
    st.write("Please enter your credentials to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------------- MAIN APP ----------------------
def main():
    st.sidebar.title("Navigation")
    st.sidebar.write(f"👋 Logged in as: **{st.session_state.username}**")
    logout_button()

    user_type = st.sidebar.radio("I am a:", ("Student", "Professional"))
    page = st.sidebar.selectbox("Choose page", ["Home", "NLU Analysis", "Q&A", "Budget Summary"])

    if page == "Home":
        home_page(user_type)
    elif page == "NLU Analysis":
        nlu_analysis_page(user_type)
    elif page == "Q&A":
        qa_page(user_type)
    elif page == "Budget Summary":
        budget_summary_page(user_type)

# ---------------------- HOME ----------------------
def home_page(user_type):
    st.header("Welcome")
    st.markdown(
        "This chatbot provides personalized financial guidance, simple NLU analysis, Q&A, and budget summaries. "
        "Now with PDF upload and AI-powered question answering."
    )
    st.info(f"Current persona: **{user_type}** — answers will be tailored accordingly.")
    if st.button("Show a quick financial tip"):
        st.success(format_tip(user_type))

def format_tip(user_type):
    if user_type == "Student":
        return "Start with an emergency fund of ₹5,000–₹15,000. Automate small monthly savings."
    else:
        return "Aim for 3–6 months of living expenses in an emergency fund. Review tax-efficient investments."

# ---------------------- NLU ----------------------
def nlu_analysis_page(user_type):
    st.header("NLU Analysis")
    text = st.text_area("Enter text to analyze:")
    run_ibm = st.checkbox("Use IBM Watson NLU (if configured)")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter text.")
            return
        if run_ibm and watson_nlp:
            st.info("Calling Watson NLU...")
            result = watson_nlp.analyze(text=text, features={"entities": {}, "keywords": {}}).get_result()
            st.json(result)
        sentiments = simple_sentiment(text)
        keywords = extract_keywords_local(text)
        entities = extract_entities_local(text)
        st.subheader("Sentiment"); st.write(sentiments)
        st.subheader("Keywords"); st.write(keywords)
        st.subheader("Entities"); st.write(entities)
        st.subheader("Response"); st.markdown(personalized_nlu_response(user_type, sentiments, keywords))

def simple_sentiment(text):
    text_l = text.lower()
    pos = ["good","great","saved","save","profit","gain","positive","up","increase"]
    neg = ["debt","loss","lost","overspend","bad","negative","down","decrease","late"]
    score = sum(w in text_l for w in pos) - sum(w in text_l for w in neg)
    return {"polarity": "positive" if score>0 else "negative" if score<0 else "neutral", "score": score}

def extract_keywords_local(text, top_k=6):
    stop = set(["the","a","an","and","or","in","on","for","to","of","is","was","i","we","you"])
    tokens = [t.strip(".,!?;:\"'()[]") for t in text.lower().split() if t.strip()]
    tokens = [t for t in tokens if t not in stop and not t.isdigit()]
    freq = {t: tokens.count(t) for t in set(tokens)}
    return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]]

def extract_entities_local(text):
    ents = {}
    words = text.split()
    dates = [w for w in words if w.lower().startswith(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))]
    curr = [w for w in words if any(ch in w for ch in ['₹','$','£']) or w.endswith(('rs','rupees'))]
    perc = [w for w in words if w.endswith('%')]
    if dates: ents['dates'] = dates
    if curr: ents['currencies'] = curr
    if perc: ents['percents'] = perc
    return ents or {"note": "No strong entities found"}

def personalized_nlu_response(user_type, sentiments, keywords):
    base = f"I detect a {sentiments['polarity']} tone."
    extra = " As a student, focus on small repeatable savings." if user_type=="Student" else " As a professional, review your tax-efficient investments."
    if keywords: extra += f" Key topics: {', '.join(keywords[:5])}."
    return base + extra

# ---------------------- Q&A + PDF ----------------------
def qa_page(user_type):
    st.header("Finance Q&A (with PDF support)")

    uploaded_faq = st.file_uploader("Upload FAQ CSV", type=['csv'])
    if uploaded_faq is not None:
        kb_df = pd.read_csv(uploaded_faq)
    else:
        kb_df = pd.DataFrame([
            ("How much should I save each month?", "Save ~20% of income."),
            ("Emergency fund size?", "3–6 months of essential expenses."),
            ("How to pay off debt fast?", "Prioritize high-interest debt."),
        ], columns=['question','answer'])

    q = st.text_input("Ask a finance question:")
    use_semantic = st.checkbox("Use semantic matching", value=True)

    if st.button("Get KB Answer"):
        if use_semantic and SKLEARN_AVAILABLE:
            ans,_ = semantic_search_answer(kb_df, q)
        else:
            ans = rule_based_answer(kb_df, q)
        st.write(ans)
        st.markdown(personalized_qa_followup(user_type))

    st.markdown("---")
    st.subheader("Ask AI from a PDF")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        st.text_area("PDF Preview", pdf_text[:2000] + "...", height=200)
        pdf_q = st.text_input("Ask about the PDF content:")
        if st.button("Get PDF Answer"):
            if not HF_QA_AVAILABLE:
                st.error("Transformers not installed.")
            else:
                ans = hf_qa_model(question=pdf_q, context=pdf_text)["answer"]
                st.write("**HuggingFace Answer:**", ans)
                if USE_GRANITE:
                    inputs = tokenizer(pdf_q + "\n" + pdf_text, return_tensors="pt").to(granite_model.device)
                    outputs = granite_model.generate(**inputs, max_new_tokens=150)
                    granite_ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.write("**Granite Answer:**", granite_ans)

def rule_based_answer(kb_df, q):
    q_tokens = set(q.lower().split())
    best = (None, 0)
    for _, row in kb_df.iterrows():
        qt = set(str(row['question']).lower().split())
        score = len(q_tokens & qt)
        if score > best[1]:
            best = (row['answer'], score)
    return best[0] if best[0] else "No direct answer found."

def semantic_search_answer(kb_df, q):
    corpus = kb_df['question'].astype(str).tolist()
    vect = TfidfVectorizer(stop_words='english').fit(corpus + [q])
    mat = vect.transform(corpus + [q])
    sims = linear_kernel(mat[-1], mat[:-1]).flatten()
    idx = np.argmax(sims)
    return kb_df.iloc[idx]['answer'], float(sims[idx])

def personalized_qa_followup(user_type):
    return "Students: start small and automate savings." if user_type=="Student" else "Professionals: review tax-saving options."

def extract_text_from_pdf(file):
    text = ""
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_doc:
        text += page.get_text()
    return text

# ---------------------- BUDGET SUMMARY ----------------------
def budget_summary_page(user_type):
    st.header("Budget Summary")
    sample_csv = "date,amount,category,description\n2025-07-01,-2500,Groceries,Supermarket\n2025-07-03,-150,Transport,Metro\n2025-07-05,30000,Salary,Monthly salary\n"
    uploaded = st.file_uploader("Upload transactions CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=['date'])
    elif st.checkbox("Use sample data"):
        df = pd.read_csv(StringIO(sample_csv), parse_dates=['date'])
    else:
        st.info("Upload CSV or use sample.")
        return

    st.dataframe(df.head())
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    total_income = df[df['amount'] > 0]['amount'].sum()
    total_expense = -df[df['amount'] < 0]['amount'].sum()
    balance = total_income - total_expense
    st.metric("Income", f"₹{total_income:,.2f}")
    st.metric("Expenses", f"₹{total_expense:,.2f}")
    st.metric("Net", f"₹{balance:,.2f}")
    cat = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
    st.dataframe(cat.reset_index().rename(columns={'amount':'spent'}))

# ---------------------- RUN APP ----------------------
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
        st.stop()
    else:
        main()

import streamlit as st

# -------------------- LOGIN SYSTEM --------------------
# Demo credentials (username:password)
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
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Show login page if not logged in
if not st.session_state.logged_in:
    login_page()
    st.stop()
else:
    st.sidebar.write(f"👋 Logged in as: **{st.session_state.username}**")
    logout_button()

# -------------------- YOUR EXISTING CHATBOT CODE BELOW --------------------

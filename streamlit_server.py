
import streamlit as st
import requests

st.title("IT Support AI Assistant")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question}
            )
            answer = response.json().get("answer", "No answer returned")
        except Exception as e:
            answer = f"Error: {e}"

    st.write("### Answer:")
    st.write(answer)


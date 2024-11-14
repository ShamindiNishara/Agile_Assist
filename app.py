import streamlit as st
from core import get_agent_response  # Import the function from core.py

st.title("Agile Assist")

# Streamlit UI for user query input
query = st.text_input("Input your query here")

# Button to get response
if st.button("Get Answer") or query:
    response = get_agent_response(query)
    
    if "error" in response:
        st.write(f"An error occurred: {response['error']}")
    else:
        st.write(response["output"])
        st.markdown(
            f"<p style='color:blue;'>Overall Response Time: {response['response_time_overall']} seconds</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:blue;'>LLM Response Time: {response['response_time_llm']} seconds</p>",
            unsafe_allow_html=True
        )
else:
    st.write("Please enter a query")

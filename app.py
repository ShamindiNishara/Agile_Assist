import streamlit as st
from core import get_agent_response  # Import the function from core.py

st.title("Agile Assist")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input in a chat-style format
if prompt := st.chat_input("Ask me anything about Agile processes!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the assistant
    response_data = get_agent_response(prompt)
    
    if "error" in response_data:
        response_text = f"**Error:** {response_data['error']}"
    else:
        # Format the response using Markdown for clarity
        response_text = f"**Response:**\n\n{response_data['output']}\n\n"
        response_text += "---\n\n"
        response_text += f"**Overall Response Time:** {response_data['response_time_overall']} seconds\n"
        response_text += f"**LLM Response Time:** {response_data['response_time_llm']} seconds\n"

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Add assistant's formatted response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

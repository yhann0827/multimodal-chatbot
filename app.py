import streamlit as st
import openai
from openai import OpenAI

openai.api_key = st.secrets['OPENAI_API_KEY']
NVIDIA_API_KEY = st.secrets['NVIDIA_API_KEY']

def select_model_version(model_choice):
    if model_choice == "OpenAI":
        selected_model = st.sidebar.selectbox(
            "Choose OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "chatgpt-4o-latest", "gpt-4o-mini"],
            index=3
        )
    elif model_choice == "Llama":
        selected_model = st.sidebar.selectbox(
            "Choose Llama Model",
            ["llama-3.1-nemotron-70b-instruct", "llama-3.1-nemotron-51b-instruct","llama3-chatqa-1.5-70b", "llama3-chatqa-1.5-8b", "nemotron-mini-4b-instruct", "llama3-chatqa-1.5-8b"],
            index=0
        )
    else:
        selected_model = None

    if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model 
    return st.session_state.selected_model

def prepare_messages_for_model(model_choice, messages):
    if model_choice=="Llama":
        converted_messages = []
        last_role = None
        
        for msg in messages:
            current_role = msg['role']
            content = msg['content']
            
            if current_role == 'system':
                if not converted_messages:
                    converted_messages.append({
                        'role': 'user',
                        'content': content
                    })
                    last_role = 'user'
                continue  
                
            if last_role is None:
                converted_messages.append({
                    'role': 'user' if current_role == 'user' else 'assistant',
                    'content': content
                })
            elif last_role == 'user' and current_role == 'user':
                converted_messages[-1]['content'] += "\n\n" + content
            elif last_role == 'assistant' and current_role == 'assistant':
                converted_messages[-1]['content'] += "\n\n" + content
            else:
                converted_messages.append({
                    'role': current_role,
                    'content': content
                })
            
            last_role = converted_messages[-1]['role']

        if converted_messages and converted_messages[-1]['role'] == 'assistant':
            converted_messages.pop()
            
        return converted_messages
    return messages

def call_openai(selected_model, messages, temperature, max_tokens, top_p, frequency_penalty):
    
    response = openai.chat.completions.create(
        model=selected_model,
        messages=messages, 
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty
    )

    return response.choices[0].message.content

def call_llama(api_key, selected_model, messages, temperature, max_tokens, top_p, frequency_penalty):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    response = client.chat.completions.create( 
        model=f"nvidia/{selected_model}",
        messages=messages, 
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        stream=True
    )

    result = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content

    return result

def get_system_message(model_choice, selected_model, language, temperature, max_tokens, top_p, frequency_penalty, prompt_template):
    model_name = {
        "OpenAI": selected_model,
        "Llama": selected_model,
    }.get(model_choice, model_choice)
    
    system_message = (
        f"As an AI assistant using {model_name}, communicate exclusively in {language}. "
        f"Operating with temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, "
        f"and frequency_penalty={frequency_penalty}. "
        f"Introduce yourself to the user and let them know which model you are. {prompt_template}"
    )
    return system_message

def main():
    # Configure streamlit page settings
    st.set_page_config(
        page_title="MultiModal Chatbot",
        page_icon="💬",
        layout="centered"
    )
    st.title('🤖 MultiModal Chatbot')
    st.sidebar.title('Choose a Model')

    model_choice = st.sidebar.radio(
        "Select Model:",
        ('OpenAI', 'Llama'),
        index=0
    )

    # Model Parameters (User Input)
    st.sidebar.markdown("### ⚙️Model Parameters")
    temperature = st.sidebar.number_input("Temperature (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    max_tokens = st.sidebar.number_input("Max Tokens (50 - 4096)", min_value=50, max_value=4096, value=500, step=10)
    top_p = st.sidebar.number_input("Top-p (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.sidebar.number_input("Frequency Penalty (-2.0 to 2.0)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # prompt template
    st.sidebar.markdown("### 📝Prompt Template")
    language = st.sidebar.selectbox("Select your language", ("English", "Simplified Chinese", "Traditional Chinese", "Japanese", "Korean", "Tamil"))
    prompt_template = st.sidebar.text_area("Prompt Template", "You are a helpful AI assistant.")
    
    # initialise chat session if it not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    
    # display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    user_prompt = st.chat_input('Ask something...')

    selected_model = select_model_version(model_choice)
    st.write(selected_model)

    if user_prompt:

        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({'role':'user', 'content': user_prompt})

        system_message = get_system_message(
            model_choice,
            selected_model,
            language,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            prompt_template
        )
        messages = [{'role': 'system', 'content': system_message}, 
        *st.session_state.chat_history]

        processed_messages = prepare_messages_for_model(model_choice, messages)
        if model_choice=="OpenAI":
            assistant_response = call_openai(
                selected_model,
                messages,
                temperature,
                max_tokens,
                top_p,
                frequency_penalty
            )
        elif model_choice=="Llama":
            assistant_response = call_llama(
                NVIDIA_API_KEY,
                selected_model,
                processed_messages,
                temperature,
                max_tokens,
                top_p,
                frequency_penalty
            )
        st.session_state.chat_history.append({'role':'assistant', 'content': assistant_response})

        with st.chat_message('assistant'):
            st.markdown(assistant_response)

if __name__ == "__main__":
    main()
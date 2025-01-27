from imports import *

st.set_page_config(
    page_title="GenOnet - Generative Open xG Network Simulation",
    page_icon=".streamlit/img/logo-Icon.png", layout="wide", initial_sidebar_state="auto")

# Streamlit Title
st.title("A Generative Open xG Network Simulation")


# Display the logo at the top middle of the sidebar
logo = Image.open(".streamlit/img/logo.png")
st.sidebar.image(logo, use_column_width=True)
st.sidebar.markdown("---")
# Custom CSS to set font size for various elements
st.markdown(
    """
    <style>
    body, p, div, span, a, li, button {
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)











# Define your avatar path and message
avatar_path = ".streamlit/img/logo-Icon.png"
user_path = ".streamlit/img/user.png"
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get the base64 encoded image
avatar_base64 = get_image_as_base64(avatar_path)
user_base64 = get_image_as_base64(user_path)





# Load custom CSS
#with open(".streamlit/styles.css") as f:
#    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env if it exists
load_dotenv(find_dotenv())


def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except openai.AuthenticationError:
        return False

# Sidebar for API key input
#st.sidebar.title("Configuration")



OPENAI_API_KEY = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

if OPENAI_API_KEY and check_openai_api_key(OPENAI_API_KEY):
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logger.info("OpenAI API key is valid.")
else:
    OPENAI_API_KEY = None
    st.sidebar.error("Please enter a valid OpenAI API key.")

openai.api_key = OPENAI_API_KEY 
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

if not openai.api_key or not pinecone_api_key or not pinecone_environment:
    logger.error("API keys or environment variables are not set correctly.")
    st.error("Please provide a valid OpenAI API key and ensure all environment variables are set.")
else:
    logger.info("API keys and environment variables loaded successfully.")

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        logger.info("Pinecone initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")

    # Set your index name
    index_name = 'ap-v0'

    # Check if the index exists, if not create it
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name, 
                dimension=1536, 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.environ.get("PINECONE_ENVIRONMENT")  # or your specific region, e.g., 'us-west-2'
                )
            )
            logger.info(f"Index '{index_name}' created.")
        else:
            logger.info(f"Index '{index_name}' already exists.")
    except Exception as e:
        logger.error(f"Error creating index: {e}")

    # Set up LLM and embedding models
    llm_model = "gpt-4o-mini"
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Set up Pinecone vector store
    vector_store = PineconeVectorStore(embedding=embed_model, index_name=index_name)

    # Verify indexing
    try:
        index = pinecone.Index(index_name)
        index_stats = index.describe_index_stats()
        logger.info(f"Index status: {index_stats}")
    except Exception as e:
        logger.error(f"Error retrieving index stats: {e}")

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name=llm_model),
        retriever=vector_store.as_retriever(search_kwargs={'k': 12}),
        return_source_documents=True,
        verbose=False
    )

    # Initialize C++ code execution tools
    cpp_tools = [CppSubprocessTool(allow_dangerous_code=True)]
    cpp_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful coding assistant. Make sure to use the CppSubprocessTool tool for code execution.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    cpp_agent = create_tool_calling_agent(ChatOpenAI(model_name=llm_model), cpp_tools, cpp_prompt)
    cpp_agent_executor = AgentExecutor(agent=cpp_agent, tools=cpp_tools, verbose=True)
    
    # Langchain agent for Python queries
    tools = [PythonREPLTool()]

    instructions = """You are an agent designed to write and execute python code to answer questions.
                    You have access to a python REPL, which you can use to execute python code.
                    If you get an error, debug your code and try again.
                    Only use the output of your code to answer the question. 
                    You might know the answer without running any code, but you should still run the code to get the answer.
                    A detailed interpretation of the output. 
                    Explaining what the result means in the context of the question after executation.
                    Include all exact and raw output completely without summarizing after execuation.
                    Formatted the response as follows:

                    Code:
                    <code>

                    Output:
                    <output>

                    Interpretation:
                    <interpretation>
                    """
    instruction_execution = """You are an agent designed to execute code and display the output as it would appear in a Linux terminal.
                    Ensure to provide the raw and exact output as shown in a Linux terminal.
                    After execution, interpret and explain what the result means.  
                    Formatted the response as follows:

                    Output
                    <output>

                    Interpretation:
                    <interpretation>
                    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt_python = base_prompt.partial(instructions=instructions)
    prompt_exec = base_prompt.partial(instructions=instruction_execution)


    # Toggle switch to enable/disable model selection
    toggle = st.sidebar.toggle("Active gpt-4o-mini model")
    

    if toggle:
        st.sidebar.markdown("OxGPT is optimized so I am very cheap, don't worry! :rocket:")
        python_agent = create_openai_functions_agent(ChatOpenAI(model_name=llm_model, temperature=0), tools, prompt_python)
        exec_agent = create_openai_functions_agent(ChatOpenAI(model_name=llm_model, temperature=0), tools, prompt_exec)
    else:
        python_agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt_python)
        exec_agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt_exec)
    st.sidebar.markdown("---")
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)
    ns3_agent_executor = AgentExecutor(agent=exec_agent, tools=tools, verbose=True)

    # Apply the custom CSS classes
    
    #st.markdown(
    #    """
    #    <div class="box">
    #        <p class="title">OxGPT 🤖</p>
    #    </div>
    #    """, unsafe_allow_html=True
    #)
    #st.markdown('<p class="header">I am a Senior Software Network Engineer at Open xG, specializing in software networks, architecture, standards, and ns-3 simulation. I can simultaneously generate, debug, interpret, and execute software code. </p>', unsafe_allow_html=True)


    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Ensure the initial bot message is always at the top
    initial_bot_message = "Hello! How can I assist you with GenOnet today?"
    if not st.session_state['generated'] and not st.session_state['past']:
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        st.session_state.generated.append(initial_bot_message)

    # Function to handle queries
    def handle_query(user_input, option):
        try:
            with st.spinner("Generating response..."):
                if option == "ns-3 C++ Generation":
                    # Generate or Execute C++ code
                    logger.info(f"Handling C++ query: {user_input}")
                    result = cpp_agent_executor.invoke({"input": user_input})
                    answer = result.get('output', "Failed to handle C++ query.")
                    logger.info(f"C++ Query Result: {answer}")
                elif option == "ns-3 Python Generation":
                    # Handle Python queries using Langchain agent
                    logger.info(f"Handling Python query: {user_input}")
                    result = python_agent_executor.invoke({"input": user_input})
                    answer = result.get('output', "Failed to handle Python query.")
                    code_generated = result.get('generated_code', '')
                    if code_generated:
                        answer += f"\n\nGenerated Code:\n```python\n{code_generated}\n```"
                    logger.info(f"Python Query Result: {answer}")
                elif option == "ns3 Execution & Debugging & Interpretation":
                    # Handle ns3 queries
                    logger.info(f"Handling ns3 query: {user_input}")
                    result = ns3_agent_executor.invoke({"input": run_ns3_simulation(user_input)})
                    answer = result.get('output', "Failed to handle ns3 query.")
                    logger.info(f"ns3 Query Result: {answer}")
                else:
                    # Regular query handling
                    result = qa_chain.invoke({'question': user_input, 'chat_history': st.session_state['past']})
                    answer = result.get('answer', "Failed to get answer from QA Chain.")
                    logger.info(f"QA Chain Answer: {answer}")

                st.session_state.past.append((user_input, answer))
                st.session_state.generated.append(answer)

        except Exception as e:
            logger.error(f"Error during QA process: {e}")

    # Sidebar toggle buttons for query type selection
    query_type = st.sidebar.radio("Select the type of query:", ["Regular", "ns-3 C++ Generation", "ns-3 Python Generation", "ns3 Execution & Debugging & Interpretation"])

    def render_message(message, is_user=False):
        avatar = user_base64 if is_user else avatar_base64
        background_color = '#5681d0' if is_user else '#1a1a3d'
        margin = 'margin-right: auto;' if is_user else 'margin-left: auto;'
        max_width = '96%'  # Adjust this value as needed

        # Display the avatar and message in a container
        st.markdown(
            f"""
            <div style="display: flex; {margin}; margin-bottom: 20px; align-items: flex-start;">
                <img src="data:image/png;base64,{avatar}" style="width: 40px; height: 40px; border-radius: 50%; margin: 5px;">
                <div style="max-width: {max_width}; padding: 10px; border-radius: 10px; background-color: {background_color}; word-wrap: break-word; margin-left: 10px;">
                {message}
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Display the initial bot message at the top
    if st.session_state['generated']:
        render_message(st.session_state['generated'][0])

    # Display past messages with avatars
    if st.session_state['generated']:
        for i in range(1, len(st.session_state['generated'])):
            if i <= len(st.session_state['past']):
                render_message(st.session_state['past'][i-1][0], is_user=True)
            render_message(st.session_state['generated'][i])

    # Add the chat input box at the bottom
    user_input = st.chat_input("Ask your question...")
    if user_input:
        handle_query(user_input, query_type)
        st.rerun()  # Force a rerun to update the UI in real-time

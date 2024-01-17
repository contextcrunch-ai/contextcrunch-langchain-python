# ContextCrunch-LangChain-python
Integration for [ContextCrunch](https://contextcrunch.com) in a LangChain pipeline. 


## Quickstart
1. Install this package with `pip install contextcrunch-langchain`.
2. Add your [ContextCrunch](https://contextcrunch.com) API key in your environment file, such as `CONTEXTCRUNCH_API_KEY="aaa-bbb-ccc-ddd"`

### RAG
You can easily modify an existing RAG pipeline by simply applying a `ContextCruncher()` to the context before filling the prompt template.

For example, if you are using [this example](https://python.langchain.com/docs/use_cases/question_answering/quickstart#preview) from the LangChain docs, the modified pipeline becomes:
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ContextCruncher(compression_ratio=0.95)
    | prompt
    | llm
    | StrOutputParser()
)
```

### Conversations
You can use `ConversationCruncher()` to compress a long message history.

Here is an example using ConversationBufferMemory, which is a LangChain memory module that stores the entire conversation history.
```python
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Conversation Summary:\n{history}"),
        ("human", "{input}"),
    ]
)
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("My favourite color is purple, my favourite food is pizza.")
memory.chat_memory.add_ai_message("I understand. Your favourite color is purple, and your favourite food is pizza.")

chain = (
    {'history': RunnableLambda(memory.load_memory_variables) | itemgetter("history"), 'input': RunnablePassthrough()} # Fetch the history, feed the input to the next step
    | ConversationCruncher() # history and input is compressed, and fed into the prompt template (which takes 'history' and 'input' as inputs).
    | prompt
    | model
)
chain.invoke("What is favourite color?") # small contexts won't get compressed, so ConversationCruncher() will act as a passthrough.
```

## Usage

### ContextCruncher (RAG)

The `ContextCruncher` is a [Runnable Lambda](https://python.langchain.com/docs/expression_language/how_to/functions) that takes in 2 inputs (as an input dictionary):

- `context`: This is the retrieved information from RAG
- `question`: The relevant query to find in the data. ContextCrunch uses this narrow down the context to only the most essential parts.

**Return**
`ContextCruncher` returns a dictionary with:

- `context`: The updated (compressed) context.
- `question`: The original question (for later uses in a chain).

### ConversationCruncher (Chat)

The `ConversationCruncher` is a [Runnable Lambda](https://python.langchain.com/docs/expression_language/how_to/functions) that takes in 2 inputs (as an input dictionary):

- `history`: Your relevant conversation history.
- `input`: The most recent user message. ContextCrunch uses this narrow down the conversation history to the parts relevant to the input.

**Return**
`ConversationCruncher` returns a dictionary with:

- `history`: The compressed message history, as a single string. Ideally, you can feed this into a system message indicating that this is the conversation history.
- `input`: The user message, unmodified (for later uses in a chain).

### ContextCrunchDocumentCompressor (RAG w/ Documents)

As an alternative to `ContextCruncher`, if your data is already in the form of LangChain Documents, or if you would prefer to work with a larger document compression pipeline, you can use `ContextCrunchDocumentCompressor`. It takes in a list of documents, and a query string.

Here's how it would look in a typical document compression pipeline:

```python
import bs4
from contextcrunch_langchain_python import ContextCrunchDocumentCompressor
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

cc_compressor = ContextCrunchDocumentCompressor(compression_ratio=0.8)
## Make sure to initialize base retriever as you would like, for example, here is a web scraper/splitter with chromadb, from the LangChain docs
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever()

# now we wrap the base_retrievier in a compression retriever using ContextCrunch
retriever = ContextualCompressionRetriever(base_compressor=cc_compressor, base_retriever=base_retriever)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("What is Task Decomposition?")

```

### Compresion Ratio

When initializing both `ContextCruncher()`, `ConversationCruncher()`, or `ContextCrunchDocumentCompressor()` there is also an optional `compression_ratio` parameter that controls how aggresively the algorithm should compress. The general trend is the higher the compression ratio, the less information is retained. Generally, a compression ratio of 0.9 is a good start, though for small contexts, the algorithm may compress less than requested compression ratio.


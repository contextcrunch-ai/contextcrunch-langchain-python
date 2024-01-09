# ContextCrunch-LangChain-python
Integration for [ContextCrunch](https://contextcrunch.com) in a LangChain pipeline. 

## Usage
The `ContextCruncher` is a [Runnable Lambda](https://python.langchain.com/docs/expression_language/how_to/functions) that takes in 2 inputs (as an input dictionary):
- `context`: This could be your conversation history, or retrieved information from RAG
- `question`: The relevant query to find in the data. ContextCrunch uses this narrow down the context to only the most essential parts.

When initializing the `ContextCruncher()` there is also an optional `compression_ratio` parameter that controls how aggresively the algorithm should compress. The general trend is the higher the compression ratio, the less information is retained. Generally, a compression ratio of 0.9 is a good start, though for small contexts, the algorithm may only actually compress 50% of the context (even if you ask for 90%).

## Return
`ContextCruncher` returns a dictionary with:
- `context`: The updated (compressed) context.
- `question`: The original question (for later uses in a chain).

## Quickstart
Install this package with `pip install contextcrunch-langchain`.

### RAG
You can easily modify an existing RAG pipeline by simply applying a `ContextCruncher()` to the context before filling the prompt template.

For example, if you are using [this example](https://python.langchain.com/docs/use_cases/question_answering/quickstart#retrieval-and-generation-generate) from the LangChain docs, the modified pipeline becomes:
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ContextCruncher(compression_ratio=0.95)
    | prompt
    | llm
    | StrOutputParser()
)
```
```

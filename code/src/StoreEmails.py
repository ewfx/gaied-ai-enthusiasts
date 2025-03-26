from langchain.schema import Document
def store_email(email_text, vectorstore, embedding_model):
    """Stores an email in the FAISS vectorstore."""

    # Create embedding for the current email
    email_embedding = embedding_model.embed_query(email_text)

    # Add the email to the vectorstore
    # Instead of dictionary, use a Document object
    from langchain.schema import Document
    vectorstore.add_documents([Document(page_content=email_text, metadata={'embedding': email_embedding, 'source': email_text})])
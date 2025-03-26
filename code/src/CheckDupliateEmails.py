def check_duplicate(email_text, vectorstore, embedding_model, threshold):

    # Create embedding for the current email
    email_embedding = embedding_model.embed_query(email_text)
    print("Current email embedding:", email_embedding) # Print for debugging

    # Search for similar emails in FAISS
    # Use similarity_search_with_score to get scores
    similar_emails_with_scores = vectorstore.similarity_search_with_score(email_text, k=5)
    print("Similar emails:", similar_emails_with_scores) # Print for debugging

    # Check for duplicates based on similarity score
    # Access the score from the tuple
    if similar_emails_with_scores and len(similar_emails_with_scores) > 1 and similar_emails_with_scores[1][1] > threshold:
        # Access metadata and score appropriately
        # Check if 'source' key exists before accessing it
        if 'source' in similar_emails_with_scores[1][0].metadata:
            print("Found duplicate with score:", similar_emails_with_scores[1][1]) # Print for debugging
            return True, similar_emails_with_scores[1][0].metadata['source'], similar_emails_with_scores[1][1]
        else:
            print("Found potential duplicate but missing 'source' metadata, score:", similar_emails_with_scores[1][1]) # Print for debugging
            return True, None, similar_emails_with_scores[1][1] # Handle case where 'source' key is missing and return score
    else:
        # If not a duplicate, return score which will be below threshold
        print("No duplicate found, score:", similar_emails_with_scores[1][1] if similar_emails_with_scores and len(similar_emails_with_scores) > 1 else 0.0) # Print for debugging
        return False, None, similar_emails_with_scores[1][1] if similar_emails_with_scores and len(similar_emails_with_scores) > 1 else 0.0
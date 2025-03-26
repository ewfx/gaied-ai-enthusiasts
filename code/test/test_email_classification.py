import pytest
from unittest.mock import mock_open, patch, MagicMock
import os
import sys
import json
from eml_parser import EmlParser

# Add the directory containing the module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Create mock for Ollama response with streaming support
mock_ollama_response = {
    "model": "llama2",
    "created_at": "2024-03-26T12:00:00.000Z",
    "response": json.dumps({
        "request_type": "test",
        "sub_request_type": "test"
    }),
    "done": True
}

# Create proper streaming response format
mock_stream_response = {
    "model": "llama2",
    "created_at": "2024-03-26T12:00:00.000Z",
    "response": "Test response content",
    "done": True
}

@pytest.fixture(autouse=True)
def mock_ollama_and_requests():
    """Mock both Ollama LLM and its HTTP requests"""
    with patch('langchain.llms.ollama.Ollama') as mock_ollama, \
         patch('requests.post') as mock_post, \
         patch('requests.Session') as mock_session:
        
        # Mock HTTP responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [json.dumps({
            "response": "Test response",
            "done": True
        }).encode()]
        mock_post.return_value = mock_response
        mock_session.return_value.post.return_value = mock_response

        # Mock Ollama instance
        mock_ollama_instance = MagicMock()
        mock_ollama_instance._create_stream.return_value = iter([{
            "response": "Test response",
            "done": True
        }])
        mock_ollama_instance._stream_with_aggregation.return_value = "Test response"
        mock_ollama_instance.invoke.return_value = "Test response"
        mock_ollama.return_value = mock_ollama_instance

        yield mock_ollama_instance

# Now import the module
from emailclassification_surendra import (
    extract_text_from_pdf, 
    extract_text_from_eml
)

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies for tests"""
    with patch('langchain.embeddings.HuggingFaceEmbeddings') as mock_embeddings, \
         patch('sentence_transformers.SentenceTransformer') as mock_transformer:
        
        # Configure mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock transformer
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value = [[1.0] * 768]
        
        yield

# Test cases for PDF extraction
def test_extract_text_from_pdf():
    # Mock PyPDFLoader and its behavior
    mock_document = MagicMock()
    mock_document.page_content = "Test PDF content page 1\n"
    
    with patch('emailclassification_surendra.PyPDFLoader') as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_document]
        mock_loader.return_value = mock_loader_instance
        
        # Test with single PDF
        result = extract_text_from_pdf(['test.pdf'])
        assert result == "Test PDF content page 1\n"
        mock_loader.assert_called_once_with('test.pdf')

def test_extract_text_from_pdf_multiple_files():
    # Mock PyPDFLoader for multiple files
    mock_document1 = MagicMock()
    mock_document1.page_content = "PDF 1 content\n"
    mock_document2 = MagicMock()
    mock_document2.page_content = "PDF 2 content\n"
    
    with patch('emailclassification_surendra.PyPDFLoader') as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.side_effect = [[mock_document1], [mock_document2]]
        mock_loader.return_value = mock_loader_instance
        
        result = extract_text_from_pdf(['test1.pdf', 'test2.pdf'])
        assert result == "PDF 1 content\nPDF 2 content\n"
        assert mock_loader.call_count == 2

# Test cases for EML extraction
def test_extract_text_from_eml_plain():
    mock_eml_content = b"Test email content"
    mock_parsed_eml = {'body': ['Plain text email content']}
    
    with patch('builtins.open', mock_open(read_data=mock_eml_content)):
        with patch.object(EmlParser, 'decode_email_bytes', return_value=mock_parsed_eml) as mock_decode:
            result = extract_text_from_eml('test.eml')
            assert result == 'Plain text email content'

def test_extract_text_from_eml_html():
    mock_eml_content = b"Test email content"
    mock_parsed_eml = {
        'body': [''],
        'html': ['<html><body><p>HTML email content</p></body></html>']
    }
    
    with patch('builtins.open', mock_open(read_data=mock_eml_content)):
        with patch.object(EmlParser, 'decode_email_bytes', return_value=mock_parsed_eml):
            result = extract_text_from_eml('test.eml')
            assert result.strip() == 'HTML email content'

def test_extract_text_from_eml_empty():
    mock_eml_content = b""
    mock_parsed_eml = {'body': ['']}
    
    with patch('builtins.open', mock_open(read_data=mock_eml_content)):
        with patch.object(EmlParser, 'decode_email_bytes', return_value=mock_parsed_eml):
            result = extract_text_from_eml('test.eml')
            assert result == ''


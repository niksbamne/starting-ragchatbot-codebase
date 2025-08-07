"""
Shared test fixtures and configuration for the RAG system tests
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# Import necessary components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore
from session_manager import SessionManager

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
    config.ANTHROPIC_API_KEY = "test_key"
    return config

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search_course_catalog.return_value = []
    mock_store.search_course_content.return_value = []
    mock_store.add_courses.return_value = None
    mock_store.get_course_count.return_value = 0
    mock_store.get_all_course_titles.return_value = []
    return mock_store

@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock(spec=SessionManager)
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = []
    mock_manager.add_exchange.return_value = None
    mock_manager.clear_session.return_value = None
    return mock_manager

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_rag_system(mock_vector_store, mock_session_manager):
    """Mock RAG system with dependencies"""
    mock_rag = Mock(spec=RAGSystem)
    mock_rag.vector_store = mock_vector_store
    mock_rag.session_manager = mock_session_manager
    mock_rag.query.return_value = ("Test answer", ["Test source"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    return mock_rag

@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting"""
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict
    from fastapi import HTTPException
    
    app = FastAPI(title="Test RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define the same models as in the main app
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Optional[str]]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class SessionCleanupRequest(BaseModel):
        session_id: str

    class SessionCleanupResponse(BaseModel):
        success: bool
        message: str
    
    # Mock RAG system for the test app
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = ("Test answer", ["Test source"])
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test_session_123"
    mock_rag_system.session_manager.clear_session.return_value = None
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/cleanup", response_model=SessionCleanupResponse)
    async def cleanup_session(request: SessionCleanupRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return SessionCleanupResponse(
                success=True,
                message=f"Session {request.session_id} cleaned up successfully"
            )
        except Exception as e:
            return SessionCleanupResponse(
                success=False,
                message=f"Error cleaning up session: {str(e)}"
            )

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)

@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test_session_123"
    }

@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return [
        {
            "title": "Introduction to Machine Learning",
            "instructor": "Dr. Smith",
            "course_link": "http://example.com/ml",
            "lessons": [
                {
                    "lesson_number": 1,
                    "title": "What is Machine Learning?",
                    "lesson_link": "http://example.com/ml/lesson1",
                    "content": "Machine learning is a subset of artificial intelligence..."
                },
                {
                    "lesson_number": 2,
                    "title": "Types of Machine Learning",
                    "lesson_link": "http://example.com/ml/lesson2",
                    "content": "There are three main types of machine learning..."
                }
            ]
        }
    ]

@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test_key_123',
        'CHROMA_PATH': './test_chroma_db'
    }):
        yield
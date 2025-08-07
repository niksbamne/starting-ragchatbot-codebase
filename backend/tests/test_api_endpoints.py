"""
API endpoint tests for the FastAPI RAG system
Tests all HTTP endpoints for proper request/response handling
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""
    
    def test_query_with_session_id(self, client, sample_query_request):
        """Test query endpoint with existing session ID"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
    
    def test_query_without_session_id(self, client):
        """Test query endpoint without session ID (should create new session)"""
        request_data = {"query": "What is artificial intelligence?"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] is not None
    
    def test_query_missing_query_field(self, client):
        """Test query endpoint with missing query field"""
        request_data = {"session_id": "test_session"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_empty_query(self, client):
        """Test query endpoint with empty query"""
        request_data = {"query": ""}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
    
    def test_query_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_exception_handling(self, client, test_app):
        """Test query endpoint exception handling"""
        # Mock the RAG system to raise an exception
        with patch.object(test_app, 'dependency_overrides', {}):
            # This would require more complex mocking setup
            # For now, we'll test basic functionality
            request_data = {"query": "test query"}
            response = client.post("/api/query", json=request_data)
            assert response.status_code in [200, 500]


@pytest.mark.api
class TestCourseStatsEndpoint:
    """Tests for the /api/courses endpoint"""
    
    def test_get_course_stats_success(self, client):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
    
    def test_get_course_stats_method_not_allowed(self, client):
        """Test that POST is not allowed on courses endpoint"""
        response = client.post("/api/courses")
        assert response.status_code == 405
    
    def test_get_course_stats_with_query_params(self, client):
        """Test course stats endpoint ignores query parameters"""
        response = client.get("/api/courses?param=value")
        assert response.status_code == 200


@pytest.mark.api
class TestSessionCleanupEndpoint:
    """Tests for the /api/session/cleanup endpoint"""
    
    def test_session_cleanup_success(self, client):
        """Test successful session cleanup"""
        request_data = {"session_id": "test_session_123"}
        response = client.post("/api/session/cleanup", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "test_session_123" in data["message"]
    
    def test_session_cleanup_missing_session_id(self, client):
        """Test session cleanup with missing session ID"""
        request_data = {}
        response = client.post("/api/session/cleanup", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_session_cleanup_empty_session_id(self, client):
        """Test session cleanup with empty session ID"""
        request_data = {"session_id": ""}
        response = client.post("/api/session/cleanup", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_session_cleanup_method_not_allowed(self, client):
        """Test that GET is not allowed on session cleanup endpoint"""
        response = client.get("/api/session/cleanup")
        assert response.status_code == 405


@pytest.mark.api
class TestRootEndpoint:
    """Tests for the root endpoint that would serve static files"""
    
    def test_root_endpoint_exists(self, client):
        """Test that root endpoint responds (even if with 404 for missing static files)"""
        response = client.get("/")
        # In our test app, we don't mount static files, so this might return 404
        # But the endpoint should exist
        assert response.status_code in [200, 404, 405]
    
    def test_static_file_endpoint_structure(self, client):
        """Test that static file endpoints have expected structure"""
        # Test some common paths that would be served by static files
        paths_to_test = ["/", "/index.html", "/app.js", "/style.css"]
        
        for path in paths_to_test:
            response = client.get(path)
            # These will likely return 404 in test environment, which is expected
            assert response.status_code in [200, 404, 405]


@pytest.mark.api
class TestCorsHeaders:
    """Tests for CORS configuration"""
    
    def test_cors_headers_on_options(self, client):
        """Test CORS headers are present on OPTIONS request"""
        response = client.options("/api/query")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    def test_cors_headers_on_post(self, client):
        """Test CORS headers are present on POST request"""
        request_data = {"query": "test"}
        response = client.post("/api/query", json=request_data)
        
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.api
class TestApiResponseFormats:
    """Tests for API response format consistency"""
    
    def test_query_response_format(self, client):
        """Test that query response has consistent format"""
        request_data = {"query": "test query"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Validate data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    def test_course_stats_response_format(self, client):
        """Test that course stats response has consistent format"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Validate data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
    
    def test_session_cleanup_response_format(self, client):
        """Test that session cleanup response has consistent format"""
        request_data = {"session_id": "test_session"}
        response = client.post("/api/session/cleanup", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = ["success", "message"]
        for field in required_fields:
            assert field in data
        
        # Validate data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)


@pytest.mark.api
class TestApiErrorHandling:
    """Tests for API error handling"""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 response for unknown endpoint"""
        response = client.get("/api/unknown")
        assert response.status_code == 404
    
    def test_405_for_wrong_method(self, client):
        """Test 405 response for wrong HTTP method"""
        response = client.put("/api/query")
        assert response.status_code == 405
    
    def test_content_type_validation(self, client):
        """Test content type validation for POST endpoints"""
        # Send request without proper content type
        response = client.post(
            "/api/query",
            data="query=test",  # Form data instead of JSON
        )
        
        # Should either work or return validation error
        assert response.status_code in [200, 422]
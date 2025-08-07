import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults
from models import CourseChunk


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute() method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_with_basic_query(self):
        """Test execute method with a basic query"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is test content from the course"],
            metadata=[{
                'course_title': 'Test Course',
                'lesson_number': 1
            }],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("test query")
        
        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        self.assertIn("[Test Course - Lesson 1]", result)
        self.assertIn("This is test content from the course", result)
    
    def test_execute_with_course_filter(self):
        """Test execute method with course name filter"""
        mock_results = SearchResults(
            documents=["Content specific to MCP course"],
            metadata=[{
                'course_title': 'MCP Introduction',
                'lesson_number': 2
            }],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("tools", course_name="MCP")
        
        # Verify correct parameters passed
        self.mock_vector_store.search.assert_called_once_with(
            query="tools",
            course_name="MCP",
            lesson_number=None
        )
        
        self.assertIn("[MCP Introduction - Lesson 2]", result)
    
    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{
                'course_title': 'Advanced Course',
                'lesson_number': 3
            }],
            distances=[0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("advanced topics", lesson_number=3)
        
        self.mock_vector_store.search.assert_called_once_with(
            query="advanced topics",
            course_name=None,
            lesson_number=3
        )
        
        self.assertIn("[Advanced Course - Lesson 3]", result)
    
    def test_execute_with_both_filters(self):
        """Test execute method with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{
                'course_title': 'Python Basics',
                'lesson_number': 5
            }],
            distances=[0.15],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("variables", course_name="Python", lesson_number=5)
        
        self.mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python",
            lesson_number=5
        )
    
    def test_execute_with_search_error(self):
        """Test execute method handling search errors"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        self.assertEqual(result, "Database connection failed")
    
    def test_execute_with_empty_results(self):
        """Test execute method with no search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        self.assertEqual(result, "No relevant content found.")
    
    def test_execute_with_empty_results_course_filter(self):
        """Test execute method with no results and course filter"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent", course_name="Missing Course")
        
        self.assertEqual(result, "No relevant content found in course 'Missing Course'.")
    
    def test_execute_with_empty_results_lesson_filter(self):
        """Test execute method with no results and lesson filter"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent", lesson_number=99)
        
        self.assertEqual(result, "No relevant content found in lesson 99.")
    
    def test_execute_with_multiple_results(self):
        """Test execute method with multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First result content",
                "Second result content"
            ],
            metadata=[
                {'course_title': 'Course A', 'lesson_number': 1},
                {'course_title': 'Course B', 'lesson_number': 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Mock get_lesson_link calls
        self.mock_vector_store.get_lesson_link.side_effect = [
            "http://example.com/lesson1",
            "http://example.com/lesson2"
        ]
        
        result = self.search_tool.execute("test")
        
        # Check that both results are formatted
        self.assertIn("[Course A - Lesson 1]", result)
        self.assertIn("First result content", result)
        self.assertIn("[Course B - Lesson 2]", result)
        self.assertIn("Second result content", result)
        
        # Verify lesson link calls
        self.mock_vector_store.get_lesson_link.assert_any_call('Course A', 1)
        self.mock_vector_store.get_lesson_link.assert_any_call('Course B', 2)
    
    def test_execute_sources_tracking(self):
        """Test that sources are correctly tracked"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{'course_title': 'Test Course', 'lesson_number': 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson"
        
        self.search_tool.execute("test")
        
        # Check that sources were stored
        self.assertEqual(len(self.search_tool.last_sources), 1)
        source = self.search_tool.last_sources[0]
        self.assertEqual(source['text'], 'Test Course - Lesson 1')
        self.assertEqual(source['link'], 'http://example.com/lesson')
    
    def test_execute_with_missing_metadata(self):
        """Test execute method handling missing metadata"""
        mock_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test")
        
        # Should handle missing metadata gracefully
        self.assertIn("[unknown]", result)
        self.assertIn("Content with missing metadata", result)


class TestCourseSearchToolIntegration(unittest.TestCase):
    """Integration tests for CourseSearchTool with real vector store behavior"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create a real VectorStore instance but mock ChromaDB
        with patch('chromadb.PersistentClient'), \
             patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            self.vector_store = VectorStore("./test_db", "all-MiniLM-L6-v2", 5)
            self.search_tool = CourseSearchTool(self.vector_store)
    
    @patch.object(VectorStore, 'search')
    def test_integration_real_vector_store_call(self, mock_search):
        """Test integration with real vector store method signatures"""
        # Test that the actual vector store search method is called correctly
        mock_search.return_value = SearchResults(
            documents=["Integration test content"],
            metadata=[{'course_title': 'Integration Course', 'lesson_number': 1}],
            distances=[0.1],
            error=None
        )
        
        result = self.search_tool.execute("integration test", course_name="Test Course")
        
        # Verify the actual method signature matches
        mock_search.assert_called_once_with(
            query="integration test",
            course_name="Test Course", 
            lesson_number=None
        )
        
        self.assertIn("Integration test content", result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
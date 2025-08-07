import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import VectorStore
from session_manager import SessionManager


class TestRAGSystemQuery(unittest.TestCase):
    """Test cases for RAG system query handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_db"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 2
        
        # Mock all the dependencies
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager') as mock_session_manager_class, \
             patch('rag_system.ToolManager') as mock_tool_manager_class, \
             patch('rag_system.CourseSearchTool') as mock_search_tool_class, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool_class:
            
            # Create mock instances
            self.mock_vector_store = Mock(spec=VectorStore)
            self.mock_ai_generator = Mock(spec=AIGenerator)
            self.mock_session_manager = Mock(spec=SessionManager)
            self.mock_tool_manager = Mock(spec=ToolManager)
            self.mock_search_tool = Mock(spec=CourseSearchTool)
            
            # Configure mock classes to return mock instances
            mock_vector_store_class.return_value = self.mock_vector_store
            mock_ai_generator_class.return_value = self.mock_ai_generator
            mock_session_manager_class.return_value = self.mock_session_manager
            mock_tool_manager_class.return_value = self.mock_tool_manager
            mock_search_tool_class.return_value = self.mock_search_tool
            
            # Create RAG system
            self.rag_system = RAGSystem(self.mock_config)
    
    def test_query_basic_functionality(self):
        """Test basic query processing"""
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "This is the AI response"
        
        # Mock tool manager sources
        self.mock_tool_manager.get_last_sources.return_value = [
            {"text": "Test Course - Lesson 1", "link": "http://example.com"}
        ]
        
        # Execute query
        response, sources = self.rag_system.query("What is MCP?")
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        
        # Check query was formatted correctly
        expected_query = "Answer this question about course materials: What is MCP?"
        self.assertEqual(call_args['query'], expected_query)
        
        # Check tools and tool manager were provided
        self.assertEqual(call_args['tool_manager'], self.mock_tool_manager)
        self.assertIsNotNone(call_args['tools'])
        
        # Check no conversation history was passed
        self.assertIsNone(call_args['conversation_history'])
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
        
        # Check return values
        self.assertEqual(response, "This is the AI response")
        self.assertEqual(sources, [{"text": "Test Course - Lesson 1", "link": "http://example.com"}])
    
    def test_query_with_session_id(self):
        """Test query processing with session management"""
        session_id = "test_session_123"
        
        # Mock session history
        mock_history = "User: Previous question\nAssistant: Previous answer"
        self.mock_session_manager.get_conversation_history.return_value = mock_history
        
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Response with context"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query with session
        response, sources = self.rag_system.query("Follow up question", session_id)
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify history was passed to AI generator
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['conversation_history'], mock_history)
        
        # Verify session was updated with new exchange
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "Response with context"
        )
    
    def test_query_tool_manager_integration(self):
        """Test that tool manager is properly configured"""
        # Mock AI response and sources
        self.mock_ai_generator.generate_response.return_value = "Tool-assisted response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Mock tool definitions
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search content"},
            {"name": "get_course_outline", "description": "Get outline"}
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        
        # Execute query
        response, sources = self.rag_system.query("Test tool integration")
        
        # Verify tool definitions were retrieved and passed
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['tools'], mock_tool_definitions)
    
    def test_query_error_handling(self):
        """Test query error handling"""
        # Mock AI generator to raise an exception
        self.mock_ai_generator.generate_response.side_effect = Exception("AI API error")
        
        # Query should raise the exception (not handle it internally)
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Error test")
        
        self.assertIn("AI API error", str(context.exception))
    
    def test_query_sources_handling(self):
        """Test proper sources handling"""
        # Mock different types of sources
        mock_sources = [
            {"text": "Course A - Lesson 1", "link": "http://example.com/lesson1"},
            {"text": "Course B - Lesson 2", "link": None},  # No link
            "Simple string source"  # Backward compatibility
        ]
        
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        response, sources = self.rag_system.query("Source test")
        
        # Verify sources are passed through unchanged
        self.assertEqual(sources, mock_sources)
        
        # Verify sources were reset after retrieval
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_prompt_formatting(self):
        """Test that query is properly formatted for AI"""
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        test_query = "How do I use MCP tools?"
        self.rag_system.query(test_query)
        
        # Verify query was formatted with instruction prefix
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        expected_prompt = f"Answer this question about course materials: {test_query}"
        self.assertEqual(call_args['query'], expected_prompt)
    
    def test_initialization_tool_registration(self):
        """Test that tools are properly registered during initialization"""
        # This tests the __init__ method behavior
        
        # Verify tools were registered
        self.mock_tool_manager.register_tool.assert_any_call(self.mock_search_tool)
        self.assertEqual(self.mock_tool_manager.register_tool.call_count, 2)  # search + outline tools


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for RAG system components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_config = Mock(spec=Config)
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_db"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_end_to_end_query_flow(self, mock_session_mgr_class, mock_ai_gen_class, 
                                   mock_vector_store_class, mock_doc_proc_class):
        """Test complete end-to-end query processing"""
        # Create real instances with mocked behavior
        mock_vector_store = Mock()
        mock_ai_generator = Mock()
        mock_session_manager = Mock()
        
        mock_vector_store_class.return_value = mock_vector_store
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_session_mgr_class.return_value = mock_session_manager
        
        # Create RAG system
        rag_system = RAGSystem(self.mock_config)
        
        # Mock the complete flow
        mock_session_manager.get_conversation_history.return_value = None
        mock_ai_generator.generate_response.return_value = "MCP is Model Context Protocol"
        
        # Mock tool manager behavior
        with patch.object(rag_system.tool_manager, 'get_tool_definitions') as mock_get_tools, \
             patch.object(rag_system.tool_manager, 'get_last_sources') as mock_get_sources, \
             patch.object(rag_system.tool_manager, 'reset_sources') as mock_reset_sources:
            
            mock_get_tools.return_value = [{"name": "search_course_content"}]
            mock_get_sources.return_value = [{"text": "MCP Course", "link": None}]
            
            # Execute query
            response, sources = rag_system.query("What is MCP?")
            
            # Verify the complete flow
            self.assertEqual(response, "MCP is Model Context Protocol")
            self.assertEqual(sources, [{"text": "MCP Course", "link": None}])
            
            # Verify all components were called
            mock_ai_generator.generate_response.assert_called_once()
            mock_get_sources.assert_called_once()
            mock_reset_sources.assert_called_once()


class TestRAGSystemFailureScenarios(unittest.TestCase):
    """Test various failure scenarios in RAG system"""
    
    def setUp(self):
        """Set up failure scenario test fixtures"""
        self.mock_config = Mock(spec=Config)
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_db"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_ai_generator_failure(self, mock_session_mgr_class, mock_ai_gen_class,
                                  mock_vector_store_class, mock_doc_proc_class):
        """Test behavior when AI generator fails"""
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        
        # Mock other components
        mock_vector_store_class.return_value = Mock()
        mock_session_mgr_class.return_value = Mock()
        mock_doc_proc_class.return_value = Mock()
        
        rag_system = RAGSystem(self.mock_config)
        
        # Mock AI generator failure
        mock_ai_generator.generate_response.side_effect = Exception("API key invalid")
        
        # Query should propagate the exception
        with self.assertRaises(Exception) as context:
            rag_system.query("Test query")
        
        self.assertIn("API key invalid", str(context.exception))
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tool_manager_failure(self, mock_session_mgr_class, mock_ai_gen_class,
                                  mock_vector_store_class, mock_doc_proc_class):
        """Test behavior when tool manager fails"""
        # Set up mocks
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_vector_store_class.return_value = Mock()
        mock_session_mgr_class.return_value = Mock()
        mock_doc_proc_class.return_value = Mock()
        
        rag_system = RAGSystem(self.mock_config)
        
        # Mock successful AI response but tool manager failure
        mock_ai_generator.generate_response.return_value = "AI response"
        
        with patch.object(rag_system.tool_manager, 'get_last_sources') as mock_get_sources:
            mock_get_sources.side_effect = Exception("Tool manager error")
            
            # Should propagate tool manager error
            with self.assertRaises(Exception) as context:
                rag_system.query("Test query")
            
            self.assertIn("Tool manager error", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
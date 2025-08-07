import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


class TestSequentialToolCallingIntegration(unittest.TestCase):
    """Integration tests for sequential tool calling with RAG system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_db"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 2
        
        # Mock the anthropic client to prevent real API calls
        self.anthropic_patcher = patch('ai_generator.anthropic.Anthropic')
        self.mock_anthropic_class = self.anthropic_patcher.start()
        self.mock_client = Mock()
        self.mock_anthropic_class.return_value = self.mock_client
        
    def tearDown(self):
        """Clean up patches"""
        self.anthropic_patcher.stop()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    def test_rag_system_with_sequential_tools(self, mock_session_mgr_class, 
                                              mock_vector_store_class, mock_doc_proc_class):
        """Test that RAG system supports sequential tool calling"""
        
        # Set up mock dependencies
        mock_vector_store = Mock()
        mock_session_manager = Mock()
        
        mock_vector_store_class.return_value = mock_vector_store
        mock_session_mgr_class.return_value = mock_session_manager
        mock_doc_proc_class.return_value = Mock()
        
        # Create RAG system
        rag_system = RAGSystem(self.mock_config)
        
        # Mock sequential tool calling scenario
        # Round 1: Get course outline
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"course_title": "MCP"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use_1]
        mock_response_1.stop_reason = "tool_use"
        
        # Round 2: Search course content based on outline
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "lesson 4 content", "course_name": "MCP"}
        
        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_use_2]
        mock_response_2.stop_reason = "tool_use"
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "MCP Lesson 4 covers advanced features including custom tools and protocols"
        
        # Set up API call sequence
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response
        ]
        
        # Mock tool manager behavior
        with patch.object(rag_system.tool_manager, 'get_tool_definitions') as mock_get_tools, \
             patch.object(rag_system.tool_manager, 'execute_tool') as mock_execute_tool, \
             patch.object(rag_system.tool_manager, 'get_last_sources') as mock_get_sources, \
             patch.object(rag_system.tool_manager, 'reset_sources') as mock_reset_sources:
            
            # Mock tool definitions
            mock_get_tools.return_value = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search course content"}
            ]
            
            # Mock tool executions
            mock_execute_tool.side_effect = [
                "MCP Course Outline:\nLesson 1: Introduction\nLesson 2: Basic Setup\nLesson 3: Simple Tools\nLesson 4: Advanced Features",
                "Lesson 4 content: Advanced MCP features include custom tool development, protocol extensions, and integration patterns"
            ]
            
            # Mock sources
            mock_get_sources.return_value = [
                {"text": "MCP Course - Lesson 4", "link": "http://example.com/mcp/lesson4"}
            ]
            
            # Mock session manager
            mock_session_manager.get_conversation_history.return_value = None
            
            # Execute query that should trigger sequential tool calling
            response, sources = rag_system.query("What does lesson 4 of MCP course cover?")
            
            # Verify the response
            self.assertIn("advanced features", response.lower())
            self.assertIn("custom tools", response.lower())
            
            # Verify sources were retrieved
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["text"], "MCP Course - Lesson 4")
            
            # Verify sequential tool calls were made
            self.assertEqual(mock_execute_tool.call_count, 2)
            
            # Check the sequence of tool calls
            tool_calls = mock_execute_tool.call_args_list
            self.assertEqual(tool_calls[0][0], ("get_course_outline",))
            self.assertEqual(tool_calls[1][0], ("search_course_content",))
            
            # Verify multiple API calls were made (sequential rounds)
            self.assertEqual(self.mock_client.messages.create.call_count, 3)
            
            # Verify tools were available in each round
            for call_args in self.mock_client.messages.create.call_args_list[:-1]:  # Exclude final call
                self.assertIn('tools', call_args[1])
                tools = call_args[1]['tools']
                self.assertTrue(len(tools) > 0)
    
    def test_sequential_tool_calling_preserves_max_rounds(self):
        """Test that sequential tool calling respects configured max rounds"""
        
        # Test that the default max_rounds parameter (2) is preserved
        # when passed through the RAG system to AI generator
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.SessionManager') as mock_session_mgr_class:
            
            mock_vector_store_class.return_value = Mock()
            mock_session_mgr_class.return_value = Mock()
            
            rag_system = RAGSystem(self.mock_config)
            
            # Mock a scenario that would exceed max rounds
            mock_responses = []
            for i in range(5):  # More responses than max_rounds
                mock_tool_use = Mock()
                mock_tool_use.type = "tool_use"
                mock_tool_use.name = "search_course_content"
                mock_tool_use.id = f"tool_{i}"
                mock_tool_use.input = {"query": f"search_{i}"}
                
                mock_response = Mock()
                mock_response.content = [mock_tool_use]
                mock_response.stop_reason = "tool_use"
                mock_responses.append(mock_response)
            
            # Final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = "Final response after max rounds"
            mock_responses.append(mock_final_response)
            
            self.mock_client.messages.create.side_effect = mock_responses
            
            # Mock tool manager
            with patch.object(rag_system.tool_manager, 'get_tool_definitions') as mock_get_tools, \
                 patch.object(rag_system.tool_manager, 'execute_tool') as mock_execute_tool, \
                 patch.object(rag_system.tool_manager, 'get_last_sources') as mock_get_sources, \
                 patch.object(rag_system.tool_manager, 'reset_sources'):
                
                mock_get_tools.return_value = [{"name": "search_course_content"}]
                mock_execute_tool.return_value = "Search result"
                mock_get_sources.return_value = []
                
                # Execute query
                response, sources = rag_system.query("Complex multi-step query")
                
                # Should stop after max_rounds (2) + final call = 3 API calls maximum
                self.assertLessEqual(self.mock_client.messages.create.call_count, 3)
                
                # Should execute at most 2 tool calls (max_rounds)
                self.assertLessEqual(mock_execute_tool.call_count, 2)
    
    def test_backward_compatibility_single_tool_call(self):
        """Test that single tool calls still work (backward compatibility)"""
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.SessionManager') as mock_session_mgr_class:
            
            mock_vector_store_class.return_value = Mock()
            mock_session_mgr_class.return_value = Mock()
            
            rag_system = RAGSystem(self.mock_config)
            
            # Mock single tool call scenario (like before)
            mock_tool_use = Mock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.name = "search_course_content"
            mock_tool_use.id = "tool_1"
            mock_tool_use.input = {"query": "Python basics"}
            
            mock_response_1 = Mock()
            mock_response_1.content = [mock_tool_use]
            mock_response_1.stop_reason = "tool_use"
            
            # Claude decides no more tools needed
            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = "Python is a programming language"
            mock_final_response.stop_reason = "end_turn"
            
            self.mock_client.messages.create.side_effect = [
                mock_response_1,
                mock_final_response
            ]
            
            with patch.object(rag_system.tool_manager, 'get_tool_definitions') as mock_get_tools, \
                 patch.object(rag_system.tool_manager, 'execute_tool') as mock_execute_tool, \
                 patch.object(rag_system.tool_manager, 'get_last_sources') as mock_get_sources, \
                 patch.object(rag_system.tool_manager, 'reset_sources'):
                
                mock_get_tools.return_value = [{"name": "search_course_content"}]
                mock_execute_tool.return_value = "Python content"
                mock_get_sources.return_value = [{"text": "Python Course", "link": None}]
                
                response, sources = rag_system.query("What is Python?")
                
                # Should work exactly like before
                self.assertEqual(response, "Python is a programming language")
                self.assertEqual(len(sources), 1)
                self.assertEqual(mock_execute_tool.call_count, 1)
                self.assertEqual(self.mock_client.messages.create.call_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
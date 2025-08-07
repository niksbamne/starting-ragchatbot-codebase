import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create AIGenerator with mock API key and model
        self.ai_generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Create mock tool manager and tools
        self.mock_tool_manager = Mock(spec=ToolManager)
        self.mock_search_tool = Mock(spec=CourseSearchTool)
        
        # Mock tool definitions
        self.tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "course_name": {"type": "string", "description": "Course name filter"},
                        "lesson_number": {"type": "integer", "description": "Lesson number filter"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test basic response generation without tool usage"""
        # Mock Claude API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a general knowledge answer"
        mock_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_client.return_value = mock_client_instance
        
        # Test response generation
        result = self.ai_generator.generate_response("What is Python?")
        
        # Verify API was called correctly
        mock_client_instance.messages.create.assert_called_once()
        call_args = mock_client_instance.messages.create.call_args[1]
        
        self.assertEqual(call_args['model'], "claude-sonnet-4-20250514")
        self.assertEqual(call_args['temperature'], 0)
        self.assertEqual(call_args['max_tokens'], 800)
        self.assertEqual(call_args['messages'], [{"role": "user", "content": "What is Python?"}])
        self.assertIn("You are an AI assistant", call_args['system'])
        
        # Verify no tools were used
        self.assertNotIn('tools', call_args)
        
        self.assertEqual(result, "This is a general knowledge answer")
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Answer with context"
        mock_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_client.return_value = mock_client_instance
        
        # Test with history
        history = "User: Previous question\nAssistant: Previous answer"
        result = self.ai_generator.generate_response(
            "Follow-up question", 
            conversation_history=history
        )
        
        # Verify history was included in system prompt
        call_args = mock_client_instance.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args['system'])
        self.assertIn("User: Previous question", call_args['system'])
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response with tools available but no tool use triggered"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct answer without tools"
        mock_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_client.return_value = mock_client_instance
        
        result = self.ai_generator.generate_response(
            "What is machine learning?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify tools were provided to API
        call_args = mock_client_instance.messages.create.call_args[1]
        self.assertEqual(call_args['tools'], self.tool_definitions)
        self.assertEqual(call_args['tool_choice'], {"type": "auto"})
        
        # Verify no tool execution occurred
        self.mock_tool_manager.execute_tool.assert_not_called()
        
        self.assertEqual(result, "Direct answer without tools")
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_client):
        """Test response generation when Claude decides to use tools"""
        # Mock initial response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_call_123"
        mock_tool_use.input = {"query": "MCP basics", "course_name": "MCP"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on the search results: MCP stands for..."
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_initial_response,  # Initial call with tool use
            mock_final_response     # Final call after tool execution
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "MCP (Model Context Protocol) is..."
        
        result = self.ai_generator.generate_response(
            "What is MCP?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify tool was executed
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP basics",
            course_name="MCP"
        )
        
        # Verify two API calls were made
        self.assertEqual(mock_client_instance.messages.create.call_count, 2)
        
        # Check the second call includes tool results
        second_call_args = mock_client_instance.messages.create.call_args_list[1][1]
        messages = second_call_args['messages']
        
        # Should have: original user message, assistant tool use, tool results
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[2]['role'], 'user')
        
        # Tool result should be in the last message
        tool_result = messages[2]['content'][0]
        self.assertEqual(tool_result['type'], 'tool_result')
        self.assertEqual(tool_result['tool_use_id'], 'tool_call_123')
        self.assertEqual(tool_result['content'], 'MCP (Model Context Protocol) is...')
        
        self.assertEqual(result, "Based on the search results: MCP stands for...")
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client):
        """Test handling multiple tool calls in one response"""
        # Mock multiple tool uses
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"query": "basics"}
        
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"course_title": "MCP"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use_1, mock_tool_use_2]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Combined response from multiple tools"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock multiple tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Outline result 2"
        ]
        
        result = self.ai_generator.generate_response(
            "Tell me about MCP",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Check tool execution calls
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("search_course_content",))
        self.assertEqual(calls[0][1], {"query": "basics"})
        self.assertEqual(calls[1][0], ("get_course_outline",))
        self.assertEqual(calls[1][1], {"course_title": "MCP"})
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_client):
        """Test error handling during tool execution"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_error"
        mock_tool_use.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Error was handled gracefully"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool execution error
        self.mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        result = self.ai_generator.generate_response(
            "Error test",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify error message was passed back to Claude
        second_call_args = mock_client_instance.messages.create.call_args_list[1][1]
        tool_result = second_call_args['messages'][2]['content'][0]
        self.assertEqual(tool_result['content'], 'Tool execution failed: Database error')
        
        self.assertEqual(result, "Error was handled gracefully")
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_client):
        """Test that AI can make sequential tool calls across 2 rounds"""
        # Mock first round: Claude makes initial tool call
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"course_title": "MCP"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use_1]
        mock_response_1.stop_reason = "tool_use"
        
        # Mock second round: Claude makes follow-up tool call after seeing first results
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "lesson 4 content", "course_name": "MCP"}
        
        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_use_2]
        mock_response_2.stop_reason = "tool_use"
        
        # Mock final response: Claude provides final answer after both tool calls
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on both searches: MCP Lesson 4 covers advanced topics"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_response_1,      # First round tool call
            mock_response_2,      # Second round tool call
            mock_final_response   # Final response without tools
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course outline for MCP with Lesson 4: Advanced Features",
            "Lesson 4 content: Advanced MCP features and implementation"
        ]
        
        result = self.ai_generator.generate_response(
            "What does lesson 4 of MCP course cover?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify 3 API calls were made (2 rounds + final response)
        self.assertEqual(mock_client_instance.messages.create.call_count, 3)
        
        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Check tool execution sequence
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("get_course_outline",))
        self.assertEqual(calls[1][0], ("search_course_content",))
        
        # Verify final response
        self.assertIn("Based on both searches", result)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic_client):
        """Test that sequential calling terminates when Claude doesn't need more tools"""
        # Mock first round with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "Python basics"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use]
        mock_response_1.stop_reason = "tool_use"
        
        # Mock second round: Claude provides final answer (no more tools)
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Python is a programming language used for..."
        mock_final_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_response_1,      # First round with tools
            mock_final_response   # Second round, no tools needed
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Python basics content"
        
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify only 2 API calls (1 tool round + 1 final response)
        self.assertEqual(mock_client_instance.messages.create.call_count, 2)
        
        # Verify only 1 tool was executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)
        
        self.assertEqual(result, "Python is a programming language used for...")
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_limit(self, mock_anthropic_client):
        """Test that sequential calling respects maximum round limit"""
        # Mock responses for multiple rounds, all with tool use
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"query": "first search"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use_1]
        mock_response_1.stop_reason = "tool_use"
        
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "second search"}
        
        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_use_2]
        mock_response_2.stop_reason = "tool_use"
        
        # Final response when max rounds reached
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final answer after max rounds reached"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_response_1,      # Round 1
            mock_response_2,      # Round 2 (max reached)
            mock_final_response   # Final call without tools
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "First search result",
            "Second search result"
        ]
        
        result = self.ai_generator.generate_response(
            "Complex query requiring multiple searches",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
            max_rounds=2  # Explicitly set max rounds
        )
        
        # Verify 3 API calls (2 rounds + final without tools)
        self.assertEqual(mock_client_instance.messages.create.call_count, 3)
        
        # Verify 2 tool executions (max rounds reached)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify final call was made without tools
        final_call_args = mock_client_instance.messages.create.call_args_list[2][1]
        self.assertNotIn('tools', final_call_args)
        
        self.assertEqual(result, "Final answer after max rounds reached")
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_tool_execution_error(self, mock_anthropic_client):
        """Test graceful handling of tool execution errors in sequential calls"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_error"
        mock_tool_use.input = {"query": "test"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use]
        mock_response_1.stop_reason = "tool_use"
        
        # Second round should continue despite tool error
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Handled error gracefully and provided partial answer"
        mock_final_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [
            mock_response_1,
            mock_final_response
        ]
        mock_anthropic_client.return_value = mock_client_instance
        
        # Mock tool execution error
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        result = self.ai_generator.generate_response(
            "Test error handling",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Should still complete successfully with error handling
        self.assertIn("Handled error gracefully", result)
        
        # Verify error was passed to Claude in tool result
        second_call_args = mock_client_instance.messages.create.call_args_list[1][1]
        messages = second_call_args['messages']
        
        # Find the tool result message
        tool_result_message = next(msg for msg in messages if msg['role'] == 'user' and 'content' in msg and isinstance(msg['content'], list))
        tool_result = tool_result_message['content'][0]
        
        self.assertEqual(tool_result['type'], 'tool_result')
        self.assertIn('Error executing tool', tool_result['content'])
    
    def test_system_prompt_contains_required_instructions(self):
        """Test that the system prompt contains tool usage instructions"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        # Verify key instructions are present
        self.assertIn("search_course_content", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
        self.assertIn("Tool Usage Guidelines", system_prompt)
        self.assertIn("Maximum 2 rounds of tool calls", system_prompt)
        self.assertIn("Course outline questions", system_prompt)
        self.assertIn("Course content questions", system_prompt)
    
    def test_base_params_initialization(self):
        """Test that base parameters are correctly initialized"""
        self.assertEqual(self.ai_generator.base_params['model'], "claude-sonnet-4-20250514")
        self.assertEqual(self.ai_generator.base_params['temperature'], 0)
        self.assertEqual(self.ai_generator.base_params['max_tokens'], 800)


class TestAIGeneratorToolCallDetection(unittest.TestCase):
    """Test cases for detecting when tools should be called"""
    
    def setUp(self):
        """Set up fixtures for tool call detection tests"""
        self.ai_generator = AIGenerator("test_key", "test_model")
    
    def test_system_prompt_course_content_detection(self):
        """Test system prompt guides toward content search tool"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        # Check for content search guidance
        self.assertIn("Course content questions", system_prompt)
        self.assertIn("search_course_content", system_prompt)
        
        # Check for outline search guidance  
        self.assertIn("Course outline questions", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
    
    def test_system_prompt_encourages_sequential_tools(self):
        """Test system prompt supports sequential tool calls"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        self.assertIn("Sequential tool use", system_prompt)
        self.assertIn("Maximum 2 rounds of tool calls", system_prompt)
        self.assertIn("Sequential Reasoning", system_prompt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
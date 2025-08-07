import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestSequentialToolCalling(unittest.TestCase):
    """Test cases specifically for sequential tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the anthropic client at the class level
        self.anthropic_patcher = patch('ai_generator.anthropic.Anthropic')
        self.mock_anthropic_class = self.anthropic_patcher.start()
        self.mock_client = Mock()
        self.mock_anthropic_class.return_value = self.mock_client
        
        # Create AIGenerator (will use mocked client)
        self.ai_generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Create mock tool manager
        self.mock_tool_manager = Mock(spec=ToolManager)
        
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
        
    def tearDown(self):
        """Clean up patches"""
        self.anthropic_patcher.stop()
    
    def test_sequential_tool_calling_two_rounds(self):
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
        
        # Set up the sequence of API responses
        self.mock_client.messages.create.side_effect = [
            mock_response_1,      # First round tool call
            mock_response_2,      # Second round tool call
            mock_final_response   # Final response without tools
        ]
        
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
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Check tool execution sequence
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("get_course_outline",))
        self.assertEqual(calls[1][0], ("search_course_content",))
        
        # Verify final response
        self.assertIn("Based on both searches", result)
    
    def test_sequential_tool_calling_early_termination(self):
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
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,      # First round with tools
            mock_final_response   # Second round, no tools needed
        ]
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Python basics content"
        
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify only 2 API calls (1 tool round + 1 final response)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Verify only 1 tool was executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)
        
        self.assertEqual(result, "Python is a programming language used for...")
    
    def test_sequential_tool_calling_max_rounds_limit(self):
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
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,      # Round 1
            mock_response_2,      # Round 2 (max reached)
            mock_final_response   # Final call without tools
        ]
        
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
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Verify 2 tool executions (max rounds reached)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify final call was made without tools
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn('tools', final_call_args)
        
        self.assertEqual(result, "Final answer after max rounds reached")
    
    def test_sequential_tool_calling_tool_execution_error(self):
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
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_final_response
        ]
        
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
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args['messages']
        
        # Find the tool result message
        tool_result_message = next(msg for msg in messages if msg['role'] == 'user' and 'content' in msg and isinstance(msg['content'], list))
        tool_result = tool_result_message['content'][0]
        
        self.assertEqual(tool_result['type'], 'tool_result')
        self.assertIn('Error executing tool', tool_result['content'])
    
    def test_conversation_context_preservation(self):
        """Test that conversation context is preserved across sequential tool calls"""
        # Mock single round to test context preservation
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "test"}
        
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use]
        mock_response_1.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Context preserved response"
        mock_final_response.stop_reason = "end_turn"
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_final_response
        ]
        
        self.mock_tool_manager.execute_tool.return_value = "Tool result"
        
        result = self.ai_generator.generate_response(
            "Test query",
            conversation_history="Previous: User asked about X\nAssistant: Answered about X",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Check that conversation history was included in system prompt
        first_call_args = self.mock_client.messages.create.call_args_list[0][1]
        self.assertIn("Previous conversation:", first_call_args['system'])
        self.assertIn("User asked about X", first_call_args['system'])
        
        # Check that final response includes the context
        self.assertEqual(result, "Context preserved response")
    
    def test_system_prompt_sequential_instructions(self):
        """Test that system prompt contains sequential tool calling instructions"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        # Verify sequential tool calling guidance
        self.assertIn("Sequential tool use", system_prompt)
        self.assertIn("Maximum 2 rounds of tool calls", system_prompt)
        self.assertIn("Sequential Reasoning", system_prompt)
        self.assertIn("After each tool use, evaluate if you need additional information", system_prompt)
        self.assertIn("Use follow-up searches to gather complete context", system_prompt)
        
        # Verify examples are provided
        self.assertIn("get_course_outline → search_course_content → provide complete answer", system_prompt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
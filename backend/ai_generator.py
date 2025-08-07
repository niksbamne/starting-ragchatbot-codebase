import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Available Tools:
1. **search_course_content**: For questions about specific course content or detailed educational materials
2. **get_course_outline**: For questions about course outlines, structure, lesson lists, or complete course overviews

Tool Usage Guidelines:
- **Course outline questions** (e.g., "What is the outline of...", "What lessons are in...", "Show me the structure of..."): Use get_course_outline
- **Course content questions**: Use search_course_content for specific content within courses
- **Sequential tool use**: You may use multiple tools to gather comprehensive information
  - For comparisons: Search multiple courses/lessons separately
  - For complex questions: Break down into multiple searches  
  - For comprehensive answers: Combine outline and content searches
- **Maximum 2 rounds of tool calls per user query**
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Sequential Reasoning:
- After each tool use, evaluate if you need additional information
- Use follow-up searches to gather complete context before responding
- Combine information from multiple searches for comprehensive answers
- Example flow: get_course_outline → search_course_content → provide complete answer

Course Outline Responses:
When using get_course_outline, always include:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles
- Present information exactly as returned by the tool

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline first, then answer
- **Course-specific content questions**: Use search_course_content first, then answer
- **Complex questions**: Use multiple tools as needed to gather complete information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare initial messages
        initial_messages = [{"role": "user", "content": query}]
        
        # Handle sequential tool calling if tools are available
        if tools and tool_manager:
            return self._handle_sequential_tool_calls(
                initial_messages, system_content, tools, tool_manager, max_rounds
            )
        
        # Handle regular response without tools
        api_params = {
            **self.base_params,
            "messages": initial_messages,
            "system": system_content
        }
        
        response = self.client.messages.create(**api_params)
        return response.content[0].text
    
    def _handle_sequential_tool_calls(self, initial_messages: List, system_content: str, 
                                     tools: List, tool_manager, max_rounds: int) -> str:
        """
        Handle multiple rounds of tool calling with conversation context preservation.
        
        Args:
            initial_messages: Starting conversation messages
            system_content: System prompt content
            tools: Available tools for AI
            tool_manager: Tool execution manager
            max_rounds: Maximum rounds allowed
            
        Returns:
            Final response after all tool calling rounds
        """
        messages = initial_messages.copy()
        round_count = 0
        
        while round_count < max_rounds:
            # Prepare API call with tools
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
            
            try:
                # Get Claude's response
                response = self.client.messages.create(**api_params)
                
                # Add Claude's response to conversation
                messages.append({"role": "assistant", "content": response.content})
                
                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use":
                    # Execute tools and add results
                    tool_results = self._execute_tools(response, tool_manager)
                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                        round_count += 1
                        continue
                    else:
                        # Tool execution failed, break the loop
                        break
                else:
                    # No more tool use, return final response
                    return response.content[0].text
                    
            except Exception as e:
                # Handle API errors gracefully
                return f"Error during tool calling: {str(e)}"
        
        # If we've reached max rounds, make final call without tools
        return self._make_final_call(messages, system_content)
    
    def _execute_tools(self, response, tool_manager) -> Optional[List]:
        """
        Execute all tool calls from Claude's response.
        
        Args:
            response: Claude's response containing tool calls
            tool_manager: Tool execution manager
            
        Returns:
            List of tool results or None if execution fails
        """
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Log tool execution error and continue
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}"
                    })
        
        return tool_results if tool_results else None
    
    def _make_final_call(self, messages: List, system_content: str) -> str:
        """
        Make final API call without tools to get Claude's final response.
        
        Args:
            messages: Complete conversation history
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # Intentionally no tools for final response
        }
        
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            return f"Error generating final response: {str(e)}"
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
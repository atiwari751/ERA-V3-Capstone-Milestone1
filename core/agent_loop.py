import asyncio
from module.perception import extract_perception
from module.memory import MemoryManager, MemoryItem
from module.decision import generate_plan
from module.action import execute_tool
from module.utils import log  # Import the log function

max_steps = 30

async def agent_loop(user_input: str, session, tools, tool_descriptions: str):
    memory = MemoryManager()
    session_id = f"session-{int(asyncio.get_event_loop().time())}"
    query = user_input  # Store original intent
    step = 0
    results_so_far = {}  # New: store important results

    while step < max_steps:
        log("loop", f"Step {step + 1} started")

        # Add accumulated results to the user input for better context
        context_input = user_input
        if results_so_far:
            context_input += "\n\nPrevious results: " + ", ".join([f"{k}: {v}" for k, v in results_so_far.items()])
        
        perception = extract_perception(context_input)
        log("perception", f"Intent: {perception.intent}, Tool hint: {perception.tool_hint}")

        # Improve memory retrieval by including all previous tool outputs
        retrieved = memory.retrieve(query=context_input, top_k=5, session_filter=session_id)
        log("memory", f"Retrieved {len(retrieved)} relevant memories")

        plan = generate_plan(perception, retrieved, tool_descriptions=tool_descriptions)
        log("plan", f"Plan generated: {plan}")

        if plan.startswith("FINAL_ANSWER:"):
            log("agent", f"✅ FINAL RESULT: {plan}")
            break

        try:
            result = await execute_tool(session, tools, plan)
            log("tool", f"{result.tool_name} returned: {result.result}")

            # Store important results based on tool type
            if result.tool_name in ['add', 'subtract', 'multiply', 'divide']:
                results_so_far[f"math_{step}"] = result.result
            elif result.tool_name == 'search_documents':
                # Extract key information from search results
                if isinstance(result.result, list) and result.result:
                    # Store search results with their query to avoid repetition
                    query_key = str(result.arguments).replace(" ", "_")[:30]  # Create a short key based on the query
                    results_so_far[f"search_{query_key}"] = f"Retrieved information about: {result.arguments}"
                    
                    # Explicitly add a summary of what was found to help the agent remember
                    search_summary = f"Found information about {result.arguments}"
                    memory.add(MemoryItem(
                        text=f"SEARCH SUMMARY: {search_summary}",
                        type="fact",
                        tool_name="search_summary",
                        user_query=user_input,
                        tags=["search_summary"],
                        session_id=session_id
                    ))
            elif result.tool_name == 'ai_form_schemer':
                # Handle the ai_form_schemer tool result
                form_key = str(result.arguments).replace(" ", "_")[:30]
                results_so_far[f"form_schema_{form_key}"] = f"Created form schema for: {result.arguments}"
                
                # Add explicit memory about this form schema creation
                memory.add(MemoryItem(
                    text=f"FORM SCHEMA: Created schema for {result.arguments} with result: {result.result}",
                    type="form_schema",
                    tool_name="ai_form_schemer",
                    user_query=user_input,
                    tags=["form_schema", "ai_form_schemer"],
                    session_id=session_id
                ))
            elif result.tool_name.startswith('search_') or result.tool_name.startswith('get_'):
                # For all other search/retrieval tools, track what was retrieved
                param_key = str(result.arguments).replace(" ", "_")[:30]
                results_so_far[f"{result.tool_name}_{param_key}"] = f"Retrieved data about {result.arguments}"
                
                # Add explicit memory about this retrieval
                memory.add(MemoryItem(
                    text=f"RETRIEVAL SUMMARY: Used {result.tool_name} to get information about {result.arguments}",
                    type="fact",
                    tool_name=result.tool_name,
                    user_query=user_input,
                    tags=["retrieval_summary"],
                    session_id=session_id
                ))
            
            memory.add(MemoryItem(
                text=f"Tool call: {result.tool_name} with {result.arguments}, got: {result.result}",
                type="tool_output",
                tool_name=result.tool_name,
                user_query=user_input,
                tags=[result.tool_name],
                session_id=session_id
            ))

            user_input = f"Original task: {query}\nPrevious steps: {results_so_far}\nWhat should I do next?"

        except Exception as e:
            log("error", f"Tool execution failed: {e}")
            break

        step += 1

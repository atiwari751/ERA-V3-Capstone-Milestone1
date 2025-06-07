import asyncio
import time
import os
import datetime
from perception import extract_perception
from memory import MemoryManager, MemoryItem
from decision import generate_plan
from action import execute_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 # use this to connect to running server

import shutil
import sys
from dotenv import load_dotenv
import subprocess

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

max_steps = 30

# Load environment variables from .env
load_dotenv()

async def main(user_input: str):
    try:
        print("[agent] Starting agent...")
        print(f"[agent] Current working directory: {os.getcwd()}")
        # Pass environment to the subprocess
        server_params = StdioServerParameters(
            command="python",
            args=["mcp-server.py"],
            cwd="./.",
            env=os.environ  # Pass the environment variables to the subprocess
        )

        try:
            async with stdio_client(server_params) as (read, write):
                print("Connection established, creating session...")
                try:
                    async with ClientSession(read, write) as session:
                        print("[agent] Session created, initializing...")
 
                        try:
                            await session.initialize()
                            print("[agent] MCP session initialized")

                            # Your reasoning, planning, perception etc. would go here
                            tools = await session.list_tools()
                            print("Available tools:", [t.name for t in tools.tools])

                            # Get available tools
                            print("Requesting tool list...")
                            tools_result = await session.list_tools()
                            tools = tools_result.tools
                            tool_descriptions = "\n".join(
                                f"- {tool.name}: {getattr(tool, 'description', 'No description')}" 
                                for tool in tools
                            )

                            log("agent", f"{len(tools)} tools loaded")

                            memory = MemoryManager()
                            session_id = f"session-{int(time.time())}"
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
                        except Exception as e:
                            print(f"[agent] Session initialization error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            if "TaskGroup" in str(e):
                                print("[agent] Trying to extract TaskGroup exception details...")
                                try:
                                    inner_exc = getattr(e, "__context__", None) or getattr(e, "__cause__", None)
                                    if inner_exc:
                                        print(f"[agent] Inner exception: {type(inner_exc).__name__}: {str(inner_exc)}")
                                except:
                                    print("[agent] Could not extract inner exception")
                except Exception as e:
                    print(f"[agent] Session creation error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Ensure pipes are properly closed
                    if 'read' in locals():
                        read.close()
                    if 'write' in locals():
                        write.close()
        except Exception as e:
            print(f"[agent] Connection error: {str(e)}")
    except Exception as e:
        print(f"[agent] Overall error: {str(e)}")
    finally:
        # Clean up any remaining event loop resources
        try:
            # Get the current event loop
            loop = asyncio.get_running_loop()
            
            # Get all tasks except the current one
            current_task = asyncio.current_task(loop)
            tasks = [task for task in asyncio.all_tasks(loop) 
                    if task is not current_task and not task.done()]
            
            if tasks:
                # Cancel remaining tasks
                for task in tasks:
                    task.cancel()
                
                # Wait for cancellation to complete with a timeout
                try:
                    # Use wait_for to add a timeout to the cleanup
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    print("[agent] Cleanup timeout - some tasks may not have cleaned up properly")
                except Exception as e:
                    print(f"[agent] Cleanup warning: {str(e)}")
            
            # Ensure the loop runs one more time to process any remaining callbacks
            await asyncio.sleep(0)
            
        except Exception as e:
            print(f"[agent] Cleanup error: {str(e)}")
        finally:
            log("agent", "Agent session complete.")

if __name__ == "__main__":
    try:
        query = input("🧑 What do you want to solve today? → ")
        asyncio.run(main(query))
    except KeyboardInterrupt:
        print("\n[agent] Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"[agent] Error during execution: {str(e)}")
    finally:
        # Ensure we exit cleanly
        if sys.platform == 'win32':
            import msvcrt
            try:
                while msvcrt.kbhit():
                    msvcrt.getch()
            except: pass

# Find the ASCII values of characters in INDIA and then return sum of exponentials of those values.
# How much Anmol singh paid for his DLF apartment via Capbridge? 
# What do you know about Don Tapscott and Anthony Williams?
# What is the relationship between Gensol and Go-Auto?
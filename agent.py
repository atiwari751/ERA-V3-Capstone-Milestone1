import asyncio
import time
import os
import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import sys
from dotenv import load_dotenv

from module.utils import log  # Import the log function from utils.py
from core.agent_loop import agent_loop  # Import the agent_loop function from agent_loop.py

# Load environment variables from .env
load_dotenv()

async def main(user_input: str):
    try:
        log("agent", "Starting agent...")
        log("agent", f"Current working directory: {os.getcwd()}")
        # Pass environment to the subprocess
        server_params = StdioServerParameters(
            command="python",
            args=["mcp-server.py"],
            cwd="./.",
            env=os.environ  # Pass the environment variables to the subprocess
        )

        try:
            async with stdio_client(server_params) as (read, write):
                log("agent", "Connection established, creating session...")
                try:
                    async with ClientSession(read, write) as session:
                        log("agent", "Session created, initializing...")
 
                        try:
                            await session.initialize()
                            log("agent", "MCP session initialized")

                            # Get available tools
                            log("agent", "Requesting tool list...")
                            tools_result = await session.list_tools()
                            tools = tools_result.tools
                            tool_descriptions = "\n".join(
                                f"- {tool.name}: {getattr(tool, 'description', 'No description')}" 
                                for tool in tools
                            )

                            log("agent", f"{len(tools)} tools loaded")

                            await agent_loop(
                                user_input=user_input,
                                session=session,
                                tools=tools,
                                tool_descriptions=tool_descriptions
                            )

                        except Exception as e:
                            log("agent", f"Session initialization error: {str(e)}")
                            import traceback
                            log("agent", traceback.format_exc())
                            if "TaskGroup" in str(e):
                                log("agent", "Trying to extract TaskGroup exception details...")
                                try:
                                    inner_exc = getattr(e, "__context__", None) or getattr(e, "__cause__", None)
                                    if inner_exc:
                                        log("agent", f"Inner exception: {type(inner_exc).__name__}: {str(inner_exc)}")
                                except:
                                    log("agent", "Could not extract inner exception")
                except Exception as e:
                    log("agent", f"Session creation error: {str(e)}")
                    import traceback
                    log("agent", traceback.format_exc())
                finally:
                    # Ensure pipes are properly closed
                    if 'read' in locals():
                        read.close()
                    if 'write' in locals():
                        write.close()
        except Exception as e:
            log("agent", f"Connection error: {str(e)}")
    except Exception as e:
        log("agent", f"Overall error: {str(e)}")
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
                    log("agent", "Cleanup timeout - some tasks may not have cleaned up properly")
                except Exception as e:
                    log("agent", f"Cleanup warning: {str(e)}")
            
            # Ensure the loop runs one more time to process any remaining callbacks
            await asyncio.sleep(0)
            
        except Exception as e:
            log("agent", f"Cleanup error: {str(e)}")
        finally:
            log("agent", "Agent session complete.")

if __name__ == "__main__":
    try:
        query = input("🧑 What do you want to solve today? → ")
        asyncio.run(main(query))
    except KeyboardInterrupt:
        print("\n[agent] Received keyboard interrupt, shutting down...")
    except Exception as e:
        log("agent", f"Error during execution: {str(e)}")
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
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
import json
import shutil
import sys
from dotenv import load_dotenv
import subprocess
import random  # Import the random module

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

max_steps = 10

# Load environment variables from .env
load_dotenv()

# Define a function to generate random input data
def generate_ai_form_inputs():
    """Generates a list of 5 different input data combinations for ai_form_parser."""
    input_combinations = [
        {'extents_x_m': 23, 'extents_y_m': 23, 'grid_spacing_x_m': 6, 'grid_spacing_y_m': 6, 'no_of_floors': 3},
        {'extents_x_m': 13, 'extents_y_m': 31, 'grid_spacing_x_m': 8, 'grid_spacing_y_m': 6, 'no_of_floors': 9},
        {'extents_x_m': 19, 'extents_y_m': 25, 'grid_spacing_x_m': 5, 'grid_spacing_y_m': 8, 'no_of_floors': 9},
        {'extents_x_m': 17, 'extents_y_m': 24, 'grid_spacing_x_m': 7, 'grid_spacing_y_m': 7, 'no_of_floors': 17},
        {'extents_x_m': 25, 'extents_y_m': 30, 'grid_spacing_x_m': 6, 'grid_spacing_y_m': 8, 'no_of_floors': 5}
    ]
    return input_combinations

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
                            original_query = user_input  # Store original intent
                            scheme_results = []
                            current_scheme_params = {} # To hold the inputs for the current scheme
                            current_steel_factor = None # To hold the 2050 embodied carbon factor
                            step = 0
                            # Perform initial perception to get the task context
                            initial_perception = extract_perception(original_query)
                            num_schemes_to_generate = initial_perception.num_schemes_required or 5 # Generate 5 schemes
                            results_so_far = {}  # New: store important results
                            ai_form_inputs = generate_ai_form_inputs() # Generate input data

                            while step < max_steps and len(scheme_results) < num_schemes_to_generate:
                                log("loop", f"Step {step + 1} started. Total schemes processed: {len(scheme_results)}/{num_schemes_to_generate}")

                                # Prepare perception input for the LLM
                                # It's important to keep the original query and current status clear
                                perception_input = f"Original task: {original_query}\n"
                                perception_input += f"Currently processing scheme {len(scheme_results) + 1} of {num_schemes_to_generate}.\n"

                                # The perception itself should be based on original query, but planning needs dynamic context
                                perception = extract_perception(original_query) # Perception remains on original input for core intent
                                # Override perception with current state for decision making
                                perception.num_schemes_required = num_schemes_to_generate
                                perception.task_description = initial_perception.task_description
                                perception.site_area_sqm = initial_perception.site_area_sqm
                                perception.fsi_limit = initial_perception.fsi_limit
                                perception.location = initial_perception.location
                                perception.building_type = initial_perception.building_type
                                perception.comparison_metrics = initial_perception.comparison_metrics
                                perception.additional_requests = initial_perception.additional_requests


                                log("perception", f"Intent: {perception.intent}, Tool hint: {perception.tool_hint}")

                                # Improve memory retrieval by including all previous tool outputs
                                retrieved = memory.retrieve(query=perception_input, top_k=5, session_filter=session_id)
                                log("memory", f"Retrieved {len(retrieved)} relevant memories")

                                # Generate plan based on current state
                                if len(scheme_results) < num_schemes_to_generate:
                                    # Select input data for the current scheme
                                    input_data = ai_form_inputs[len(scheme_results)]
                                    plan = f"FUNCTION_CALL: run_ai_form_parser|input={json.dumps(input_data)}"
                                else:
                                    plan = "FINAL_ANSWER: All schemes processed. Generating final comparison and recommendations."

                                log("plan", f"Plan generated: {plan}")

                                if plan.startswith("FINAL_ANSWER:"):
                                    log("agent", f"✅ FINAL RESULT: {plan}")
                                    break

                                try:
                                    result = await execute_tool(session, tools, plan)
                                    log("tool", f"{result.tool_name} returned: {result.result}")
                                    memory_tags = [result.tool_name, f"scheme_idx_{len(scheme_results)}"] # Tag with index of current scheme being built

                                    # Parse and store specific data based on tool called
                                    if result.tool_name == "run_ai_form_parser":
                                        memory_tags.append("ai_form_output")
                                        # Store inputs and outputs for this specific scheme
                                        current_scheme_params = result.arguments['input'] # Capture the inputs
                                        # The result is a dict-like string, parse it
                                        try:
                                            # Extract the JSON string from the list
                                            if isinstance(result.result, list) and len(result.result) > 0:
                                                ai_form_output_str = result.result[0]
                                            else:
                                                log("error", f"Unexpected ai_form_parser output format: {result.result}")
                                                continue  # Skip processing if the format is unexpected
                                            
                                            ai_form_output_dict = json.loads(str(ai_form_output_str).replace("'", "\""))
                                            # Add specific metrics to the current scheme's data structure
                                            steel_tonnage = ai_form_output_dict.get('steel_tonnage_tons_per_m2')
                                            concrete_tonnage = ai_form_output_dict.get('concrete_tonnage_tons_per_m2')
                                            current_scheme_params.update({
                                                'steel_tonnage_tons_per_m2': steel_tonnage,
                                                'concrete_tonnage_tons_per_m2': concrete_tonnage,
                                                'trustworthiness': ai_form_output_dict.get('trustworthiness')
                                            })
                                            log("agent", f"AI Form data for scheme {len(scheme_results) + 1} captured.")

                                            # Search for fabricated structural steel
                                            search_plan = "FUNCTION_CALL: search_2050_products|input.product_name=Fabricated Structural Steel"
                                            search_result = await execute_tool(session, tools, search_plan)
                                            log("tool", f"{search_result.tool_name} returned: {search_result.result}")

                                            # Extract slug ID and manufacturing data
                                            if search_result.tool_name == "search_2050_products" and search_result.result:
                                                try:
                                                    products = json.loads(str(search_result.result).replace("'", "\""))
                                                    if products and isinstance(products, list) and len(products) > 0:
                                                        first_product = products[0]
                                                        slug_id = first_product.get('unique_product_uuid_v2')
                                                        
                                                        # Get 2050 product details by slug
                                                        details_plan = f"FUNCTION_CALL: get_2050_product_details_by_slug|input.slug_id={slug_id}"
                                                        details_result = await execute_tool(session, tools, details_plan)
                                                        log("tool", f"{details_result.tool_name} returned: {details_result.result}")

                                                        if details_result.tool_name == "get_2050_product_details_by_slug":
                                                            try:
                                                                details = json.loads(str(details_result.result).replace("'", "\""))
                                                                manufacturing_data = details.get('material_facts', {}).get('manufacturing')
                                                                if manufacturing_data:
                                                                    current_steel_factor = float(manufacturing_data)
                                                                    log("agent", f"2050 embodied carbon factor captured: {current_steel_factor}")

                                                                    # Multiply steel tonnage by manufacturing data
                                                                    multiply_plan = f"FUNCTION_CALL: multiply|a={steel_tonnage}|b={current_steel_factor}"
                                                                    multiply_result = await execute_tool(session, tools, multiply_plan)
                                                                    log("tool", f"{multiply_result.tool_name} returned: {multiply_result.result}")

                                                                    if multiply_result.tool_name == "multiply":
                                                                        total_steel_embodied_carbon = float(str(multiply_result.result))
                                                                        log("agent", f"Total steel embodied carbon calculated: {total_steel_embodied_carbon}")
                                                                    else:
                                                                        log("warn", "Could not calculate total steel embodied carbon.")
                                                                else:
                                                                    log("warn", "Could not extract manufacturing data from 2050 product details.")
                                                            except json.JSONDecodeError as e:
                                                                log("error", f"Failed to parse get_2050_product_details_by_slug output: {e} - {details_result.result}")
                                                except json.JSONDecodeError as e:
                                                    log("error", f"Failed to parse search_2050_products output: {e} - {search_result.result}")
                                        except json.JSONDecodeError as e:
                                            log("error", f"Failed to parse ai_form_parser output: {e} - { result.result }")

                                    elif result.tool_name == "get_2050_product_details_by_slug":
                                        memory_tags.append("2050_product_details")
                                        # The raw_response from MCP contains the Pydantic model directly
                                        if hasattr(result.raw_response, 'content') and hasattr( result.raw_response.content, 'material_facts'):
                                            material_facts_raw = result.raw_response.content.material_facts.raw_data
                                            current_steel_factor = float(material_facts_raw.get('manufacturing', 0.0))
                                            log("agent", f"2050 embodied carbon factor captured: {current_steel_factor}")
                                        else:
                                            log("warn", "Could not extract material_facts from 2050 product details.")

                                    # Store important results based on tool type
                                    elif result.tool_name in ['add', 'subtract', 'multiply', 'divide']:
                                        memory_tags.append("embodied_carbon_calculated")
                                        # This signifies completion of a scheme's calculation
                                        total_steel_embodied_carbon = float(str(result.result)) # Extract the float result

                                        # Finalize data for the current scheme
                                        scheme_data = {
                                            "scheme_number": len(scheme_results) + 1,
                                            "input_params": current_scheme_params,
                                            "steel_embodied_carbon_kg_co2e": total_steel_embodied_carbon,
                                            "steel_factor_kg_co2e_per_kg_steel": current_steel_factor,
                                            # Calculate built-up area and FSI here
                                            "built_up_area_sqm": (current_scheme_params.get('extents_x_m', 0) * current_scheme_params.get('extents_y_m', 0) *
                                                                current_scheme_params.get('no_of_floors', 0)),
                                            "green_area_sqm": None, # Placeholder: Needs assumption or calculation from agent
                                            "fsi": None, # Placeholder: Calculate below
                                        }

                                        # Calculate FSI and Green Area
                                        if initial_perception.site_area_sqm and scheme_data["built_up_area_sqm"]:
                                            scheme_data["fsi"] = scheme_data["built_up_area_sqm"] / initial_perception.site_area_sqm

                                        footprint_area = current_scheme_params.get('extents_x_m', 0) * current_scheme_params.get('extents_y_m', 0)
                                        if initial_perception.site_area_sqm and footprint_area:
                                            scheme_data["green_area_sqm"] = initial_perception.site_area_sqm - footprint_area # Simple assumption

                                        scheme_results.append(scheme_data)
                                        current_scheme_params = {} # Reset for next scheme
                                        current_steel_factor = None # Reset for next scheme
                                        log("agent", f"Scheme {scheme_data['scheme_number']} completed. Total schemes done: {len(scheme_results)}/{num_schemes_to_generate}")

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
                            else:
                                log("loop", f"Maximum steps reached or all schemes processed. Exiting loop.")

                            # After generating all schemes, generate final comparison and recommendations
                            if not len(scheme_results) == num_schemes_to_generate:
                                # Generate final comparison and recommendations
                                log("agent", "Generating final comparison and recommendations...")

                                # Request GRIHA suggestions using the search_documents tool
                                griha_query = "suggest GRIHA strategies for this building"
                                griha_plan = f"FUNCTION_CALL: search_documents|query={griha_query}"
                                try:
                                    griha_result = await execute_tool(session, tools, griha_plan)
                                    log("tool", f"{griha_result.tool_name} returned: {griha_result.result}")

                                    # Process GRIHA suggestions
                                    if griha_result.tool_name == 'search_documents' and isinstance(griha_result.result, list):
                                        griha_suggestions = "\n".join(griha_result.result)
                                        log("agent", f"GRIHA Suggestions: {griha_suggestions}")
                                        final_answer = f"FINAL_ANSWER: Comparison and recommendations based on generated schemes. GRIHA Suggestions: {griha_suggestions}"
                                    else:
                                        final_answer = "FINAL_ANSWER: Comparison and recommendations based on generated schemes, but no GRIHA suggestions found."
                                except Exception as e:
                                    log("error", f"Failed to get GRIHA suggestions: {e}")
                                    final_answer = "FINAL_ANSWER: Comparison and recommendations based on generated schemes, but failed to retrieve GRIHA suggestions."

                                log("agent", f"✅ FINAL RESULT: {final_answer}")
                            else:
                                log("agent", f"Scheme generation incomplete, {len(scheme_results)}/{num_schemes_to_generate} schemes processed.")

                        except Exception as e:
                            print(f"[agent] Session initialization error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            # Check if there's a TaskGroup error we can unwrap
                            if "TaskGroup" in str(e):
                                print("[agent] Trying to extract TaskGroup exception details...")
                                try:
                                    # Access potential __context__ or __cause__ attributes to get the real error
                                    inner_exc = getattr(e, "__context__", None) or getattr(e, "__cause__", None)
                                    if inner_exc:
                                        print(f"[agent] Inner exception: {type(inner_exc).__name__}: {str(inner_exc)}")
                                except:
                                    print("[agent] Could not extract inner exception")
                except Exception as e:
                    print(f"[agent] Session creation error: {str(e)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[agent] Connection error: {str(e)}")
    except Exception as e:
        print(f"[agent] Overall error: {str(e)}")

    log("agent", "Agent session complete.")

if __name__ == "__main__":
    query = input("🧑 What do you want to solve today? → ")
    asyncio.run(main(query))


# Find the ASCII values of characters in INDIA and then return sum of exponentials of those values.
# How much Anmol singh paid for his DLF apartment via Capbridge? 
# What do you know about Don Tapscott and Anthony Williams?
# What is the relationship between Gensol and Go-Auto?
from perception import PerceptionResult
from memory import MemoryItem
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
import os
import json

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str] = None
) -> str:
    """Generates a plan (tool call or final answer) using LLM based on structured perception and memory."""

    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"

    # Extract previous math results if available
    math_results = []
    for mem in memory_items:
        if mem.tool_name in ['add', 'subtract', 'multiply', 'divide']:
            math_results.append(f"- {mem.tool_name} result: {mem.text}")

    math_context = "\n".join(math_results) if math_results else "No previous calculations."

    tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else ""

    completed_schemes_count = 0
    scheme_data_in_memory = []
    # Parse memory items to get structured scheme data
    for m in memory_items:
        if "scheme_result" in m.tags and m.text.startswith("Scheme completed:"):
            try:
                # Assuming you've structured the 'scheme_result' memory item to contain JSON or a parsable string
                # This is a placeholder; you'd need to ensure 'agent.py' stores this structurally.
                # For now, let's assume the LLM can also infer from past tool outputs.
                completed_schemes_count += 1
                # Extract structured data if possible
                # E.g., if memory.add(MemoryItem(text=f"Scheme completed: {json.dumps(scheme_data)}", ...))
                # then: scheme_data_in_memory.append(json.loads(m.text.split("Scheme completed: ", 1)[1]))
            except Exception as e:
                log("memory_parse", f"Could not parse scheme result from memory: {m.text}, Error: {e}")

    # Construct dynamic task context for the LLM
    dynamic_task_context = f"""
Current Task Context derived from your input:
- Overall task: {perception.task_description or "Analyze building schemes."}
- Location: {perception.location or "Not specified."}
- Building Type: {perception.building_type or "Not specified."}
- Schemes required: {perception.num_schemes_required or "Undetermined. Aim for a reasonable number (e.g., 3-5) if not specified."}
"""

    if perception.site_area_sqm:
        dynamic_task_context += f"- Site Area: {perception.site_area_sqm} sq.m.\n"
    if perception.fsi_limit:
        dynamic_task_context += f"- FSI Limit: {perception.fsi_limit}.\n"
        if perception.site_area_sqm:
            max_built_up_area = perception.site_area_sqm * perception.fsi_limit
            dynamic_task_context += f"  (Max Built-up Area: {max_built_up_area} sq.m)\n"

    dynamic_task_context += f"- Metrics to compare: {', '.join(perception.comparison_metrics) or 'Not specified. Consider standard metrics like area, tonnage, carbon.'}\n"
    if perception.additional_requests:
        dynamic_task_context += f"- Additional requests: {', '.join(perception.additional_requests)}.\n"

    # General instructions for generating scheme inputs (will be refined by specific constraints)
    dynamic_task_context += """
For each scheme, you need to:
1. Determine valid input parameters for `run_ai_form_parser` (Extents X, Extents Y, Grid Spacing X, Grid Spacing Y, No. of floors) that are plausible for the specified site area and FSI. Aim for varied plausible building shapes and floor counts.
**IMPORTANT: Grid Spacing X and Grid Spacing Y MUST be integers.**
**IMPORTANT: Use ONLY the following input combinations:**
**Extents X (m): 23, Extents Y (m): 23, Grid Spacing X (m): 6, Grid Spacing Y (m): 6, No. of floors: 3**
**Extents X (m): 13, Extents Y (m): 31, Grid Spacing X (m): 8, Grid Spacing Y (m): 6, No. of floors: 9**
**Extents X (m): 19, Extents Y (m): 25, Grid Spacing X (m): 5, Grid Spacing Y (m): 8, No. of floors: 9**
**Extents X (m): 17, Extents Y (m): 24, Grid Spacing X (m): 7, Grid Spacing Y (m): 7, No. of floors: 17**
2. Call `run_ai_form_parser` with these inputs to get 'Steel tonnage (tons/m2)' and 'Concrete tonnage (tons/m2)'.
3. Find the embodied carbon for 'Fabricated Structural Steel' using `search_2050_products` and then `get_2050_product_details_by_slug`. The relevant value is `material_facts.manufacturing` from the 2050 API output.
4. Calculate 'Total Steel Embodied Carbon' by multiplying 'Steel tonnage (tons/m2)' by `material_facts.manufacturing` and the total built-up area for the scheme (Extents X * Extents Y * No. of floors).
5. Accumulate results for all schemes.
"""

    # Update progress based on dynamically parsed schemes
    dynamic_task_context += f"\nCurrent Progress: {completed_schemes_count} out of {perception.num_schemes_required or 'N/A'} schemes processed."


    prompt = f"""
You are a reasoning-driven AI agent with access to tools. Your job is to solve the user's request step-by-step by reasoning through the problem, selecting a tool if needed, and continuing until the FINAL_ANSWER is produced.{tool_context}

Always follow this loop:

1. Think step-by-step about the problem.
2. If a tool is needed, respond using the format:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2
   For tools that take a complex input object (e.g., named 'input'), use dot notation for nested parameters:
   FUNCTION_CALL: tool_name|input.nested_param1=valueA|input.nested_param2=valueB
   FUNCTION_CALL: run_ai_form_parser|input={json.dumps({"extents_x_m": 23, "extents_y_m": 23, "grid_spacing_x_m": 6, "grid_spacing_y_m": 6, "no_of_floors": 3})}
3. When the final answer is known, respond using:
   FINAL_ANSWER: [your final result]

Important context:
- Respond using EXACTLY ONE of the formats above per step.
- Do NOT include extra text, explanation, or formatting.
- Use nested keys (e.g., input.string) and square brackets for lists.
- You're currently on step {len(math_results) + 1} of solving this problem
- Previous calculations: {math_context}
- You can reference these relevant memories:
{memory_texts}

Input Summary:
- User input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {', '.join(perception.entities)}
- Tool hint: {perception.tool_hint or 'None'}
- Current results so far: {perception.user_input.split("Previous results:")[-1] if "Previous results:" in perception.user_input else "None"}

{dynamic_task_context}

IMPORTANT INSTRUCTIONS FOR MULTI-PART QUERIES:
1. Break down the user request into distinct operations
2. Keep track of what information you've already retrieved
3. DO NOT repeat searches or retrievals you've already performed - check memory first
4. Once you have gathered ALL needed information, provide a FINAL_ANSWER that includes:
   - Results of mathematical operations 
   - Information retrieved from searches
   - Any relationships or conclusions requested

When you see "Retrieved information about X" or "SEARCH SUMMARY" in your memory, this means you've already searched for this information. DO NOT search for it again.

✅ Examples:
- FUNCTION_CALL: add|a=5|b=3
- FUNCTION_CALL: strings_to_chars_to_int|input.string=INDIA
- FUNCTION_CALL: int_list_to_exponential_sum|input.int_list=[73,78,68,73,65]
- FUNCTION_CALL: search_2050_products|input.product_name=Recycled Steel S235
- FUNCTION_CALL: get_2050_product_details_by_slug|input.slug_id=some-product-slug-v2
- FINAL_ANSWER: [42]

✅ Examples:
- User asks: "What's the relationship between Cricket and Sachin Tendulkar"
  - FUNCTION_CALL: search_documents|query="relationship between Cricket and Sachin Tendulkar"
  - [receives a detailed document]
  - FINAL_ANSWER: [Sachin Tendulkar is widely regarded as the "God of Cricket" due to his exceptional skills, longevity, and impact on the sport in India. He is the leading run-scorer in both Test and ODI cricket, and the first to score 100 centuries in international cricket. His influence extends beyond his statistics, as he is seen as a symbol of passion, perseverance, and a national icon. ]


IMPORTANT:
- 🚫 Do NOT invent tools. Use only the tools listed below.
- 📄 If the question may relate to factual knowledge, use the 'search_documents' tool to look for the answer.
- 🧮 If the question is mathematical or needs calculation, use the appropriate math tool.
- 🤖 If the previous tool output already contains factual information, DO NOT search again. Instead, summarize the relevant facts and respond with: FINAL_ANSWER: [your answer]
- Only repeat `search_documents` if the last result was irrelevant or empty.
- ❌ Do NOT repeat function calls with the same parameters.
- ❌ Do NOT output unstructured responses.
- 🧠 Think before each step. Verify intermediate results mentally before proceeding.
- 💥 If unsure or no tool fits, skip to FINAL_ANSWER: [unknown]
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        log("plan", f"LLM output: {raw}")

        for line in raw.splitlines():
            if line.strip().startswith("FUNCTION_CALL:") or line.strip().startswith("FINAL_ANSWER:"):
                return line.strip()

        # If no explicit FUNCTION_CALL or FINAL_ANSWER, assume the raw response is the final answer
        return f"FINAL_ANSWER: {raw.strip()}"

    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "FINAL_ANSWER: [unknown]"

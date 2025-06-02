from perception import PerceptionResult
from memory import MemoryItem
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
import os

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

    prompt = f"""
You are a reasoning-driven AI agent with access to tools. Your job is to solve the user's request step-by-step by reasoning through the problem, selecting a tool if needed, and continuing until the FINAL_ANSWER is produced.{tool_context}

Always follow this loop:

1. Think step-by-step about the problem.
2. If a tool is needed, respond using the format:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2
   For tools that take a complex input object (e.g., named 'input'), use dot notation for nested parameters:
   FUNCTION_CALL: tool_name|input.nested_param1=valueA|input.nested_param2=valueB
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

        return raw.strip()

    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "FINAL_ANSWER: [unknown]"

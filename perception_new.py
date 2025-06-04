# perception.py
from pydantic import BaseModel
from typing import Optional, List, Dict # Import Dict
import os
from dotenv import load_dotenv
from google import genai
import re

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


class PerceptionResult(BaseModel):
    user_input: str
    intent: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None
    # New fields for structured task context
    task_description: Optional[str] = None # A summarized natural language description of the task
    num_schemes_required: Optional[int] = None
    site_area_sqm: Optional[int] = None
    fsi_limit: Optional[float] = None
    location: Optional[str] = None
    building_type: Optional[str] = None
    comparison_metrics: List[str] = [] # e.g., ["built up area", "green area", ...]
    additional_requests: List[str] = [] # e.g., ["suggest GRIHA strategies"]


def extract_perception(user_input: str) -> PerceptionResult:
    """Extracts intent, entities, tool hints, and structured task details using LLM"""

    prompt = f"""
You are an AI that extracts structured facts and detailed task requirements from user input.
Analyze the following user request and provide a detailed breakdown of the task.

User Input: "{user_input}"

Return the response as a Python dictionary with keys:
- intent: (brief phrase about what the user wants)
- entities: a list of strings representing keywords or values (e.g., ["INDIA", "ASCII"])
- tool_hint: (name of the MCP tool that might be useful, if any)
- task_description: (A concise, natural language summary of the overall task, e.g., "Generate building schemes for a site with specific constraints.")
- num_schemes_required: (Integer, if the user specifies a number of schemes, otherwise null)
- site_area_sqm: (Integer, total site area in square meters, otherwise null)
- fsi_limit: (Float, FSI limit, otherwise null)
- location: (String, location of the site, otherwise null)
- building_type: (String, type of building, e.g., "commercial", "residential", otherwise null)
- comparison_metrics: (List of strings, specific metrics the user wants to compare, e.g., ["built up area", "green area", "FSI", "total steel tonnage", "embodied carbon"])
- additional_requests: (List of strings, other specific requests like "suggest GRIHA strategies", "provide cost estimates")

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. Ensure `entities`, `comparison_metrics`, and `additional_requests` are lists of strings.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Consider gemini-1.5-flash for better parsing of complex structured output
            contents=prompt
        )
        raw = response.text.strip()
        log("perception", f"LLM output: {raw}")

        # Strip Markdown backticks if present
        clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = eval(clean) # Using eval for parsing Python dict string
        except Exception as e:
            log("perception", f"⚠️ Failed to parse cleaned output: {e}. Raw: {clean}")
            # Fallback to a basic PerceptionResult if parsing fails
            return PerceptionResult(user_input=user_input)

        # Ensure lists are lists, not dicts or other types, handling common LLM quirks
        if isinstance(parsed.get("entities"), dict): parsed["entities"] = list(parsed["entities"].values())
        if not isinstance(parsed.get("entities"), list): parsed["entities"] = []
        if not isinstance(parsed.get("comparison_metrics"), list): parsed["comparison_metrics"] = []
        if not isinstance(parsed.get("additional_requests"), list): parsed["additional_requests"] = []

        # Clean up nulls to Nones for Pydantic
        for key in ['num_schemes_required', 'site_area_sqm', 'fsi_limit', 'location', 'building_type', 'task_description']:
            if parsed.get(key) == 'null' or parsed.get(key) == '':
                parsed[key] = None

        return PerceptionResult(user_input=user_input, **parsed)

    except Exception as e:
        log("perception", f"⚠️ Extraction failed: {e}")
        return PerceptionResult(user_input=user_input)
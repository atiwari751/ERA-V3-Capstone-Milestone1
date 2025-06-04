from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
import os
import json
import faiss
import numpy as np
from pathlib import Path
import requests
from markitdown import MarkItDown
import time
from models import AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput
from models import ( Search2050ProductsInput, Search2050ProductsOutput, ProductInfo,
    Get2050ProductDetailsInput, Get2050ProductDetailsOutput, MaterialFacts, AiFormFinderInput, AiFormFinderOutput )
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv
import logging
import traceback
from typing import Optional
from ai_form_parser import AiFormFinder

load_dotenv()  # This loads the variables from .env

mcp = FastMCP("MultiToolAgentServer")

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
ROOT = Path(__file__).parent.resolve()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp-server")

# --- BEGIN 2050 Materials API Integration ---
TOKEN_CACHE_FILE = ROOT / '2050_token_cache.json' # Store cache in server's directory
BASE_API_URL = "https://app.2050-materials.com/"
TOKEN_URL = f"{BASE_API_URL}developer/api/token/getapitoken/"
# IMPORTANT: Ensure DEVELOPER_TOKEN is set as an environment variable where mcp-server.py runs
DEVELOPER_TOKEN = os.getenv("DEVELOPER_TOKEN")
# --- AiFormFinder Initialization ---
AI_FORM_SHEET_URL = os.getenv("AI_FORM_GOOGLE_SHEET_URL")
ai_form_finder_instance: Optional[AiFormFinder] = None

def initialize_services():
    global ai_form_finder_instance
    # Initialize AiFormFinder
    if AI_FORM_SHEET_URL:
        try:
            mcp_log("info", f"Initializing AiFormFinder with URL: {AI_FORM_SHEET_URL}")
            ai_form_finder_instance = AiFormFinder(AI_FORM_SHEET_URL)
            if not ai_form_finder_instance.is_ready:
                mcp_log("error", "AiFormFinder initialized but is not ready. Check sheet URL and data integrity.")
                ai_form_finder_instance = None # Ensure it's None if not ready
            else:
                mcp_log("info", "AiFormFinder initialized successfully and is ready.")
        except Exception as e:
            mcp_log("error", f"Failed to initialize AiFormFinder: {e}")
            ai_form_finder_instance = None
    else:
        mcp_log("warn", "AI_FORM_GOOGLE_SHEET_URL not set. AiFormFinder tool will not be available.")

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level.upper()}: {message}\n")
    sys.stderr.flush()
    
    # Also log to the logger
    log_level = getattr(logging, level.upper() if level.upper() in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') else 'INFO')
    logger.log(log_level, message)

def load_cached_token():
    if TOKEN_CACHE_FILE.exists():
        with open(TOKEN_CACHE_FILE, 'r') as f:
            try:
                token_data = json.load(f)
                if token_data.get("expires_at", 0) > time.time():
                    mcp_log("info", "Using cached 2050 API token.")
                    return token_data.get("api_token")
            except json.JSONDecodeError:
                mcp_log("warn", "Could not decode token cache file.")
                return None
    mcp_log("info", "No valid cached 2050 API token found.")
    return None

def save_token_to_cache(api_token, expires_in=3600):
    token_data = {
        "api_token": api_token,
        "expires_at": time.time() + expires_in - 60  # buffer of 60 seconds
    }
    with open(TOKEN_CACHE_FILE, 'w') as f:
        json.dump(token_data, f)
    mcp_log("info", "Saved new 2050 API token to cache.")

def get_2050_api_token():
    if not DEVELOPER_TOKEN:
        mcp_log("error", "DEVELOPER_TOKEN environment variable is not set.")
        raise ValueError("DEVELOPER_TOKEN is not set.")

    cached_token = load_cached_token()
    if cached_token:
        return cached_token

    headers = {'Authorization': f'Bearer {DEVELOPER_TOKEN}'}
    try:
        mcp_log("info", f"Requesting new 2050 API token from {TOKEN_URL}")
        response = requests.get(TOKEN_URL, headers=headers)
        response.raise_for_status()
        tokens = response.json()
        api_token = tokens["api_token"]
        save_token_to_cache(api_token)
        return api_token
    except requests.RequestException as e:
        mcp_log("error", f"Failed to fetch 2050 API token: {e}")
        raise Exception(f"Failed to fetch 2050 API token: {e}")
    except KeyError:
        mcp_log("error", "Invalid response format from token API.")
        raise Exception("Invalid response format from 2050 token API.")

@mcp.tool()
def search_2050_products(input: Search2050ProductsInput) -> Search2050ProductsOutput:
    """Search for products on the 2050 Materials platform by product name."""
    mcp_log("tool_call", f"search_2050_products with name: {input.product_name}")
    try:
        api_token = get_2050_api_token()
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json',
        }
        search_url = f"{BASE_API_URL}developer/api/get_products_open_api"
        params = {"name": input.product_name}

        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        api_products = data.get("products", data.get("results", []))
        output_products = []
        if api_products:
            for p in api_products:
                output_products.append(ProductInfo(
                    unique_product_uuid_v2=p.get("unique_product_uuid_v2"),
                    name=p.get("name"),
                    raw_data=p
                ))
            mcp_log("tool_result", f"Found {len(output_products)} product(s).")
            return Search2050ProductsOutput(products=output_products, message=f"Found {len(output_products)} product(s).")
        else:
            mcp_log("tool_result", "No matching products found.")
            return Search2050ProductsOutput(products=[], message="No matching products found.")

    except ValueError as ve: # Catch DEVELOPER_TOKEN not set
        mcp_log("error", f"Configuration error in search_2050_products: {ve}")
        return Search2050ProductsOutput(products=[], message=str(ve))
    except requests.RequestException as e:
        mcp_log("error", f"API request failed in search_2050_products: {e}")
        return Search2050ProductsOutput(products=[], message=f"API request failed: {e}")
    except Exception as e:
        mcp_log("error", f"Unexpected error in search_2050_products: {e}")
        return Search2050ProductsOutput(products=[], message=f"An unexpected error occurred: {e}")

@mcp.tool()
def get_2050_product_details_by_slug(input: Get2050ProductDetailsInput) -> Get2050ProductDetailsOutput:
    """Get material facts for a product from the 2050 Materials platform using its slug ID."""
    mcp_log("tool_call", f"get_2050_product_details_by_slug with slug_id: {input.slug_id}")
    try:
        api_token = get_2050_api_token()
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json',
        }
        url = f"{BASE_API_URL}api/get_product_slug?slug_id={input.slug_id}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "product" in data and "material_facts" in data["product"]:
            mf_data = data["product"]["material_facts"]
            material_facts = MaterialFacts(
                total_co2e_kg_mf=mf_data.get("total_co2e_kg_mf"),
                total_biogenic_co2e=mf_data.get("total_biogenic_co2e"),
                raw_data=mf_data
            )
            mcp_log("tool_result", f"Fetched details for slug: {input.slug_id}")
            return Get2050ProductDetailsOutput(
                product_slug_id=input.slug_id,
                material_facts=material_facts,
                message="Successfully fetched product details."
            )
        else:
            mcp_log("warn", f"Product or material_facts not found in response for slug: {input.slug_id}")
            return Get2050ProductDetailsOutput(product_slug_id=input.slug_id, message="Product or material_facts not found in API response.")

    except ValueError as ve: # Catch DEVELOPER_TOKEN not set
        mcp_log("error", f"Configuration error in get_2050_product_details_by_slug: {ve}")
        return Get2050ProductDetailsOutput(message=str(ve))
    except requests.RequestException as e:
        mcp_log("error", f"API request failed in get_2050_product_details_by_slug: {e}")
        return Get2050ProductDetailsOutput(message=f"API request failed: {e}")
    except KeyError as e:
        mcp_log("error", f"Missing expected data for slug {input.slug_id}: {e}")
        return Get2050ProductDetailsOutput(message=f"Missing expected data in API response: {e}")
    except Exception as e:
        mcp_log("error", f"Unexpected error in get_2050_product_details_by_slug: {e}")
        return Get2050ProductDetailsOutput(message=f"An unexpected error occurred: {e}")

# --- END 2050 Materials API Integration ---
# --- AiFormFinder Tool ---
@mcp.tool(description="Looks up structural engineering outputs (like steel tonnage per m2) based on building geometry inputs. Uses a predefined dataset from a Google Sheet. Requires: extents_x_m, extents_y_m, grid_spacing_x_m, grid_spacing_y_m, no_of_floors.")
def run_ai_form_parser(input: AiFormFinderInput) -> AiFormFinderOutput:
    mcp_log("tool_call", f"run_ai_form_parser with inputs: {input.model_dump_json()}")
    global ai_form_finder_instance
    if not ai_form_finder_instance or not ai_form_finder_instance.is_ready:
        mcp_log("error", "AiFormFinder tool called but not initialized or not ready.")
        return AiFormFinderOutput(message="AiFormFinder tool is not available or not properly initialized. Check server logs and AI_FORM_GOOGLE_SHEET_URL environment variable.")

    # Map Pydantic model field names to the keys expected by AiFormFinder.get_output()
    form_input_data = {
        'Extents X (m)': input.extents_x_m,
        'Extents Y (m)': input.extents_y_m,
        'Grid Spacing X (m)': input.grid_spacing_x_m,
        'Grid Spacing Y (m)': input.grid_spacing_y_m,
        'No. of floors': input.no_of_floors
    }
    mcp_log("info", f"Calling AiFormFinder with: {form_input_data}")
    try:
        output_dict = ai_form_finder_instance.get_output(form_input_data)
        if output_dict:
            mcp_log("tool_result", f"AiFormFinder returned: {output_dict}")
            # Map AiFormFinder output keys back to Pydantic model field names
            return AiFormFinderOutput(
                steel_tonnage_tons_per_m2=output_dict.get('Steel tonnage (tons/m2)'),
                column_size_mm=output_dict.get('Column size (mm)'),
                structural_depth_mm=output_dict.get('Structural depth (mm)'),
                concrete_tonnage_tons_per_m2=output_dict.get('Concrete tonnage (tons/m2)'),
                trustworthiness=output_dict.get('Trustworthiness', False),
                message="Successfully retrieved AiForm data."
            )
        else:
            mcp_log("tool_result", "No matching AiForm data found for the given inputs.")
            return AiFormFinderOutput(message="No matching AiForm data found for the given inputs.")
    except Exception as e:
        mcp_log("error", f"Error during AiFormFinder lookup: {e}")
        return AiFormFinderOutput(message=f"An error occurred during AiFormFinder lookup: {str(e)}")
# --- End AiFormFinder Tool ---

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level.upper()}: {message}\n")
    sys.stderr.flush()

@mcp.tool()
def search_documents(query: str) -> list[str]:
    """Search for relevant content from uploaded documents."""
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
        metadata = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
        query_vec = get_embedding(query).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        results = []
        for idx in I[0]:
            data = metadata[idx]
            results.append(f"{data['chunk']}\n[Source: {data['doc']}, ID: {data['chunk_id']}]")
        return results
    except Exception as e:
        return [f"ERROR: Failed to search: {str(e)}"]

@mcp.tool()
def add(input: AddInput) -> AddOutput:
    """Add two numbers"""
    print("CALLED: add(AddInput) -> AddOutput")
    return AddOutput(result=input.a + input.b)

# subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    print("CALLED: subtract(a: int, b: int) -> int:")
    return int(a - b)

# multiplication tool
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    print("CALLED: multiply(a: float, b: float) -> float:")
    return float(a * b)

#  division tool
@mcp.tool() 
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    print("CALLED: divide(a: int, b: int) -> float:")
    return float(a / b)


# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

def process_documents():
    """Process documents and create FAISS index"""
    mcp_log("INFO", "Indexing documents with MarkItDown...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    def file_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None
    all_embeddings = []
    converter = MarkItDown()

    for file in DOC_PATH.glob("*.*"):
        fhash = file_hash(file)
        if file.name in CACHE_META and CACHE_META[file.name] == fhash:
            mcp_log("SKIP", f"Skipping unchanged file: {file.name}")
            continue

        mcp_log("PROC", f"Processing: {file.name}")
        try:
            result = converter.convert(str(file))
            markdown = result.text_content
            chunks = list(chunk_text(markdown))
            embeddings_for_file = []
            new_metadata = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
                embedding = get_embedding(chunk)
                embeddings_for_file.append(embedding)
                new_metadata.append({"doc": file.name, "chunk": chunk, "chunk_id": f"{file.stem}_{i}"})
            if embeddings_for_file:
                if index is None:
                    dim = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack(embeddings_for_file))
                metadata.extend(new_metadata)
            CACHE_META[file.name] = fhash
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {file.name}: {e}")

    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        mcp_log("SUCCESS", "Saved FAISS index and metadata")
    else:
        mcp_log("WARN", "No new documents or updates to process.")

def ensure_faiss_ready():
    from pathlib import Path
    index_path = ROOT / "faiss_index" / "index.bin"
    meta_path = ROOT / "faiss_index" / "metadata.json"
    if not (index_path.exists() and meta_path.exists()):
        mcp_log("INFO", "Index not found — running process_documents()...")
        process_documents()
    else:
        mcp_log("INFO", "Index already exists. Skipping regeneration.")

# Modify the FastMCP run method to add tracing
original_run = FastMCP.run
def run_with_logging(self, *args, **kwargs):
    logger.info(f"Starting MCP server with args: {args}, kwargs: {kwargs}")
    try:
        return original_run(self, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in MCP server: {e}")
        logger.error(traceback.format_exc())
        raise
FastMCP.run = run_with_logging

# Add logging to the embeddings function
original_get_embedding = get_embedding
def get_embedding_with_logging(text: str) -> np.ndarray:
    logger.debug(f"Getting embedding for text: {text[:50]}...")
    try:
        result = original_get_embedding(text)
        logger.debug(f"Successfully got embedding of shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        logger.error(traceback.format_exc())
        raise
get_embedding = get_embedding_with_logging

if __name__ == "__main__":
    logger.info("STARTING THE SERVER")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Just print a test message and exit for quick testing
        print("MCP server test mode: OK")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "dev":
        logger.info("Running in dev mode without transport")
        try:
            initialize_services() # Initialize services like AiFormFinder
            mcp.run() # Run without transport for dev server
            logger.info("MCP server run completed normally")
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            logger.error(traceback.format_exc())
    else:
        # Start the server in a separate thread
        import threading
        logger.info("Starting server thread with stdio transport")
        
        def run_server():
            try:
                initialize_services()  # Initialize services like AiFormFinder
                logger.info("Initialized services successfully")
                logger.info("Running MCP server with stdio transport")
                # Run the MCP server with stdio transport
                ensure_faiss_ready()  # Ensure FAISS index is ready before starting server
                logger.info("FAISS index is ready, starting server...")
                # Start the MCP server
                mcp.run(transport="stdio")
                logger.info("MCP server thread completed normally")
            except Exception as e:
                logger.error(f"Error in MCP server thread: {e}")
                logger.error(traceback.format_exc())
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Process documents after server is running
        try:
            logger.info("Starting document processing")
            process_documents()
            logger.info("Document processing completed")
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            logger.error(traceback.format_exc())
        
        # Keep the main thread alive
        try:
            logger.info("Main thread waiting...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")

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
    Get2050ProductDetailsInput, Get2050ProductDetailsOutput, MaterialFacts,
    AiFormSchemerInput, AiFormSchemerOutput )
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv
import logging
import traceback
from typing import Dict, Any
from datetime import datetime

load_dotenv()  # This loads the variables from .env

mcp = FastMCP("Calculator")

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
        
        for product_json in api_products:
            parsed_product = parse_product_data(product_json)
            output_products.append(parsed_product)
            
        # Enhanced message with more details if products found
        message = "No products found."
        if output_products:
            product = output_products[0]
            message = (
                f"Found {len(output_products)} product(s). "
                f"First product: {product.name} from {product.city}, {product.manufacturing_country}. "
                f"Manufacturing emissions: {product.manufacturing_emissions} {product.declared_unit}"
            )
            
        return Search2050ProductsOutput(
            products=output_products,
            message=message
        )

    except Exception as e:
        mcp_log("error", f"API request failed in search_2050_products: {e}")
        return Search2050ProductsOutput(
            products=[],
            message=f"Error: {str(e)}"
        )

def parse_product_data(product_json: Dict[str, Any]) -> ProductInfo:
    """Helper function to parse product JSON and extract only the essential values"""
    try:
        material_facts = product_json.get("material_facts", {})
        
        return ProductInfo(
            name=product_json.get("name"),
            material_type=product_json.get("material_type"),
            manufacturing_country=product_json.get("manufacturing_country"),
            city=product_json.get("city"),
            declared_unit=material_facts.get("declared_unit"),
            manufacturing_emissions=float(material_facts.get("manufacturing", 0))
        )
    except Exception as e:
        mcp_log("error", f"Error parsing product data: {e}")
        return ProductInfo()  # Return empty product info with all fields None

# --- END 2050 Materials API Integration ---

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

@mcp.tool()
def ai_form_schemer(input: AiFormSchemerInput) -> AiFormSchemerOutput:
    """Use the structural surrogate model to evaluate a building's form."""
    try:
        # Create input parameters array from the input model
        input_params = [
            input.grid_spacing_x,
            input.grid_spacing_y,
            input.extents_x,
            input.extents_y,
            input.no_of_floors
        ]
        
        # Create model with config from environment variables
        model = create_structural_surrogate_model()
        
        # Get prediction from the model
        results = model.predict(input_params)
        
        # Extract values for the output model - strip units and convert to correct types
        steel_tonnage_str = results["steelTonnage"]["value"]
        steel_tonnage = float(steel_tonnage_str.split()[0])
        
        column_size_str = results["columnSize"]["value"]
        column_size = int(column_size_str.split()[0])
        
        structural_depth_str = results["structuralDepth"]["value"]
        structural_depth = int(structural_depth_str.split()[0])
        
        concrete_tonnage_str = results["concreteTonnage"]["value"]
        concrete_tonnage = float(concrete_tonnage_str.split()[0])
        
        trustworthy_str = results["trustworthiness"]["value"]
        trustworthy = trustworthy_str.startswith("True")
        
        mcp_log("info", f"AI Form Schema prediction completed successfully")
        
        return AiFormSchemerOutput(
            steel_tonnage=steel_tonnage,
            column_size=column_size,
            structural_depth=structural_depth,
            concrete_tonnage=concrete_tonnage,
            trustworthy=trustworthy
        )
    except Exception as e:
        mcp_log("error", f"AI Form Schema prediction failed: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")

# Classes for token handling
class Token:
    def __init__(self, token_json):
        self.token_type = token_json.get('token_type')
        self.expires_in = token_json.get('expires_in')
        self.access_token = token_json.get('access_token')
        self.timestamp = datetime.now()

    def is_expired(self):
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.expires_in

class ClientCredentials:
    def __init__(self, client, auth):
        self.client = client
        self.host = auth['host']
        self.authorize = auth['authorizePath']
        self.token = None

    def get_token_or_refresh(self):
        if self.token is None or self.token.is_expired():
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client['id'],
                'client_secret': self.client['secret'],
                'scope': self.client['scope']
            }
            url = f"{self.host}/{self.authorize}"
            response = requests.post(url, data=data)
            response.raise_for_status()
            self.token = Token(response.json())
        return self.token

class StructuralSurrogateModel:
    def __init__(self, config):
        self.api_url = config['apiUrl']
        self.api_endpoint = config['apiEndpoint']
        self.client_credentials = ClientCredentials(config['clientConfig']['client'], config['clientConfig']['auth'])
        self.confidence_level = "0.9"

    def predict(self, input_params):
        response = self.make_prediction_request(input_params)
        if not response:
            raise RuntimeError("Failed to get response from API")
        return self.parse_response(response)

    def make_prediction_request(self, input_data):
        api_url = f"{self.api_url}/{self.api_endpoint}"
        token = self.client_credentials.get_token_or_refresh()

        request_data = {
            "type": "list",
            "inputs": {
                "type": "torch_tensor",
                "data": input_data,
                "shape": [len(input_data)],
                "dtype": "torch.float32"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.access_token}"
        }

        response = requests.post(api_url, headers=headers, json=request_data)
        response.raise_for_status()
        return response.json()

    def parse_response(self, response):
        # Extract predictions
        predictions_json = json.loads(response['data']['predictions'])
        predictions = predictions_json['data'][0]['data']

        # Trustworthiness defaults
        trustworthiness = {"value": "True (75%)", "confidence": 75}
        if 'classification_predictions' in response['data']:
            classification_predictions = json.loads(response['data']['classification_predictions'])
            classification_uncertainty = json.loads(response['data']['classification_uncertainty'])

            is_trustworthy = classification_predictions['data'][0]['data'][0] == 1.0
            confidence_percent = round(classification_uncertainty['data'][0]['data'][0] * 100)
            trustworthiness = {
                "value": f"{'True' if is_trustworthy else 'False'} ({confidence_percent}%)",
                "confidence": confidence_percent
            }

        # Extract HDIs (highest density intervals)
        hdis_json = json.loads(response['data']['hdis'])
        hdis_data = hdis_json['data'][0]['data']

        results = []
        for i, value in enumerate(predictions):
            lower = hdis_data[self.confidence_level]['data']['lower']['data'][i]
            upper = hdis_data[self.confidence_level]['data']['upper']['data'][i]
            rng = abs(upper - lower)
            mean_value = abs(value)
            uncertainty_percent = (rng / (2 * mean_value)) * 100 if mean_value > 0 else 0
            results.append({
                "value": value,
                "uncertainty": min(100, uncertainty_percent)
            })

        return {
            "steelTonnage": {
                "value": f"{results[0]['value']*1000:.3f} kg/m²",
                "uncertainty": results[0]['uncertainty']
            },
            "columnSize": {
                "value": f"{round(results[1]['value'])} mm",
                "uncertainty": results[1]['uncertainty']
            },
            "structuralDepth": {
                "value": f"{round(results[2]['value'])} mm",
                "uncertainty": results[2]['uncertainty']
            },
            "concreteTonnage": {
                "value": f"{results[3]['value']*1000:.2f} kg/m²",
                "uncertainty": results[3]['uncertainty']
            },
            "trustworthiness": trustworthiness
        }

def create_structural_surrogate_model():
    """Create a StructuralSurrogateModel with config from environment variables"""
    # Check required environment variables
    required_vars = [
        'API_URL', 'API_ENDPOINT_NAME',
        'AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_SCOPE'
    ]
    missing_vars = [v for v in required_vars if not os.getenv(v)]
    if missing_vars:
        mcp_log("error", f"Missing environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    # Create model with config
    model = StructuralSurrogateModel({
        "apiUrl": os.getenv('API_URL'),
        "apiEndpoint": os.getenv('API_ENDPOINT_NAME'),
        "clientConfig": {
            "client": {
                "id": os.getenv('AZURE_CLIENT_ID'),
                "secret": os.getenv('AZURE_CLIENT_SECRET'),
                "scope": os.getenv('AZURE_SCOPE'),
            },
            "auth": {
                "authorizePath": "auth/token",
                "host": os.getenv('API_URL')
            }
        }
    })
    
    return model

if __name__ == "__main__":
    logger.info("STARTING THE SERVER")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Just print a test message and exit for quick testing
        print("MCP server test mode: OK")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "dev":
        logger.info("Running in dev mode without transport")
        try:
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

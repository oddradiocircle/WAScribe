import os
import shutil
from pathlib import Path
import re
import json
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from datetime import datetime
import signal
import sys
import atexit
import base64
import requests
from PIL import Image
import io
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import tempfile


def setup_logging(verbose=False, log_file=None):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    detailed_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, 'a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(detailed_format))
        root_logger.addHandler(file_handler)
        print(f"📝 Detailed logs will be written to: {log_file}")
    
    logger = logging.getLogger('image_seo_supername')
    if verbose:
        logger.debug("Verbose logging enabled")
    return logger


class SEONamingError(Exception):
    """Custom exception for SEO naming errors."""
    pass


class AISEOProcessor:
    """AI-powered image analysis and SEO suggestion processor"""
    
    def __init__(self, api_key: Optional[str] = None, logger=None, language='en'):
        """Initialize AI processor with optional API key."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.logger = logger or logging.getLogger('image_seo_supername')
        self.language = language
        
        if not self.api_key:
            self.logger.warning("No Mistral API key found. AI-powered features will be disabled.")
            print("⚠️ No Mistral API key found. AI-powered features will be disabled.")
        else:
            self.logger.info("Mistral API key found.")
            self._validate_api_connection()
    
    def _validate_api_connection(self):
        """Validate API connection with a simple request."""
        try:
            self.logger.debug("Validating Mistral API connection...")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=5
            )
            self.logger.debug(f"Mistral API response status: {response.status_code}")
            if response.status_code == 200:
                models = response.json()
                available_models = [model["id"] for model in models.get("data", [])]
                self.logger.debug(f"Available Mistral models: {available_models}")
                if "pixtral-12b-2409" in available_models:
                    self.logger.info("Successfully connected to Mistral API and vision model is available.")
                else:
                    self.logger.warning("Vision model 'pixtral-12b-2409' not found.")
            elif response.status_code == 401:
                self.logger.error("Authentication failed: Invalid Mistral API key.")
                self.api_key = None
            else:
                self.logger.error(f"Mistral API returned status code: {response.status_code}")
        except Exception as e:
            self.logger.exception(f"Error validating Mistral API connection: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def analyze_image(self, image_path: Path, context: dict) -> Dict[str, Any]:
        """
        Analyze image using Mistral Vision API with user-provided context.
        Returns structured information for SEO naming.
        """
        if not self.api_key:
            return {"error": "No Mistral API key available"}
        
        if not image_path.exists():
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Create a temporary file for the optimized image
            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
                try:
                    # Open the original image
                    original_image = Image.open(image_path)
                    
                    # Resize to 1920px on the longest dimension while maintaining aspect ratio
                    width, height = original_image.size
                    max_dimension = max(width, height)
                    if max_dimension > 1920:
                        scale_factor = 1920 / max_dimension
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
                    else:
                        resized_image = original_image
                    
                    # Save as WebP with 90% quality
                    quality = 90
                    resized_image.save(temp_path, 'WEBP', quality=quality)
                    
                    # Check file size and reduce quality if needed
                    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                    self.logger.debug(f"Initial optimized image size: {file_size_mb:.2f}MB")
                    
                    # If still over 10MB, progressively reduce quality until under limit
                    if file_size_mb > 10:
                        self.logger.debug("Image still too large, reducing quality")
                        for quality in [80, 70, 60, 50, 40, 30, 20]:
                            resized_image.save(temp_path, 'WEBP', quality=quality)
                            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                            self.logger.debug(f"Reduced quality to {quality}, new size: {file_size_mb:.2f}MB")
                            if file_size_mb <= 10:
                                break
                    
                    # Final size check
                    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 10:
                        return {"error": f"Unable to compress image below 10MB limit. Current size: {file_size_mb:.1f}MB"}
                    
                    # Convert optimized image to base64
                    with open(temp_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                except Exception as e:
                    return {"error": f"Image processing error: {str(e)}"}
                
            # Prepare user context as a string for the prompt
            context_str = "\n".join([f"- {key}: {value}" for key, value in context.items() if value])

            # Language-specific instructions
            language_instructions = {
                'en': "Provide a JSON response in English with:",
                'es': "Proporciona una respuesta JSON en español con:"
            }
            lang_instruction = language_instructions.get(self.language, language_instructions['en'])

            # Create the prompt for image analysis according to Mistral API docs
            prompt = f"""
            This image is for a marketing campaign. Please analyze it considering this context:

            {context_str}

            {lang_instruction}
            1. A list of 5-7 SEO keywords extracted from the image (most relevant first)
            2. Main subject or product visible
            3. Visual characteristics (colors, style, composition)
            4. Context/setting of the image
            5. Suggested title (max 60 chars) - an engaging, descriptive title
            6. Suggested caption (max 150 chars) - a brief caption explaining the image 
            7. Suggested alt text (max 125 chars) - accessible description for screen readers
            8. Suggested description (max 300 chars) - detailed SEO-rich description

            Format as: {{"keywords": ["word1", "word2",...], "subject": "...", "visual": "...", "context": "...", "title": "...", "caption": "...", "alt_text": "...", "description": "..."}}
            """
            
            # Call Mistral API following the official format in documentation
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Using the chat completions endpoint with vision capability as per docs
            data = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    }
                ],
                "max_tokens": 1200  # Increased to accommodate additional metadata
            }
            
            print(f"\r⏳ Analyzing image: {image_path.name}", end="", flush=True)
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            # Clean up the temporary file
            if temp_path.exists():
                os.unlink(temp_path)
            
            # Handle different error codes as per Mistral API docs
            if response.status_code == 200:
                # Extract and parse JSON from response
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON part from response (it might contain explanatory text)
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group(0))
                        return analysis_data
                    except json.JSONDecodeError as e:
                        return {"error": f"Failed to parse JSON from response: {e}", "raw_response": content}
                
                # Fallback: return the full text if JSON parsing fails
                return {"error": "Failed to parse structured data", "raw_response": content}
            elif response.status_code == 400:
                return {"error": "Bad request: " + response.json().get("error", {}).get("message", "Unknown error")}
            elif response.status_code == 401:
                return {"error": "Authentication error: Invalid API key"}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please try again later."}
            elif response.status_code == 500:
                return {"error": "Mistral API server error. Please try again later."}
            else:
                error_msg = f"API error ({response.status_code}): {response.text[:100]}..."
                return {"error": error_msg}
                
        except Exception as e:
            # Ensure temp file cleanup in case of exceptions
            if 'temp_path' in locals() and temp_path.exists():
                os.unlink(temp_path)
            return {"error": f"Analysis error: {str(e)}"}

    def suggest_seo_name(self, analysis: Dict[str, Any], context: dict, sequence_number: Optional[int] = None) -> str:
        """Generate SEO-friendly name from AI analysis and context."""
        components = []

        # Extract keywords from AI analysis if available - but limit to fewer keywords
        if "keywords" in analysis and isinstance(analysis["keywords"], list):
            # Only use the first 2 most relevant keywords to avoid including everything from context
            keywords = [self._normalize_text(k) for k in analysis["keywords"][:2]]
            components.extend(keywords)

        # Add subject if available - this is most reliable for what's actually in the image
        if "subject" in analysis and analysis["subject"]:
            subject = self._normalize_text(analysis["subject"])
            if subject and not any(subject in comp or comp in subject for comp in components):
                components.append(subject)
                
        # If we have visual details, extract specific items visible in the image
        if "visual" in analysis and analysis["visual"]:
            visual_text = analysis["visual"].lower()
            visual_items = []
            
            # Check for common items - only add if actually mentioned in visual description
            for item in ["camiseta", "t-shirt", "tee", "shirt", "bolso", "bag", "wallet", "billetera"]:
                if item in visual_text and not any(item in comp for comp in components):
                    normalized_item = self._normalize_text(item)
                    visual_items.append(normalized_item)
            
            # Add up to 2 visual items that are actually visible
            for item in visual_items[:2]:
                if not any(item in comp or comp in item for comp in components):
                    components.append(item)

        # Always add brand name (usually safe to include)
        if context.get("brand"):
            brand = self._normalize_text(context.get("brand", ""))
            if brand and not any(brand in comp or comp in brand for comp in components):
                components.append(brand)
                
        # Only add other context items if we don't have enough components
        if len(components) < 2:
            for key in ["product", "category"]:
                if context.get(key):
                    component = self._normalize_text(context.get(key, ""))
                    if component and not any(component in comp or comp in component for comp in components):
                        components.append(component)

        # Combine components
        if not components:
            base_name = f"{context.get('brand', 'img')}-{context.get('product', 'product')}"
            if sequence_number is not None:
                base_name = f"{base_name}-{sequence_number:03d}"
            return base_name

        # Create name with limited components - use fewer components for precision
        base_name = "-".join(components[:3])  # Limit to 3 components for more focused names

        # Truncate to whole words
        if len(base_name) > 50:
            parts = base_name.split('-')
            truncated = parts[0]
            for part in parts[1:]:
                if len(truncated) + len(part) + 1 <= 50:  # +1 for the hyphen
                    truncated += f"-{part}"
                else:
                    break
            base_name = truncated

        if sequence_number is not None:
            return f"{base_name}-{sequence_number:03d}"  # Add sequence
        else:
            return base_name

    def _normalize_text(self, text: str) -> str:
        """Normalize text for SEO filename use."""
        if not text:
            return ""
        text = str(text).lower()
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        return text.strip("-")


class SEOImageProcessor:
    def __init__(self, language='en'):
        # Stop words for English and Spanish
        self.stop_words = {
            'en': {'a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'with'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'de', 'del', 'al', 'a', 'en', 'con', 'para', 'por'}
        }
        self.language = language

    def _normalize_text(self, text: str) -> str:
        """Normalize text for SEO filename use."""
        if not text:
            return ""
        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)
        return text.strip('-')

    def generate_seo_name(self, context: dict, sequence_number: Optional[int] = None) -> str:
        """Generate SEO-optimized filename."""
        components = []
        # Add keywords if provided
        if context.get('keywords'):
            keywords = [self._normalize_text(k.strip()) for k in context.get('keywords', '').split(',') if k.strip()]
            components.extend(keywords[:2])  # Use up to 2 keywords
        # Add product, brand, and category
        for key in ['product', 'brand', 'category']:
            component = self._normalize_text(context.get(key, ''))
            if component and component not in components:
                components.append(component)
        # Filter out stop words
        stop_words = self.stop_words.get(self.language, self.stop_words['en'])
        components = [c for c in components if c and c not in stop_words]
        
        # Ensure sequence number is always included
        base_name = "-".join(components)
        
        # Add sequence number explicitly at the end
        if sequence_number is not None:
            final_name = f"{base_name}-{sequence_number:03d}"
        else:
            final_name = base_name
            
        return final_name[:60]  # Limit to 60 characters

    def validate_seo_name(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate SEO filename."""
        issues = []
        if len(filename) > 60:
            issues.append("Filename exceeds 60 characters.")
        if re.search(r'[^a-z0-9\-]', filename):
            issues.append("Filename contains invalid characters.")
        return len(issues) == 0, issues


class ImageRenamer:
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, language='en', 
                 safe_mode=True, use_ai=False, logger=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.language = language
        self.processor = SEOImageProcessor(language=language)
        self.renamed_files = {}
        self.history_file = self.output_dir / "rename_history.json"
        self.safe_mode = safe_mode
        self.use_ai = use_ai
        self.logger = logger or logging.getLogger('image_seo_supername')
        
        self.logger.info(f"Initializing ImageRenamer with input_dir={input_dir}, output_dir={self.output_dir}")
        self.logger.info(f"Configuration: language={language}, safe_mode={safe_mode}, use_ai={use_ai}")
        
        self.ai_processor = None
        if use_ai:
            try:
                self.logger.info("Initializing AI processor...")
                self.ai_processor = AISEOProcessor(logger=self.logger, language=language)
                if not self.ai_processor.api_key:
                    self.logger.warning("AI features disabled due to missing API key")
                    self.use_ai = False
            except Exception as e:
                self.logger.exception(f"Failed to initialize AI processor: {e}")
                self.use_ai = False
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_history()
        
        atexit.register(self.save_history)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
    
    def _handle_exit(self, signum, frame):
        """Handle exit signals by saving history."""
        self.logger.warning("Process interrupted. Saving history...")
        print("\n\n⚠️ Process interrupted. Saving history...")
        self.save_history()
        sys.exit(1)

    def load_history(self):
        """Load renaming history from file if exists."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.renamed_files = json.load(f)
                self.logger.info(f"Loaded history with {len(self.renamed_files)} entries.")
                print(f"📋 Loaded history with {len(self.renamed_files)} entries.")
            except Exception as e:
                self.logger.exception(f"Error loading history: {e}")
                print(f"⚠️ Error loading history: {e}")
                self.renamed_files = {}
        else:
            self.renamed_files = {}

    def save_history(self):
        """Save renaming history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.renamed_files, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved history with {len(self.renamed_files)} entries to {self.history_file}")
            print(f"\n📋 Saved history with {len(self.renamed_files)} entries to {self.history_file}")
        except Exception as e:
            self.logger.exception(f"Error saving history: {e}")
            print(f"⚠️ Error saving history: {e}")

    def collect_user_context(self) -> dict:
        """Collect user context for SEO naming."""
        prompts = {
            'en': {
                'brand': "Brand/Company (required): ",
                'product': "Product/Service (required): ",
                'category': "Category (required): ",
                'location': "Location/Geographic area: ",
                'keywords': "Keywords (comma separated): ",
                'additional': "Additional context (helps AI better understand images):"
            },
            'es': {
                'brand': "Marca/Empresa (requerido): ",
                'product': "Producto/Servicio (requerido): ",
                'category': "Categoría (requerido): ",
                'location': "Ubicación/Área geográfica: ",
                'keywords': "Palabras clave (separadas por comas): ",
                'additional': "Contexto adicional (ayuda a la IA a entender mejor las imágenes):"
            }
        }
        lang_prompts = prompts.get(self.language, prompts['en'])
        
        print("\n📝 Please provide the context for naming your images:")
        context = {
            'brand': input(lang_prompts['brand']).strip(),
            'product': input(lang_prompts['product']).strip(),
            'category': input(lang_prompts['category']).strip(),
            'location': input(lang_prompts['location']).strip(),
            'keywords': input(lang_prompts['keywords']).strip(),
        }
        
        # Collect additional context if AI is enabled
        if self.use_ai:
            print("\n👁️ AI Analysis enabled - Additional context helps improve accuracy")
            print("Examples: target audience, purpose of images, campaign details, specific features to highlight")
            context['additional'] = input(lang_prompts['additional'] + "\n").strip()
            
        # Validate required fields
        if not context['brand'] or not context['product'] or not context['category']:
            print("⚠️ Brand, product, and category are required fields.")
            return self.collect_user_context()
        
        # Show summary for confirmation
        print("\nSEO Context Summary:")
        for key, value in context.items():
            if value:  # Only show non-empty values
                print(f"  - {key.capitalize()}: {value}")
        
        # Example generated name
        example_name = self.processor.generate_seo_name(context, sequence_number=1)
        print(f"\nExample generated name: {example_name}.jpg")
        
        confirm = input("\nIs this correct? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Let's try again.")
            return self.collect_user_context()
            
        return context

    def process_images(self, context: dict):
        """Process images in the input directory."""
        # Find all images in the input directory
        image_files = [f for f in self.input_dir.glob('*.*') 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}]
        
        if not image_files:
            self.logger.warning("No images found in the directory.")
            print("⚠️ No images found in the directory.")
            return
            
        print(f"\n🔍 Found {len(image_files)} images to process.")
        self.logger.info(f"Found {len(image_files)} images to process.")
        
        # Set operation mode
        operation_mode = "COPY" if self.safe_mode else "RENAME"
        print(f"\n⚠️ Operation mode: {operation_mode}")
        self.logger.info(f"Operation mode: {operation_mode}")
        if not self.safe_mode:
            confirm = input("WARNING: Rename mode will modify original files. Are you sure? (y/n): ")
            if confirm.lower() not in ['y', 'yes']:
                print("Operation canceled by user.")
                self.logger.warning("Operation canceled by user.")
                return
        
        # Show AI mode if enabled
        if self.use_ai:
            print("🤖 AI-powered image analysis: ENABLED")
            self.logger.info("AI-powered image analysis: ENABLED")
                
        # Verify disk space for copy mode
        if self.safe_mode and self.output_dir != self.input_dir:
            total_size = sum(f.stat().st_size for f in image_files)
            free_space = shutil.disk_usage(self.output_dir).free
            if total_size > free_space * 0.9:  # Leave 10% margin
                print(f"⚠️ Warning: Low disk space. Required: {total_size/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
                self.logger.warning(f"Low disk space. Required: {total_size/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() not in ['y', 'yes']:
                    print("Operation canceled by user.")
                    self.logger.warning("Operation canceled by user due to low disk space.")
                    return

        # Verify names before processing
        test_names = {}
        duplicates = []
        
        # Process each image
        processed_count = 0
        skipped_count = 0
        error_count = 0
        ai_analyzed_count = 0
        
        # Progress bar for processing
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                sequence_number = i + 1
                ai_analysis = None
                if self.use_ai and self.ai_processor:
                    try:
                        ai_analysis = self.ai_processor.analyze_image(image_path, context)
                        if "error" in ai_analysis:
                            print(f"\n⚠️ AI analysis failed for {image_path.name}: {ai_analysis['error']}")
                            self.logger.warning(f"AI analysis failed for {image_path.name}: {ai_analysis['error']}")
                            print("   Falling back to standard naming.")
                        else:
                            ai_analyzed_count += 1
                            print(f"\r✅ AI analysis succeeded for {image_path.name}" + " " * 30, end="", flush=True)
                            self.logger.info(f"AI analysis succeeded for {image_path.name}")
                    except Exception as e:
                        print(f"\n⚠️ AI analysis failed for {image_path.name}: {e}")
                        self.logger.exception(f"AI analysis failed for {image_path.name}: {e}")
                        print("   Falling back to standard naming.")
                elif self.use_ai:
                    print(f"\n⚠️ AI analysis skipped for {image_path.name}: AI processor not available")
                    self.logger.warning(f"AI analysis skipped for {image_path.name}: AI processor not available")
                
                if ai_analysis and "error" not in ai_analysis:
                    new_name_base = self.ai_processor.suggest_seo_name(ai_analysis, context)
                    if not new_name_base:
                        new_name_base = self.processor.generate_seo_name(context, sequence_number)
                else:
                    new_name_base = self.processor.generate_seo_name(context, sequence_number)
                
                new_name_base = self.processor._normalize_text(new_name_base)
                new_name = f"{new_name_base}{image_path.suffix.lower()}"
                new_path = self.output_dir / new_name

                if new_path.exists() and not (self.safe_mode and self.output_dir == self.input_dir):
                    if ai_analysis and "error" not in ai_analysis:
                        new_name_base = self.ai_processor.suggest_seo_name(ai_analysis, context, sequence_number)
                        new_name = f"{new_name_base}{image_path.suffix.lower()}"
                        new_path = self.output_dir / new_name

                    if new_path.exists():
                        for j in range(1, 100):
                            alt_name = f"{new_name_base}-alt{j}{image_path.suffix.lower()}"
                            alt_path = self.output_dir / alt_name
                            if not alt_path.exists():
                                new_name = alt_name
                                new_path = alt_path
                                break
                        else:
                            print(f"\n⚠️ Skipping {image_path.name}: Unable to generate unique name")
                            self.logger.warning(f"Skipping {image_path.name}: Unable to generate unique name")
                            skipped_count += 1
                            continue

                self.renamed_files[str(image_path)] = {
                    "original_name": image_path.name,
                    "new_name": new_name,
                    "timestamp": datetime.now().isoformat(),
                    "ai_analyzed": ai_analysis is not None and "error" not in ai_analysis
                }

                if ai_analysis and "error" not in ai_analysis:
                    for field in ["alt_text", "title", "caption", "description"]:
                        if field in ai_analysis:
                            self.renamed_files[str(image_path)][field] = ai_analysis[field]

                if self.safe_mode:
                    shutil.copy2(image_path, new_path)
                else:
                    shutil.move(image_path, new_path)

                processed_count += 1
                if processed_count % 10 == 0:
                    self.save_history()
                    
            except Exception as e:
                print(f"\n⚠️ Error processing {image_path.name}: {e}")
                self.logger.exception(f"Error processing {image_path.name}: {e}")
                error_count += 1
        
        # Final save of history
        self.save_history()
        
        # Show summary
        print("\n✅ Processing complete!")
        print(f"   - Processed: {processed_count} images")
        print(f"   - Skipped: {skipped_count} images")
        print(f"   - Errors: {error_count} images")
        if self.use_ai:
            print(f"   - AI analyzed: {ai_analyzed_count} images")
            print("   - Generated SEO metadata: title, caption, alt text, and description")
        print(f"\n📂 Output directory: {self.output_dir}")
        print(f"📋 History saved to: {self.history_file}")
        self.logger.info(f"Processing complete: Processed={processed_count}, Skipped={skipped_count}, Errors={error_count}, AI analyzed={ai_analyzed_count}")

    def restore_files(self, history_file: Path, force: bool = False):
        """Restore files to original names using history file."""
        if not history_file.exists():
            print(f"⚠️ History file not found: {history_file}")
            self.logger.warning(f"History file not found: {history_file}")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history file: {e}")
            self.logger.exception(f"Error loading history file: {e}")
            return
        
        print(f"📋 Loaded history with {len(history)} entries.")
        self.logger.info(f"Loaded history with {len(history)} entries.")
        
        # Check which files can be restored
        restorable = []
        not_found = []
        
        for orig_path_str, info in history.items():
            orig_path = Path(orig_path_str)
            orig_dir = orig_path.parent
            orig_name = info["original_name"]
            new_name = info["new_name"]
            
            # Find the renamed file
            if self.safe_mode:
                # In COPY mode, the original should still exist
                current_path = orig_path
                if not current_path.exists():
                    not_found.append((orig_name, new_name, "Original missing"))
                    continue
            else:
                # In RENAME mode, look for the new name
                current_path = orig_dir / new_name
                if not current_path.exists():
                    not_found.append((orig_name, new_name, "Renamed file missing"))
                    continue
            
            # Check if restoration would overwrite
            target_path = orig_dir / orig_name
            if target_path.exists() and target_path != current_path and not force:
                not_found.append((orig_name, new_name, "Would overwrite existing file"))
                continue
                
            restorable.append((current_path, target_path, orig_name, new_name))
        
        if not restorable:
            print("⚠️ No files can be restored.")
            self.logger.warning("No files can be restored.")
            if not_found:
                print(f"   {len(not_found)} files could not be found or would overwrite existing files.")
                self.logger.warning(f"{len(not_found)} files could not be found or would overwrite existing files.")
            return
        
        print(f"\n✅ Found {len(restorable)} files that can be restored.")
        self.logger.info(f"Found {len(restorable)} files that can be restored.")
        if not_found:
            print(f"⚠️ {len(not_found)} files cannot be restored (use --force to override).")
            self.logger.warning(f"{len(not_found)} files cannot be restored (use --force to override).")
            
        confirm = input("\nRestore these files? (y/n): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Operation canceled by user.")
            self.logger.warning("Operation canceled by user.")
            return
        
        # Restore files
        restored = 0
        errors = 0
        
        for current_path, target_path, orig_name, new_name in tqdm(restorable, desc="Restoring files"):
            try:
                if self.safe_mode:
                    # In COPY mode, we just need to copy back to original name
                    shutil.copy2(current_path, target_path)
                else:
                    # In RENAME mode, we rename back to original
                    shutil.move(current_path, target_path)
                restored += 1
            except Exception as e:
                print(f"\n⚠️ Error restoring {new_name} to {orig_name}: {e}")
                self.logger.exception(f"Error restoring {new_name} to {orig_name}: {e}")
                errors += 1
        
        print("\n✅ Restoration complete!")
        print(f"   - Restored: {restored} files")
        print(f"   - Errors: {errors} files")
        self.logger.info(f"Restoration complete: Restored={restored}, Errors={errors}")
        
    def show_recovery_options(self, history_file: Path):
        """Show recovery options based on history file."""
        if not history_file.exists():
            print(f"⚠️ History file not found: {history_file}")
            self.logger.warning(f"History file not found: {history_file}")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history file: {e}")
            self.logger.exception(f"Error loading history file: {e}")
            return
        
        print(f"📋 Loaded history with {len(history)} entries.")
        self.logger.info(f"Loaded history with {len(history)} entries.")
        
        # Analyze history for recovery options
        original_exists = 0
        renamed_exists = 0
        both_exist = 0
        none_exist = 0
        
        for orig_path_str, info in history.items():
            orig_path = Path(orig_path_str)
            orig_dir = orig_path.parent
            renamed_path = orig_dir / info["new_name"]
            
            orig_exists = orig_path.exists()
            renamed_exists = renamed_path.exists()
            
            if orig_exists and renamed_exists:
                both_exist += 1
            elif orig_exists:
                original_exists += 1
            elif renamed_exists:
                renamed_exists += 1
            else:
                none_exist += 1
        
        print("\n🔍 Recovery Analysis:")
        print(f"   - Files with both original and renamed versions: {both_exist}")
        print(f"   - Files with only original version: {original_exists}")
        print(f"   - Files with only renamed version: {renamed_exists}")
        print(f"   - Files with neither version found: {none_exist}")
        self.logger.info(f"Recovery Analysis: Both={both_exist}, Original={original_exists}, Renamed={renamed_exists}, None={none_exist}")
        
        if both_exist + renamed_exists > 0:
            print("\n💡 Recovery Options:")
            print("   1. To restore original names:")
            print(f"      python image_seo_supername.py --restore --history \"{history_file}\"")
            if both_exist > 0:
                print("\n   2. If you want to force overwrite existing files:")
                print(f"      python image_seo_supername.py --restore --history \"{history_file}\" --force")
        else:
            print("\n⚠️ No files available for recovery.")
            self.logger.warning("No files available for recovery.")


def main():
    """Main function to process command line arguments and execute the tool."""
    parser = argparse.ArgumentParser(description="Image SEO SuperName: Optimize image filenames for SEO")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-r', '--rename', action='store_true', help='Rename images mode')
    mode_group.add_argument('-s', '--restore', action='store_true', help='Restore original filenames mode')
    mode_group.add_argument('-o', '--recovery-options', action='store_true', help='Show recovery options')
    
    parser.add_argument('-i', '--input', type=str, help='Input directory containing images')
    parser.add_argument('-O', '--output', type=str, help='Output directory (default: same as input)')
    parser.add_argument('-l', '--language', default='en', choices=['en', 'es'], help='Language for prompts (en=English, es=Spanish)')
    parser.add_argument('-m', '--move', action='store_true', help='Move files instead of copying (WARNING: This will modify original files)')
    parser.add_argument('-H', '--history', type=str, help='Path to history file for restore/recovery')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite existing files during restore')
    
    parser.add_argument('-a', '--ai', action='store_true', help='Enable AI-powered image analysis for enhanced naming')
    parser.add_argument('-k', '--api-key', type=str, help='Mistral API key (can also be set as MISTRAL_API_KEY env variable)')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging with detailed information')
    parser.add_argument('--log-file', type=str, default='image_seo_supername.log', 
                        help='Path to log file (default: image_seo_supername.log in input directory)')
    parser.add_argument('--no-log-file', action='store_true', help='Disable logging to file')
    
    args = parser.parse_args()
    
    # Place log file in the input directory when specified
    log_file = None
    if not args.no_log_file:
        if args.input:
            # Use the input directory for the log file
            input_dir = Path(args.input)
            log_file = input_dir / (args.log_file if '/' not in args.log_file else Path(args.log_file).name)
        else:
            # Fallback to the specified log file path or default
            log_file = args.log_file
    
    logger = setup_logging(verbose=args.verbose, log_file=log_file)
    logger.info("Starting Image SEO SuperName")
    logger.debug(f"Command line arguments: {args}")
    
    load_dotenv()
    
    if args.api_key:
        logger.debug("Setting API key from command line argument")
        os.environ["MISTRAL_API_KEY"] = args.api_key
    
    if args.rename:
        if not args.input:
            logger.error("Input directory is required for rename mode")
            parser.error("--input directory is required for rename mode")
        input_dir = Path(args.input)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory not found: {args.input}")
            parser.error(f"Input directory not found: {args.input}")
        
        output_dir = Path(args.output) if args.output else input_dir
        renamer = ImageRenamer(
            input_dir=input_dir,
            output_dir=output_dir,
            language=args.language,
            safe_mode=not args.move,
            use_ai=args.ai,
            logger=logger
        )
        
        logger.info("Running in rename mode")
        context = renamer.collect_user_context()
        logger.debug(f"User context: {context}")
        renamer.process_images(context)
    
    elif args.restore:
        if not args.history:
            logger.error("History file is required for restore mode")
            parser.error("--history file is required for restore mode")
        history_file = Path(args.history)
        renamer = ImageRenamer(
            input_dir=Path.cwd(),
            safe_mode=not args.move,
            logger=logger
        )
        logger.info("Running in restore mode")
        renamer.restore_files(history_file, force=args.force)
    
    elif args.recovery_options:
        if not args.history:
            logger.error("History file is required for recovery options mode")
            parser.error("--history file is required for recovery options mode")
        history_file = Path(args.history)
        renamer = ImageRenamer(input_dir=Path.cwd(), logger=logger)
        logger.info("Running in recovery options mode")
        renamer.show_recovery_options(history_file)

    logger.info("Image SEO SuperName completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
        logging.getLogger('image_seo_supername').warning("Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.getLogger('image_seo_supername').exception(f"Error: {e}")
        sys.exit(1)

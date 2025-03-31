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


class SEONamingError(Exception):
    """Custom exception for SEO naming errors."""
    pass


class AISEOProcessor:
    """AI-powered image analysis and SEO suggestion processor"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize AI processor with optional API key."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            print("⚠️ No Mistral API key found. AI-powered features will be disabled.")
            print("   Please set MISTRAL_API_KEY environment variable or use --api-key option.")
        else:
            # No format validation since Mistral API key format may vary
            print("✅ Mistral API key found.")
            # Perform a quick validation of the API connection
            self._validate_api_connection()
    
    def _validate_api_connection(self):
        """Validate API connection with a simple request."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # Test the connection using the models endpoint which is lightweight
            response = requests.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=5  # Short timeout for quick check
            )
            
            if response.status_code == 200:
                # Check if vision model is available
                models = response.json()
                available_models = [model["id"] for model in models.get("data", [])]
                if "pixtral-12b-2409" in available_models:
                    print("✅ Successfully connected to Mistral API and vision model is available.")
                else:
                    print("✅ Connected to Mistral API successfully.")
                    print("⚠️ The vision model 'pixtral-12b-2409' was not found in available models.")
                    print("   Available models:", ", ".join(available_models[:5]) + 
                          ("..." if len(available_models) > 5 else ""))
            elif response.status_code == 401:
                print("❌ Authentication failed: Invalid API key.")
                print("   Please check your Mistral API key and try again.")
                self.api_key = None  # Disable AI features
            else:
                print(f"⚠️ Mistral API returned status code: {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"⚠️ Could not validate Mistral API connection: {e}")
            print("   AI features may still work, but please check your internet connection.")
    
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
            
        # Check file size (10MB limit for Mistral API)
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            return {"error": f"Image too large ({file_size_mb:.1f}MB). Maximum size is 10MB."}
            
        try:
            # Convert to base64
            with open(image_path, "rb") as image_file:
                # Try to open the image to validate it
                Image.open(image_file).verify()
                image_file.seek(0)  # Reset file pointer
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Prepare user context as a string for the prompt
            context_str = "\n".join([f"- {key}: {value}" for key, value in context.items() if value])
                
            # Create the prompt for image analysis according to Mistral API docs
            prompt = f"""
            This image is for a marketing campaign. Please analyze it considering this context:
            
            {context_str}
            
            Provide a JSON response with:
            1. A list of 5-7 SEO keywords extracted from the image (most relevant first)
            2. Main subject or product visible
            3. Visual characteristics (colors, style, composition)
            4. Context/setting of the image
            5. Suggested alt text (max 125 chars)
            
            Format as: {{"keywords": ["word1", "word2",...], "subject": "...", "visual": "...", "context": "...", "alt_text": "..."}}
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
                "max_tokens": 800
            }
            
            print(f"\r⏳ Analyzing image: {image_path.name}", end="", flush=True)
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
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
            return {"error": f"Analysis error: {str(e)}"}


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
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, language='en', safe_mode=True, use_ai=False):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.language = language
        self.processor = SEOImageProcessor(language=language)
        self.renamed_files = {}
        self.history_file = self.output_dir / "rename_history.json"
        self.safe_mode = safe_mode  # Safe mode to copy instead of rename
        self.use_ai = use_ai  # Whether to use AI features
        
        # Initialize AI processor if needed
        self.ai_processor = None
        if use_ai:
            try:
                self.ai_processor = AISEOProcessor()
                if not self.ai_processor.api_key:
                    print("⚠️ AI features disabled due to missing API key.")
                    self.use_ai = False
            except Exception as e:
                print(f"⚠️ AI features disabled: {e}")
                self.use_ai = False
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_history()
        
        # Handle safe exit
        atexit.register(self.save_history)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
    
    def _handle_exit(self, signum, frame):
        """Handle exit signals by saving history."""
        print("\n\n⚠️ Process interrupted. Saving history...")
        self.save_history()
        sys.exit(1)

    def load_history(self):
        """Load renaming history from file if exists."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.renamed_files = json.load(f)
                print(f"📋 Loaded history with {len(self.renamed_files)} entries.")
            except Exception as e:
                print(f"⚠️ Error loading history: {e}")
                self.renamed_files = {}
        else:
            self.renamed_files = {}

    def save_history(self):
        """Save renaming history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.renamed_files, f, indent=2, ensure_ascii=False)
            print(f"\n📋 Saved history with {len(self.renamed_files)} entries to {self.history_file}")
        except Exception as e:
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
            print("⚠️ No images found in the directory.")
            return
            
        print(f"\n🔍 Found {len(image_files)} images to process.")
        
        # Set operation mode
        operation_mode = "COPY" if self.safe_mode else "RENAME"
        print(f"\n⚠️ Operation mode: {operation_mode}")
        if not self.safe_mode:
            confirm = input("WARNING: Rename mode will modify original files. Are you sure? (y/n): ")
            if confirm.lower() not in ['y', 'yes']:
                print("Operation canceled by user.")
                return
        
        # Show AI mode if enabled
        if self.use_ai:
            print("🤖 AI-powered image analysis: ENABLED")
                
        # Verify disk space for copy mode
        if self.safe_mode and self.output_dir != self.input_dir:
            total_size = sum(f.stat().st_size for f in image_files)
            free_space = shutil.disk_usage(self.output_dir).free
            if total_size > free_space * 0.9:  # Leave 10% margin
                print(f"⚠️ Warning: Low disk space. Required: {total_size/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() not in ['y', 'yes']:
                    print("Operation canceled by user.")
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
                # Generate the new name with sequence number to avoid duplicates
                sequence_number = i + 1
                
                # AI analysis for enhanced naming (if enabled)
                ai_analysis = None
                if self.use_ai and self.ai_processor:
                    try:
                        ai_analysis = self.ai_processor.analyze_image(image_path, context)
                        if "error" in ai_analysis:
                            print(f"\n⚠️ AI analysis failed for {image_path.name}: {ai_analysis['error']}")
                            print("   Falling back to standard naming.")
                        else:
                            ai_analyzed_count += 1
                            print(f"\r✅ AI analysis succeeded for {image_path.name}" + " " * 30, end="", flush=True)
                    except Exception as e:
                        print(f"\n⚠️ AI analysis failed for {image_path.name}: {e}")
                        print("   Falling back to standard naming.")
                elif self.use_ai:
                    print(f"\n⚠️ AI analysis skipped for {image_path.name}: AI processor not available")
                
                # Generate the new name
                if ai_analysis and "error" not in ai_analysis:
                    # Use AI-powered naming
                    new_name_base = self.ai_processor.suggest_seo_name(ai_analysis, context, sequence_number)
                    # Fallback to standard naming if AI naming fails
                    if not new_name_base:
                        new_name_base = self.processor.generate_seo_name(context, sequence_number)
                else:
                    # Use standard naming
                    new_name_base = self.processor.generate_seo_name(context, sequence_number)
                
                # Ensure the new name is valid, normalize if needed
                new_name_base = self.processor._normalize_text(new_name_base)
                
                # Add extension
                new_name = f"{new_name_base}{image_path.suffix.lower()}"
                
                # Check for duplicate names in the output directory
                new_path = self.output_dir / new_name
                if new_path.exists() and not (self.safe_mode and self.output_dir == self.input_dir):
                    # If duplicate, try adding additional sequence number
                    for j in range(1, 100):
                        alt_name = f"{new_name_base}-alt{j}{image_path.suffix.lower()}"
                        alt_path = self.output_dir / alt_name
                        if not alt_path.exists():
                            new_name = alt_name
                            new_path = alt_path
                            break
                    else:
                        # If all alternatives exist, skip this file
                        print(f"\n⚠️ Skipping {image_path.name}: Unable to generate unique name")
                        skipped_count += 1
                        continue
                
                # Store original and new names
                self.renamed_files[str(image_path)] = {
                    "original_name": image_path.name,
                    "new_name": new_name,
                    "timestamp": datetime.now().isoformat(),
                    "ai_analyzed": ai_analysis is not None and "error" not in ai_analysis
                }
                
                # Generate alt text if AI was used
                if ai_analysis and "alt_text" in ai_analysis:
                    self.renamed_files[str(image_path)]["alt_text"] = ai_analysis["alt_text"]
                
                # Perform the actual file operation (copy or rename)
                if self.safe_mode:
                    shutil.copy2(image_path, new_path)
                else:
                    shutil.move(image_path, new_path)
                
                processed_count += 1
                
                # Save history periodically
                if processed_count % 10 == 0:
                    self.save_history()
                    
            except Exception as e:
                print(f"\n⚠️ Error processing {image_path.name}: {e}")
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
        print(f"\n📂 Output directory: {self.output_dir}")
        print(f"📋 History saved to: {self.history_file}")

    def restore_files(self, history_file: Path, force: bool = False):
        """Restore files to original names using history file."""
        if not history_file.exists():
            print(f"⚠️ History file not found: {history_file}")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history file: {e}")
            return
        
        print(f"📋 Loaded history with {len(history)} entries.")
        
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
            if not_found:
                print(f"   {len(not_found)} files could not be found or would overwrite existing files.")
            return
        
        print(f"\n✅ Found {len(restorable)} files that can be restored.")
        if not_found:
            print(f"⚠️ {len(not_found)} files cannot be restored (use --force to override).")
            
        confirm = input("\nRestore these files? (y/n): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Operation canceled by user.")
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
                errors += 1
        
        print("\n✅ Restoration complete!")
        print(f"   - Restored: {restored} files")
        print(f"   - Errors: {errors} files")
        
    def show_recovery_options(self, history_file: Path):
        """Show recovery options based on history file."""
        if not history_file.exists():
            print(f"⚠️ History file not found: {history_file}")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history file: {e}")
            return
        
        print(f"📋 Loaded history with {len(history)} entries.")
        
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
        
        if both_exist + renamed_exists > 0:
            print("\n💡 Recovery Options:")
            print("   1. To restore original names:")
            print(f"      python image_seo_supername.py --restore --history \"{history_file}\"")
            if both_exist > 0:
                print("\n   2. If you want to force overwrite existing files:")
                print(f"      python image_seo_supername.py --restore --history \"{history_file}\" --force")
        else:
            print("\n⚠️ No files available for recovery.")


def main():
    """Main function to process command line arguments and execute the tool."""
    parser = argparse.ArgumentParser(description="Image SEO SuperName: Optimize image filenames for SEO")
    
    # Define mode groups
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-r', '--rename', action='store_true', help='Rename images mode')
    mode_group.add_argument('-s', '--restore', action='store_true', help='Restore original filenames mode')
    mode_group.add_argument('-o', '--recovery-options', action='store_true', help='Show recovery options')
    
    # Common arguments
    parser.add_argument('-i', '--input', type=str, help='Input directory containing images')
    parser.add_argument('-O', '--output', type=str, help='Output directory (default: same as input)')
    parser.add_argument('-l', '--language', default='en', choices=['en', 'es'], help='Language for prompts (en=English, es=Spanish)')
    parser.add_argument('-m', '--move', action='store_true', help='Move files instead of copying (WARNING: This will modify original files)')
    parser.add_argument('-H', '--history', type=str, help='Path to history file for restore/recovery')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite existing files during restore')
    
    # AI-specific arguments
    parser.add_argument('-a', '--ai', action='store_true', help='Enable AI-powered image analysis for enhanced naming')
    parser.add_argument('-k', '--api-key', type=str, help='Mistral API key (can also be set as MISTRAL_API_KEY env variable)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set API key from argument or environment
    if args.api_key:
        os.environ["MISTRAL_API_KEY"] = args.api_key
    
    # Rename mode
    if args.rename:
        if not args.input:
            parser.error("--input directory is required for rename mode")
        input_dir = Path(args.input)
        if not input_dir.exists() or not input_dir.is_dir():
            parser.error(f"Input directory not found: {args.input}")
        
        output_dir = Path(args.output) if args.output else input_dir
        renamer = ImageRenamer(
            input_dir=input_dir,
            output_dir=output_dir,
            language=args.language,
            safe_mode=not args.move,
            use_ai=args.ai
        )
        
        print("\n🔤 Image SEO SuperName - Rename Mode")
        if args.ai:
            print("🤖 AI-powered image analysis: ENABLED")
        context = renamer.collect_user_context()
        renamer.process_images(context)
    
    # Restore mode
    elif args.restore:
        if not args.history:
            parser.error("--history file is required for restore mode")
        history_file = Path(args.history)
        renamer = ImageRenamer(
            input_dir=Path.cwd(),  # Not used in restore mode
            safe_mode=not args.move
        )
        print("\n🔄 Image SEO SuperName - Restore Mode")
        renamer.restore_files(history_file, force=args.force)
    
    # Recovery options mode
    elif args.recovery_options:
        if not args.history:
            parser.error("--history file is required for recovery options mode")
        history_file = Path(args.history)
        renamer = ImageRenamer(input_dir=Path.cwd())  # Not used in this mode
        print("\n🛟 Image SEO SuperName - Recovery Options")
        renamer.show_recovery_options(history_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

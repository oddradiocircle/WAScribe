import os
import shutil
from pathlib import Path
import re
import json
import unicodedata
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from datetime import datetime
import signal
import sys
import atexit


class SEONamingError(Exception):
    """Custom exception for SEO naming errors."""
    pass


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
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, language='en', safe_mode=True):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.language = language
        self.processor = SEOImageProcessor(language=language)
        self.renamed_files = {}
        self.history_file = self.output_dir / "rename_history.json"
        self.safe_mode = safe_mode  # Safe mode to copy instead of rename
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_history()
        
        # Handle safe exit
        atexit.register(self.save_history)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
    
    def _handle_exit(self, signum, frame):
        """Handle safe exit by saving history"""
        print("\nInterruption detected, saving history...")
        self.save_history()
        sys.exit(0)

    def load_history(self):
        """Load renaming history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.renamed_files = data.get('renamed_files', {})
                print(f"History loaded: {len(self.renamed_files)} files previously processed.")
            except json.JSONDecodeError:
                print("Error loading history. Starting with empty history.")
                self.renamed_files = {}

    def save_history(self):
        """Save renaming history."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    'renamed_files': self.renamed_files,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"History saved to {self.history_file}")
        except Exception as e:
            print(f"Error saving history: {e}")

    def collect_user_context(self) -> dict:
        """Collect user context for SEO naming."""
        prompts = {
            'en': {
                'brand': "Brand/Company (required): ",
                'product': "Product/Service (required): ",
                'category': "Category (required): ",
                'keywords': "Keywords (comma separated): "
            },
            'es': {
                'brand': "Marca/Empresa (requerido): ",
                'product': "Producto/Servicio (requerido): ",
                'category': "Categoría (requerido): ",
                'keywords': "Palabras clave (separadas por comas): "
            }
        }
        lang_prompts = prompts.get(self.language, prompts['en'])
        print("\n📝 Please provide the context for naming your images:")
        context = {
            'brand': input(lang_prompts['brand']).strip(),
            'product': input(lang_prompts['product']).strip(),
            'category': input(lang_prompts['category']).strip(),
            'keywords': input(lang_prompts['keywords']).strip()
        }
        # Validate required fields
        if not context['brand'] or not context['product']:
            print("⚠️ Brand and product are required fields.")
            return self.collect_user_context()
        
        # Show summary for confirmation
        print("\nSEO Context Summary:")
        for key, value in context.items():
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
        
        for i, image_path in enumerate(image_files):
            base_name = self.processor.generate_seo_name(context, sequence_number=i + 1)
            new_name = f"{base_name}{image_path.suffix.lower()}"
            if new_name in test_names:
                duplicates.append((new_name, image_path, test_names[new_name]))
            test_names[new_name] = image_path
        
        # If duplicates exist, alert
        if duplicates:
            print("\n⚠️ WARNING: Duplicate names detected that would cause overwrites.")
            print("This should not occur with the sequence number, but will be corrected during processing.")

        # Process each image
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            # Skip if already processed
            if str(image_path) in self.renamed_files and Path(self.renamed_files[str(image_path)]).exists():
                skipped_count += 1
                continue
                
            try:
                # Generate SEO name with sequence number to ensure uniqueness
                base_name = self.processor.generate_seo_name(context, sequence_number=i + 1)
                new_name = f"{base_name}{image_path.suffix.lower()}"
                output_path = self.output_dir / new_name
                
                # Check for name collisions and adjust if necessary
                counter = 1
                while output_path.exists():
                    # Add an additional counter if the name already exists
                    adjusted_name = f"{base_name}-{counter}{image_path.suffix.lower()}"
                    output_path = self.output_dir / adjusted_name
                    counter += 1
                
                # Perform operation based on mode
                if self.safe_mode:
                    # Copy file (safe mode)
                    shutil.copy2(image_path, output_path)
                else:
                    # Rename file (caution: modifies original)
                    image_path.rename(output_path)
                
                # Register in history
                self.renamed_files[str(image_path)] = str(output_path)
                processed_count += 1
                
                # Save history periodically
                if processed_count % 10 == 0:
                    self.save_history()
                    
            except Exception as e:
                print(f"\n❌ Error processing {image_path}: {e}")
                error_count += 1
                
        # Save final history
        self.save_history()
        
        # Show summary
        print(f"\n✅ Process completed:")
        print(f"  - Images processed: {processed_count}")
        print(f"  - Images skipped (already processed): {skipped_count}")
        print(f"  - Errors: {error_count}")
        
        if self.safe_mode and processed_count > 0:
            print(f"\nOriginal files remain intact in:")
            print(f"  {self.input_dir}")
            print(f"Renamed versions are located in:")
            print(f"  {self.output_dir}")
        
        print(f"\nHistory saved to: {self.history_file}")


def restore_from_history(history_file: Path, force: bool = False):
    """Restore original filenames from history file."""
    if not history_file.exists():
        print(f"Error: History file {history_file} does not exist.")
        return 1
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            renamed_files = history.get('renamed_files', {})
            
        if not renamed_files:
            print("No renaming information in the history file.")
            return 1
            
        print(f"Found {len(renamed_files)} files to restore.")
        
        # Check if target files exist
        existing_destinations = [Path(new) for new in renamed_files.values() if Path(new).exists()]
        missing_destinations = [Path(new) for new in renamed_files.values() if not Path(new).exists()]
        
        print(f"- {len(existing_destinations)} renamed files found")
        print(f"- {len(missing_destinations)} renamed files not found")
        
        if not existing_destinations:
            print("No files found to restore.")
            return 1
            
        # Check if original files already exist (avoid overwriting)
        existing_originals = [Path(orig) for orig in renamed_files.keys() if Path(orig).exists()]
        if existing_originals and not force:
            print(f"\n⚠️ WARNING: {len(existing_originals)} original files already exist.")
            print("This could cause data loss. Use --force to overwrite.")
            return 1
            
        # Restore files
        restored = 0
        failed = 0
        for original, renamed in renamed_files.items():
            orig_path = Path(original)
            new_path = Path(renamed)
            if new_path.exists():
                try:
                    if orig_path.exists() and not force:
                        # If original exists, make a copy with another name
                        backup_path = orig_path.with_name(f"{orig_path.stem}_restored{orig_path.suffix}")
                        shutil.copy2(new_path, backup_path)
                        print(f"✓ Restored as: {backup_path.name} (original already existed)")
                    else:
                        # Restore to original name
                        os.makedirs(orig_path.parent, exist_ok=True)
                        shutil.copy2(new_path, orig_path)
                        print(f"✓ Restored: {orig_path.name}")
                    restored += 1
                except Exception as e:
                    print(f"✗ Error restoring {orig_path.name}: {e}")
                    failed += 1
        
        print(f"\nRestoration completed: {restored} files restored, {failed} errors.")
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"Error during restoration: {e}")
        return 1


def show_recovery_options(history_file: Path):
    """Display recovery options based on the history file."""
    if not history_file.exists():
        print(f"ERROR: History file not found at {history_file}")
        return 1
    
    try:
        # Load the history file
        with open(history_file, 'r') as f:
            history = json.load(f)
            renamed_files = history.get('renamed_files', {})
        
        # Check if any files were renamed
        if not renamed_files:
            print("No renamed files in the history.")
            return 1
            
        # Check if any target files exist
        target_exists = False
        for target_path in renamed_files.values():
            if Path(target_path).exists():
                target_exists = True
                break
                
        if not target_exists:
            print("Renamed files not found. They may have been moved or deleted.")
            
        print(f"\nCurrent situation:")
        print(f"- Total original files: {len(renamed_files)}")
        
        print("\nRecovery options:")
        
        # Option 1: Recover using this script
        print("\n1. Restore files automatically")
        print("   Run: python image_seo_supername.py --restore --history [path_to_history]")
        
        # Option 2: Recover from OneDrive
        print("\n2. Recover from OneDrive version history")
        print("   If using OneDrive, you can:")
        print("   a) Go to https://onedrive.live.com")
        print("   b) Navigate to the folder containing your images")
        print("   c) Right-click the folder and select 'Version history'")
        print("   d) Restore the version prior to running the script")
        
        # Option 3: Recycle Bin
        print("\n3. Check the Recycle Bin")
        print("   Files may be in the Recycle Bin.")
        
        # Print list of original files as reference
        print("\nList of original files (first 10):")
        for i, original in enumerate(sorted(renamed_files.keys())[:10]):
            print(f"  {i+1}. {Path(original).name}")
        
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more")
            
        return 0
        
    except Exception as e:
        print(f"Error during recovery: {e}")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SEO Image Renamer with recovery functionality")
    
    # Create argument groups for different functionalities
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--rename', '-r', action='store_true', help="Rename images mode")
    mode_group.add_argument('--restore', '-s', action='store_true', help="Restore renamed images mode")
    mode_group.add_argument('--recovery-options', '-o', action='store_true', help="Show recovery options")
    
    # Arguments for rename mode
    rename_group = parser.add_argument_group('Rename options')
    rename_group.add_argument('--input', '-i', type=str, help="Directory with images to process")
    rename_group.add_argument('--output', '-O', type=str, help="Output directory (default: same as input)")
    rename_group.add_argument('--language', '-l', type=str, choices=['en', 'es'], default='es', 
                        help="Language for messages and processing (en=English, es=Spanish)")
    rename_group.add_argument('--move', '-m', action='store_true', 
                        help="Use MOVE mode instead of COPY (DANGEROUS, not recommended)")
    
    # Arguments for restore mode
    restore_group = parser.add_argument_group('Restore options')
    restore_group.add_argument('--history', '-H', type=str, 
                         help="Renaming history file (rename_history.json)")
    restore_group.add_argument('--force', '-f', action='store_true',
                         help="Overwrite existing files if necessary during restoration")
    
    args = parser.parse_args()

    # Load environment variables (if needed in the future)
    load_dotenv()
    
    # Rename mode
    if args.rename:
        if not args.input:
            parser.error("The --input argument is required in rename mode")
            
        # Validate directories
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"ERROR: Input directory {input_dir} does not exist.")
            return 1
        
        output_dir = Path(args.output) if args.output else None
        
        # Create instance with safe mode by default (copy instead of rename)
        safe_mode = not args.move
        renamer = ImageRenamer(input_dir, output_dir, language=args.language, safe_mode=safe_mode)
        
        # Show warning in dangerous mode
        if not safe_mode:
            print("\n⚠️ WARNING: MOVE mode activated. This mode modifies original files.")
            print("If you prefer to keep your original files, press Ctrl+C and run without --move\n")
        
        try:
            # Collect user context
            context = renamer.collect_user_context()
            # Process images
            renamer.process_images(context)
            return 0
        except KeyboardInterrupt:
            print("\nOperation canceled by user.")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    # Restore mode
    elif args.restore:
        if not args.history:
            parser.error("The --history argument is required in restore mode")
        
        history_file = Path(args.history)
        return restore_from_history(history_file, args.force)
    
    # Show recovery options
    elif args.recovery_options:
        if not args.history:
            parser.error("The --history argument is required to show recovery options")
        
        history_file = Path(args.history)
        return show_recovery_options(history_file)


if __name__ == "__main__":
    sys.exit(main())

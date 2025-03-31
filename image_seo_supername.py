import os
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
        # Add sequence number if provided
        if sequence_number is not None:
            components.append(f"{sequence_number:02d}")
        # Join components with hyphens and limit length
        filename = "-".join(components)
        return filename[:60]

    def validate_seo_name(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate SEO filename."""
        issues = []
        if len(filename) > 60:
            issues.append("Filename exceeds 60 characters.")
        if re.search(r'[^a-z0-9\-]', filename):
            issues.append("Filename contains invalid characters.")
        return len(issues) == 0, issues


class ImageRenamer:
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, language='en'):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.language = language
        self.processor = SEOImageProcessor(language=language)
        self.renamed_files = {}
        self.history_file = self.output_dir / "rename_history.json"
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_history()

    def load_history(self):
        """Load renaming history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.renamed_files = json.load(f).get('renamed_files', {})

    def save_history(self):
        """Save renaming history."""
        with open(self.history_file, 'w') as f:
            json.dump({'renamed_files': self.renamed_files}, f, indent=2)

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
        print("\n📝 Por favor, proporciona el contexto para nombrar tus imágenes:")
        context = {
            'brand': input(lang_prompts['brand']).strip(),
            'product': input(lang_prompts['product']).strip(),
            'category': input(lang_prompts['category']).strip(),
            'keywords': input(lang_prompts['keywords']).strip()
        }
        # Validate required fields
        if not context['brand'] or not context['product']:
            print("⚠️ Marca y producto son campos obligatorios.")
            return self.collect_user_context()
        return context

    def process_images(self, context: dict):
        """Process images in the input directory."""
        image_files = [f for f in self.input_dir.glob('*.*') if f.suffix.lower() in {'.jpg', '.png'}]
        if not image_files:
            print("No images found.")
            return

        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            new_name = self.processor.generate_seo_name(context, sequence_number=i + 1) + image_path.suffix
            output_path = self.output_dir / new_name
            image_path.rename(output_path)
            self.renamed_files[str(image_path)] = str(output_path)
        self.save_history()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SEO Image Renamer")
    parser.add_argument('--input', '-i', type=str, required=True, help="Input directory with images")
    parser.add_argument('--output', '-o', type=str, help="Output directory (default: same as input)")
    parser.add_argument('--language', '-l', type=str, choices=['en', 'es'], default='es', 
                        help="Language for prompts and processing (en=English, es=Spanish)")
    args = parser.parse_args()

    load_dotenv()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    renamer = ImageRenamer(input_dir, output_dir, language=args.language)
    context = renamer.collect_user_context()
    renamer.process_images(context)


if __name__ == "__main__":
    main()

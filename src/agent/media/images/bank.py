"""
bank.py - Banco de imagenes local (dataset interno)

El banco de imagenes contiene:
1. Registro de imagenes locales del dataset
2. Metadata completa con licencias

Estructura del directorio:
    src/assets/images/
        plc.jpg
        plc_diagram.png
        abb_robot.jpg
        image_registry.json
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.agent.media.images.metadata import (
    ImageMetadata,
    ImageSource,
    ImageCategory,
    ImageLicense,
    LicenseType,
    ImageSearchResult,
)


logger = logging.getLogger(__name__)


class ImageBank:
    """
    Banco de imagenes local con registro persistente.

    Funcionalidades:
    - Registro de imagenes locales
    - Busqueda por categoria, tags, keywords
    - Metadata con licencias
    """

    DEFAULT_ASSETS_DIR = "src/assets/images"
    REGISTRY_FILE = "image_registry.json"

    def __init__(self, assets_dir: Optional[str] = None):
        self.assets_dir = Path(assets_dir or self.DEFAULT_ASSETS_DIR)
        self.registry_path = self.assets_dir / self.REGISTRY_FILE
        self._registry: Dict[str, ImageMetadata] = {}

        self._setup_directories()
        self._load_registry()

    def _setup_directories(self):
        """Crea la estructura de directorios"""
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _load_registry(self):
        """Carga el registro desde disco"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for img_data in data.get("images", []):
                        try:
                            img = ImageMetadata(**img_data)
                            self._registry[img.id] = img
                        except Exception as e:
                            logger.warning(f"Error loading image {img_data.get('id')}: {e}")
                logger.info(f"Loaded {len(self._registry)} images from registry")
            except Exception as e:
                logger.error(f"Error loading image registry: {e}")

    def _save_registry(self):
        """Guarda el registro a disco"""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "total_images": len(self._registry),
                "images": [img.model_dump() for img in self._registry.values()]
            }
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving image registry: {e}")

    def add(self, image: ImageMetadata) -> bool:
        """
        Agrega una imagen al banco.

        Args:
            image: Metadatos de la imagen

        Returns:
            True si se agrego correctamente
        """
        try:
            # Generar ID si no tiene
            if not image.id:
                image.id = image.generate_id()

            # Verificar que el archivo existe si es local
            if image.local_path:
                if not Path(image.local_path).exists():
                    logger.warning(f"Image file not found: {image.local_path}")

            self._registry[image.id] = image
            self._save_registry()
            return True

        except Exception as e:
            logger.error(f"Error adding image to bank: {e}")
            return False

    def remove(self, image_id: str) -> bool:
        """Elimina una imagen del banco"""
        if image_id in self._registry:
            del self._registry[image_id]
            self._save_registry()
            return True
        return False

    def get(self, image_id: str) -> Optional[ImageMetadata]:
        """Obtiene una imagen por ID"""
        return self._registry.get(image_id)

    def search(
        self,
        query: str = "",
        category: Optional[ImageCategory] = None,
        tags: Optional[List[str]] = None,
        license_types: Optional[List[LicenseType]] = None,
        max_results: int = 10
    ) -> ImageSearchResult:
        """
        Busca imagenes en el banco local.

        Args:
            query: Texto de busqueda
            category: Filtrar por categoria
            tags: Filtrar por tags
            license_types: Filtrar por tipos de licencia
            max_results: Maximo de resultados

        Returns:
            Resultado de busqueda
        """
        import time
        start = time.time()

        results: List[ImageMetadata] = []

        for img in self._registry.values():
            # Solo incluir imagenes con local_path valido
            if not img.local_path:
                continue

            # Filtrar por categoria
            if category and img.category != category:
                continue

            # Filtrar por tags
            if tags:
                img_tags_lower = [t.lower() for t in img.tags]
                if not any(t.lower() in img_tags_lower for t in tags):
                    continue

            # Filtrar por licencia
            if license_types:
                if img.license.type not in license_types:
                    continue

            # Calcular relevancia si hay query
            if query:
                score = img.matches_query(query)
                if score > 0:
                    img.relevance_score = score
                    results.append(img)
            else:
                results.append(img)

        # Ordenar por relevancia
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limitar resultados
        results = results[:max_results]

        return ImageSearchResult(
            images=results,
            total=len(results),
            query=query,
            source=ImageSource.LOCAL,
            search_time_ms=(time.time() - start) * 1000,
        )

    def get_by_category(self, category: ImageCategory) -> List[ImageMetadata]:
        """Obtiene todas las imagenes de una categoria"""
        return [img for img in self._registry.values() if img.category == category]

    def get_all(self) -> List[ImageMetadata]:
        """Obtiene todas las imagenes del banco"""
        return list(self._registry.values())

    def get_available_images(self) -> List[ImageMetadata]:
        """Obtiene solo las imagenes con archivo local disponible"""
        return [img for img in self._registry.values() if img.local_path]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadisticas del banco"""
        by_category = {}
        by_license = {}
        total_size = 0
        available_count = 0

        for img in self._registry.values():
            # Por categoria
            cat = img.category.value if hasattr(img.category, 'value') else str(img.category)
            by_category[cat] = by_category.get(cat, 0) + 1

            # Por licencia
            lic = img.license.type.value if hasattr(img.license.type, 'value') else str(img.license.type)
            by_license[lic] = by_license.get(lic, 0) + 1

            # Tamano
            if img.size_bytes:
                total_size += img.size_bytes

            # Disponibles
            if img.local_path:
                available_count += 1

        return {
            "total_images": len(self._registry),
            "available_images": available_count,
            "by_category": by_category,
            "by_license": by_license,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "assets_dir": str(self.assets_dir),
        }


# Singleton accessor
_bank_instance: Optional[ImageBank] = None


def get_image_bank(assets_dir: Optional[str] = None) -> ImageBank:
    """Obtiene la instancia singleton del banco de imagenes"""
    global _bank_instance
    if _bank_instance is None:
        _bank_instance = ImageBank(assets_dir)
    return _bank_instance

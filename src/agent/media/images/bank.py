"""
bank.py - Banco de imagenes local (dataset interno)

El banco de imagenes contiene:
1. Registro de imagenes pre-definidas por categoria
2. Imagenes descargadas y validadas
3. Metadata completa con licencias verificadas

Estructura del directorio:
    assets/images/
        plc/
        cobot/
        diagrams/
        safety/
        ...
    assets/image_registry.json
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.agent.services.images.metadata import (
    ImageMetadata,
    ImageSource,
    ImageCategory,
    ImageLicense,
    LicenseType,
    ImageSearchResult,
)


logger = logging.getLogger(__name__)


# ============================================
# REGISTRO DE IMAGENES PRE-DEFINIDAS
# ============================================

# Imagenes de ejemplo que se pueden agregar al banco
# Cada entrada define una imagen que se puede descargar de fuentes libres
PREDEFINED_IMAGES: List[Dict[str, Any]] = [
    # PLCs Siemens
    {
        "id": "plc_s7_1200_front",
        "title": "Siemens S7-1200 PLC - Vista frontal",
        "category": ImageCategory.PLC_SIEMENS,
        "tags": ["plc", "siemens", "s7-1200", "cpu", "industrial"],
        "keywords": ["S7-1200", "CPU", "Siemens PLC", "Compact PLC"],
        "suggested_search": "Siemens S7-1200 PLC industrial",
        "license_type": LicenseType.CC_BY,
    },
    {
        "id": "plc_s7_1500_rack",
        "title": "Siemens S7-1500 con modulos I/O",
        "category": ImageCategory.PLC_SIEMENS,
        "tags": ["plc", "siemens", "s7-1500", "rack", "modules"],
        "keywords": ["S7-1500", "modular PLC", "I/O modules"],
        "suggested_search": "Siemens S7-1500 rack modules",
        "license_type": LicenseType.CC_BY,
    },
    # Cobots
    {
        "id": "cobot_ur5e",
        "title": "Universal Robots UR5e Cobot",
        "category": ImageCategory.COBOT_UR,
        "tags": ["cobot", "universal robots", "ur5e", "collaborative"],
        "keywords": ["UR5e", "cobot", "collaborative robot"],
        "suggested_search": "Universal Robots UR5e collaborative robot",
        "license_type": LicenseType.CC_BY,
    },
    {
        "id": "cobot_fanuc_crx",
        "title": "FANUC CRX Cobot",
        "category": ImageCategory.COBOT_FANUC,
        "tags": ["cobot", "fanuc", "crx", "collaborative"],
        "keywords": ["FANUC CRX", "cobot"],
        "suggested_search": "FANUC CRX collaborative robot",
        "license_type": LicenseType.CC_BY,
    },
    # Diagramas
    {
        "id": "ladder_logic_example",
        "title": "Ejemplo de diagrama Ladder Logic",
        "category": ImageCategory.LADDER_LOGIC,
        "tags": ["ladder", "plc", "programming", "diagram"],
        "keywords": ["ladder logic", "PLC programming"],
        "suggested_search": "ladder logic diagram PLC",
        "license_type": LicenseType.CC0,
    },
    {
        "id": "profinet_topology",
        "title": "Topologia de red PROFINET",
        "category": ImageCategory.PROFINET,
        "tags": ["profinet", "network", "topology", "industrial ethernet"],
        "keywords": ["PROFINET", "network topology"],
        "suggested_search": "PROFINET network topology diagram",
        "license_type": LicenseType.CC_BY,
    },
    # HMI
    {
        "id": "hmi_panel_siemens",
        "title": "Panel HMI Siemens",
        "category": ImageCategory.HMI_SCREEN,
        "tags": ["hmi", "siemens", "panel", "touchscreen"],
        "keywords": ["HMI panel", "SIMATIC HMI"],
        "suggested_search": "Siemens HMI panel industrial",
        "license_type": LicenseType.CC_BY,
    },
    # Seguridad
    {
        "id": "safety_light_curtain",
        "title": "Cortina de luz de seguridad",
        "category": ImageCategory.SAFETY_EQUIPMENT,
        "tags": ["safety", "light curtain", "protection"],
        "keywords": ["safety light curtain", "machine guarding"],
        "suggested_search": "safety light curtain industrial",
        "license_type": LicenseType.CC_BY,
    },
    {
        "id": "emergency_stop_button",
        "title": "Boton de paro de emergencia E-Stop",
        "category": ImageCategory.SAFETY_EQUIPMENT,
        "tags": ["safety", "e-stop", "emergency", "button"],
        "keywords": ["E-stop", "emergency stop"],
        "suggested_search": "emergency stop button industrial",
        "license_type": LicenseType.CC0,
    },
    # General industrial
    {
        "id": "factory_automation_line",
        "title": "Linea de produccion automatizada",
        "category": ImageCategory.FACTORY,
        "tags": ["factory", "automation", "production", "line"],
        "keywords": ["factory automation", "production line"],
        "suggested_search": "factory automation production line",
        "license_type": LicenseType.UNSPLASH,
    },
    {
        "id": "control_panel_cabinet",
        "title": "Gabinete de control industrial",
        "category": ImageCategory.INDUSTRIAL_GENERAL,
        "tags": ["control panel", "cabinet", "electrical"],
        "keywords": ["control panel", "electrical cabinet"],
        "suggested_search": "industrial control panel cabinet",
        "license_type": LicenseType.CC_BY,
    },
]


class ImageBank:
    """
    Banco de imagenes local con registro persistente.

    Funcionalidades:
    - Registro de imagenes pre-definidas
    - Busqueda por categoria, tags, keywords
    - Gestion de imagenes descargadas
    - Metadata con licencias verificadas
    """

    DEFAULT_ASSETS_DIR = "assets/images"
    REGISTRY_FILE = "image_registry.json"

    def __init__(self, assets_dir: Optional[str] = None):
        self.assets_dir = Path(assets_dir or self.DEFAULT_ASSETS_DIR)
        self.registry_path = self.assets_dir / self.REGISTRY_FILE
        self._registry: Dict[str, ImageMetadata] = {}

        self._setup_directories()
        self._load_registry()
        self._init_predefined()

    def _setup_directories(self):
        """Crea la estructura de directorios"""
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Crear subdirectorios por categoria
        for category in ImageCategory:
            category_dir = self.assets_dir / category.value.replace("-", "_")
            category_dir.mkdir(exist_ok=True)

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

    def _init_predefined(self):
        """Inicializa imagenes pre-definidas si no existen"""
        for img_def in PREDEFINED_IMAGES:
            if img_def["id"] not in self._registry:
                # Crear metadata placeholder
                img = ImageMetadata(
                    id=img_def["id"],
                    title=img_def["title"],
                    source=ImageSource.LOCAL,
                    category=img_def["category"],
                    tags=img_def.get("tags", []),
                    keywords=img_def.get("keywords", []),
                    license=ImageLicense.from_type(img_def.get("license_type", LicenseType.UNKNOWN)),
                    extra={
                        "predefined": True,
                        "suggested_search": img_def.get("suggested_search"),
                        "needs_download": True,
                    }
                )
                self._registry[img.id] = img

        self._save_registry()

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

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadisticas del banco"""
        by_category = {}
        by_license = {}
        total_size = 0

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

        return {
            "total_images": len(self._registry),
            "by_category": by_category,
            "by_license": by_license,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "assets_dir": str(self.assets_dir),
        }

    def needs_download(self) -> List[ImageMetadata]:
        """Retorna imagenes que necesitan ser descargadas"""
        return [
            img for img in self._registry.values()
            if img.extra.get("needs_download") and not img.local_path
        ]

    def populate_from_search(
        self,
        sourcer,
        categories: Optional[List[ImageCategory]] = None,
        max_per_category: int = 5
    ) -> Dict[str, int]:
        """
        Pobla el banco buscando imagenes online para cada categoria.

        Args:
            sourcer: ImageSourcer para buscar online
            categories: Categorias a poblar (todas si None)
            max_per_category: Max imagenes por categoria

        Returns:
            Dict con cantidad de imagenes agregadas por categoria
        """
        from src.agent.services.images.metadata import ImageRequest

        categories = categories or list(ImageCategory)
        results = {}

        for category in categories:
            # Buscar imagenes que necesitan descarga en esta categoria
            pending = [
                img for img in self._registry.values()
                if img.category == category and img.extra.get("needs_download")
            ]

            if not pending:
                continue

            for img in pending[:max_per_category]:
                search_query = img.extra.get("suggested_search") or img.title

                request = ImageRequest(
                    query=search_query,
                    category=category,
                    max_results=1
                )

                search_result = sourcer.search(request, include_online=True)

                if search_result.images:
                    found = search_result.images[0]
                    # Actualizar metadata
                    img.source_url = found.source_url
                    img.source_page = found.source_page
                    img.author = found.author
                    img.width = found.width
                    img.height = found.height
                    img.extra["needs_download"] = False
                    img.extra["found_from"] = found.source.value

                    results[category.value] = results.get(category.value, 0) + 1

        self._save_registry()
        return results


# Singleton accessor
_bank_instance: Optional[ImageBank] = None


def get_image_bank(assets_dir: Optional[str] = None) -> ImageBank:
    """Obtiene la instancia singleton del banco de imagenes"""
    global _bank_instance
    if _bank_instance is None:
        _bank_instance = ImageBank(assets_dir)
    return _bank_instance

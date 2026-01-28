"""
sourcer.py - Servicio de busqueda de imagenes locales

Estrategia de sourcing:
1. Buscar en el banco local (dataset interno)
2. Filtrar por categoria y relevancia
3. Retornar imagenes disponibles del dataset local
"""
import time
import logging
from typing import Optional, List

from src.agent.media.images.metadata import (
    ImageMetadata,
    ImageSource,
    ImageCategory,
    ImageSearchResult,
    ImageRequest,
)
from src.agent.media.images.cache import ImageCache, get_image_cache


logger = logging.getLogger(__name__)


class ImageSourcer:
    """
    Servicio de busqueda de imagenes locales.

    Solo busca en el banco local de imagenes (dataset interno).
    No realiza busquedas online.
    """

    def __init__(self, cache: Optional[ImageCache] = None):
        self.cache = cache or get_image_cache()
        self._local_bank = None

    @property
    def local_bank(self):
        """Obtiene el banco local de imagenes"""
        if self._local_bank is None:
            from src.agent.media.images.bank import get_image_bank
            self._local_bank = get_image_bank()
        return self._local_bank

    def search(
        self,
        request: ImageRequest,
        include_online: bool = False  # Kept for API compatibility, ignored
    ) -> ImageSearchResult:
        _ = include_online  # Unused - local only
        """
        Busca imagenes en el banco local.

        Args:
            request: Solicitud de imagen
            include_online: Ignorado (solo busqueda local)

        Returns:
            Resultado de busqueda con imagenes locales
        """
        start = time.time()

        # Buscar en banco local
        local_results = self.local_bank.search(
            query=request.query,
            category=request.category,
            max_results=request.max_results
        )

        all_images: List[ImageMetadata] = list(local_results.images)

        # Filtrar por licencia si es necesario
        if request.require_commercial_license:
            all_images = [img for img in all_images if img.license.allows_commercial]

        if request.require_attribution_free:
            all_images = [img for img in all_images if not img.license.requires_attribution]

        # Filtrar por tamano minimo
        if request.min_width or request.min_height:
            all_images = [
                img for img in all_images
                if (not request.min_width or (img.width and img.width >= request.min_width))
                and (not request.min_height or (img.height and img.height >= request.min_height))
            ]

        # Ordenar por relevancia
        all_images.sort(key=lambda x: x.matches_query(request.query), reverse=True)

        # Limitar resultados
        all_images = all_images[:request.max_results]

        return ImageSearchResult(
            images=all_images,
            total=len(all_images),
            query=request.query,
            source=ImageSource.LOCAL,
            search_time_ms=(time.time() - start) * 1000,
        )

    def get_image(
        self,
        image: ImageMetadata,
        download: bool = False  # Kept for API compatibility, ignored
    ) -> Optional[str]:
        _ = download  # Unused - no online download
        """
        Obtiene la ruta local de una imagen.

        Args:
            image: Metadatos de la imagen
            download: Ignorado (no hay descarga online)

        Returns:
            Path local a la imagen o None
        """
        if image.local_path:
            return image.local_path
        return None

    def get_available_sources(self) -> List[str]:
        """Retorna lista de fuentes disponibles (solo local)"""
        return ["local"]


# Singleton accessor
_sourcer_instance: Optional[ImageSourcer] = None


def get_image_sourcer() -> ImageSourcer:
    """Obtiene la instancia singleton del sourcer"""
    global _sourcer_instance
    if _sourcer_instance is None:
        _sourcer_instance = ImageSourcer()
    return _sourcer_instance

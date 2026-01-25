"""
sourcer.py - Servicio de busqueda y obtencion de imagenes

Estrategia de sourcing:
1. Primero buscar en el banco local (dataset interno)
2. Si no hay resultados, buscar online en fuentes libres
3. Aplicar filtros de licencia segun requerimientos
4. Cachear resultados para futuras consultas

Fuentes online soportadas:
- Unsplash (API gratuita)
- Pexels (API gratuita)
- Pixabay (API gratuita)
- Wikimedia Commons (API publica)
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

import requests

from src.agent.media.images.metadata import (
    ImageMetadata,
    ImageSource,
    ImageCategory,
    ImageLicense,
    LicenseType,
    ImageSearchResult,
    ImageRequest,
)
from src.agent.media.images.cache import ImageCache, get_image_cache


logger = logging.getLogger(__name__)


# ============================================
# PROVEEDORES DE IMAGENES ONLINE
# ============================================

class ImageProvider(ABC):
    """Clase base para proveedores de imagenes"""

    @property
    @abstractmethod
    def source(self) -> ImageSource:
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        pass

    @abstractmethod
    def search(self, query: str, page: int = 1, per_page: int = 10) -> ImageSearchResult:
        pass

    def is_available(self) -> bool:
        """Verifica si el proveedor esta disponible"""
        if self.requires_api_key:
            return self.api_key is not None
        return True


class UnsplashProvider(ImageProvider):
    """Proveedor de imagenes de Unsplash"""

    BASE_URL = "https://api.unsplash.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UNSPLASH_ACCESS_KEY")

    @property
    def source(self) -> ImageSource:
        return ImageSource.UNSPLASH

    @property
    def requires_api_key(self) -> bool:
        return True

    def search(self, query: str, page: int = 1, per_page: int = 10) -> ImageSearchResult:
        if not self.api_key:
            return ImageSearchResult(query=query, source=self.source)

        start = time.time()
        try:
            response = requests.get(
                f"{self.BASE_URL}/search/photos",
                params={
                    "query": query,
                    "page": page,
                    "per_page": per_page,
                },
                headers={"Authorization": f"Client-ID {self.api_key}"},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            images = []
            for photo in data.get("results", []):
                images.append(ImageMetadata(
                    id=f"unsplash_{photo['id']}",
                    title=photo.get("description") or photo.get("alt_description") or "",
                    alt_text=photo.get("alt_description") or "",
                    source=ImageSource.UNSPLASH,
                    source_url=photo["urls"]["regular"],
                    source_page=photo["links"]["html"],
                    author=photo["user"]["name"],
                    author_url=photo["user"]["links"]["html"],
                    license=ImageLicense.from_type(LicenseType.UNSPLASH),
                    width=photo["width"],
                    height=photo["height"],
                    tags=[tag["title"] for tag in photo.get("tags", [])[:5]],
                    extra={
                        "urls": photo["urls"],
                        "color": photo.get("color"),
                        "likes": photo.get("likes", 0),
                    }
                ))

            return ImageSearchResult(
                images=images,
                total=data.get("total", 0),
                query=query,
                source=self.source,
                search_time_ms=(time.time() - start) * 1000,
                has_more=page < data.get("total_pages", 1),
                next_page=page + 1 if page < data.get("total_pages", 1) else None
            )

        except Exception as e:
            logger.error(f"Unsplash search error: {e}")
            return ImageSearchResult(query=query, source=self.source)


class PexelsProvider(ImageProvider):
    """Proveedor de imagenes de Pexels"""

    BASE_URL = "https://api.pexels.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")

    @property
    def source(self) -> ImageSource:
        return ImageSource.PEXELS

    @property
    def requires_api_key(self) -> bool:
        return True

    def search(self, query: str, page: int = 1, per_page: int = 10) -> ImageSearchResult:
        if not self.api_key:
            return ImageSearchResult(query=query, source=self.source)

        start = time.time()
        try:
            response = requests.get(
                f"{self.BASE_URL}/search",
                params={
                    "query": query,
                    "page": page,
                    "per_page": per_page,
                },
                headers={"Authorization": self.api_key},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            images = []
            for photo in data.get("photos", []):
                images.append(ImageMetadata(
                    id=f"pexels_{photo['id']}",
                    title=photo.get("alt") or "",
                    alt_text=photo.get("alt") or "",
                    source=ImageSource.PEXELS,
                    source_url=photo["src"]["large"],
                    source_page=photo["url"],
                    author=photo["photographer"],
                    author_url=photo["photographer_url"],
                    license=ImageLicense.from_type(LicenseType.PEXELS),
                    width=photo["width"],
                    height=photo["height"],
                    extra={
                        "src": photo["src"],
                        "avg_color": photo.get("avg_color"),
                    }
                ))

            return ImageSearchResult(
                images=images,
                total=data.get("total_results", 0),
                query=query,
                source=self.source,
                search_time_ms=(time.time() - start) * 1000,
                has_more=data.get("next_page") is not None,
                next_page=page + 1 if data.get("next_page") else None
            )

        except Exception as e:
            logger.error(f"Pexels search error: {e}")
            return ImageSearchResult(query=query, source=self.source)


class PixabayProvider(ImageProvider):
    """Proveedor de imagenes de Pixabay"""

    BASE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")

    @property
    def source(self) -> ImageSource:
        return ImageSource.PIXABAY

    @property
    def requires_api_key(self) -> bool:
        return True

    def search(self, query: str, page: int = 1, per_page: int = 10) -> ImageSearchResult:
        if not self.api_key:
            return ImageSearchResult(query=query, source=self.source)

        start = time.time()
        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "key": self.api_key,
                    "q": query,
                    "page": page,
                    "per_page": per_page,
                    "image_type": "photo",
                    "safesearch": "true",
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            images = []
            for hit in data.get("hits", []):
                images.append(ImageMetadata(
                    id=f"pixabay_{hit['id']}",
                    title=hit.get("tags", "").split(",")[0].strip(),
                    source=ImageSource.PIXABAY,
                    source_url=hit["largeImageURL"],
                    source_page=hit["pageURL"],
                    author=hit["user"],
                    author_url=f"https://pixabay.com/users/{hit['user']}-{hit['user_id']}/",
                    license=ImageLicense.from_type(LicenseType.PIXABAY),
                    width=hit["imageWidth"],
                    height=hit["imageHeight"],
                    tags=hit.get("tags", "").split(", "),
                    extra={
                        "downloads": hit.get("downloads", 0),
                        "likes": hit.get("likes", 0),
                        "views": hit.get("views", 0),
                    }
                ))

            total = data.get("totalHits", 0)
            total_pages = (total + per_page - 1) // per_page

            return ImageSearchResult(
                images=images,
                total=total,
                query=query,
                source=self.source,
                search_time_ms=(time.time() - start) * 1000,
                has_more=page < total_pages,
                next_page=page + 1 if page < total_pages else None
            )

        except Exception as e:
            logger.error(f"Pixabay search error: {e}")
            return ImageSearchResult(query=query, source=self.source)


class WikimediaProvider(ImageProvider):
    """Proveedor de imagenes de Wikimedia Commons (sin API key)"""

    BASE_URL = "https://commons.wikimedia.org/w/api.php"

    def __init__(self):
        pass

    @property
    def source(self) -> ImageSource:
        return ImageSource.WIKIMEDIA

    @property
    def requires_api_key(self) -> bool:
        return False

    def search(self, query: str, page: int = 1, per_page: int = 10) -> ImageSearchResult:
        start = time.time()
        try:
            # Buscar archivos (Wikimedia requires User-Agent header)
            response = requests.get(
                self.BASE_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrnamespace": "6",  # File namespace
                    "gsrsearch": f"filetype:bitmap {query}",
                    "gsrlimit": per_page,
                    "gsroffset": (page - 1) * per_page,
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                },
                headers={
                    "User-Agent": "FrEDie-ImageBank/1.0 (Educational Tutor; contact@fredie.edu)"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            images = []
            pages = data.get("query", {}).get("pages", {})

            for page_data in pages.values():
                if "imageinfo" not in page_data:
                    continue

                info = page_data["imageinfo"][0]
                extmeta = info.get("extmetadata", {})

                # Extraer licencia
                license_short = extmeta.get("LicenseShortName", {}).get("value", "")
                license_type = self._parse_license(license_short)

                # Extraer autor
                author = extmeta.get("Artist", {}).get("value", "")
                # Limpiar HTML del autor
                if "<" in author:
                    import re
                    author = re.sub(r'<[^>]+>', '', author).strip()

                images.append(ImageMetadata(
                    id=f"wikimedia_{page_data['pageid']}",
                    title=page_data.get("title", "").replace("File:", ""),
                    description=extmeta.get("ImageDescription", {}).get("value", ""),
                    source=ImageSource.WIKIMEDIA,
                    source_url=info["url"],
                    source_page=info.get("descriptionurl", ""),
                    author=author[:100] if author else None,
                    license=ImageLicense.from_type(license_type),
                    width=info.get("width"),
                    height=info.get("height"),
                    format=info.get("mime", "").split("/")[-1],
                ))

            return ImageSearchResult(
                images=images,
                total=len(images),  # Wikimedia no da total exacto
                query=query,
                source=self.source,
                search_time_ms=(time.time() - start) * 1000,
                has_more=len(images) == per_page,
                next_page=page + 1 if len(images) == per_page else None
            )

        except Exception as e:
            logger.error(f"Wikimedia search error: {e}")
            return ImageSearchResult(query=query, source=self.source)

    def _parse_license(self, license_str: str) -> LicenseType:
        """Parsea el string de licencia de Wikimedia"""
        license_lower = license_str.lower()
        if "cc0" in license_lower or "public domain" in license_lower:
            return LicenseType.CC0
        if "cc-by-sa" in license_lower or "cc by-sa" in license_lower:
            return LicenseType.CC_BY_SA
        if "cc-by-nc-sa" in license_lower:
            return LicenseType.CC_BY_NC_SA
        if "cc-by-nc" in license_lower:
            return LicenseType.CC_BY_NC
        if "cc-by" in license_lower or "cc by" in license_lower:
            return LicenseType.CC_BY
        return LicenseType.UNKNOWN


# ============================================
# SERVICIO PRINCIPAL DE SOURCING
# ============================================

class ImageSourcer:
    """
    Servicio principal de busqueda y obtencion de imagenes.

    Estrategia:
    1. Buscar en banco local primero
    2. Buscar online si no hay suficientes resultados
    3. Filtrar por licencia
    4. Cachear resultados
    """

    def __init__(
        self,
        cache: Optional[ImageCache] = None,
        enable_unsplash: bool = True,
        enable_pexels: bool = True,
        enable_pixabay: bool = True,
        enable_wikimedia: bool = True,
    ):
        self.cache = cache or get_image_cache()

        # Inicializar proveedores
        self.providers: Dict[ImageSource, ImageProvider] = {}

        if enable_unsplash:
            provider = UnsplashProvider()
            if provider.is_available():
                self.providers[ImageSource.UNSPLASH] = provider

        if enable_pexels:
            provider = PexelsProvider()
            if provider.is_available():
                self.providers[ImageSource.PEXELS] = provider

        if enable_pixabay:
            provider = PixabayProvider()
            if provider.is_available():
                self.providers[ImageSource.PIXABAY] = provider

        if enable_wikimedia:
            self.providers[ImageSource.WIKIMEDIA] = WikimediaProvider()

        # Referencia al banco local (se inicializa lazy)
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
        include_online: bool = True
    ) -> ImageSearchResult:
        """
        Busca imagenes segun la solicitud.

        Args:
            request: Solicitud de imagen
            include_online: Si buscar online si no hay suficientes locales

        Returns:
            Resultado de busqueda con imagenes
        """
        start = time.time()
        all_images: List[ImageMetadata] = []

        # 1. Buscar en banco local primero
        local_results = self.local_bank.search(
            query=request.query,
            category=request.category,
            max_results=request.max_results
        )
        all_images.extend(local_results.images)

        # 2. Si no hay suficientes, buscar online
        if include_online and len(all_images) < request.max_results:
            remaining = request.max_results - len(all_images)
            online_results = self._search_online(
                query=request.query,
                max_results=remaining,
                preferred_source=request.preferred_source
            )
            all_images.extend(online_results)

        # 3. Filtrar por licencia
        if request.require_commercial_license:
            all_images = [img for img in all_images if img.license.allows_commercial]

        if request.require_attribution_free:
            all_images = [img for img in all_images if not img.license.requires_attribution]

        # 4. Filtrar por tamano minimo
        if request.min_width or request.min_height:
            all_images = [
                img for img in all_images
                if (not request.min_width or (img.width and img.width >= request.min_width))
                and (not request.min_height or (img.height and img.height >= request.min_height))
            ]

        # 5. Ordenar por relevancia
        all_images.sort(key=lambda x: x.matches_query(request.query), reverse=True)

        # Limitar resultados
        all_images = all_images[:request.max_results]

        return ImageSearchResult(
            images=all_images,
            total=len(all_images),
            query=request.query,
            source=ImageSource.LOCAL if local_results.images else ImageSource.UNSPLASH,
            search_time_ms=(time.time() - start) * 1000,
        )

    def _search_online(
        self,
        query: str,
        max_results: int = 5,
        preferred_source: Optional[ImageSource] = None
    ) -> List[ImageMetadata]:
        """Busca en proveedores online"""
        results: List[ImageMetadata] = []

        # Determinar orden de proveedores
        if preferred_source and preferred_source in self.providers:
            provider_order = [preferred_source] + [
                s for s in self.providers.keys() if s != preferred_source
            ]
        else:
            # Orden por defecto: Unsplash > Pexels > Pixabay > Wikimedia
            provider_order = [
                ImageSource.UNSPLASH,
                ImageSource.PEXELS,
                ImageSource.PIXABAY,
                ImageSource.WIKIMEDIA,
            ]

        for source in provider_order:
            if len(results) >= max_results:
                break

            if source not in self.providers:
                continue

            provider = self.providers[source]
            try:
                search_result = provider.search(query, per_page=max_results - len(results))
                results.extend(search_result.images)
            except Exception as e:
                logger.warning(f"Error searching {source}: {e}")
                continue

        return results[:max_results]

    def get_image(
        self,
        image: ImageMetadata,
        download: bool = True
    ) -> Optional[str]:
        """
        Obtiene una imagen, descargandola si es necesario.

        Args:
            image: Metadatos de la imagen
            download: Si descargar y cachear si no esta local

        Returns:
            Path local a la imagen o None
        """
        # Si es local, ya tiene path
        if image.source == ImageSource.LOCAL and image.local_path:
            return image.local_path

        # Verificar cache
        if image.source_url:
            cached = self.cache.get(image.source_url)
            if cached:
                return str(cached[0])

            # Descargar si se solicita
            if download:
                path = self.cache.download_and_cache(image.source_url, image)
                if path:
                    return str(path)

        return None

    def get_available_sources(self) -> List[str]:
        """Retorna lista de fuentes disponibles"""
        sources = ["local"]
        sources.extend([s.value for s in self.providers.keys()])
        return sources


# Singleton accessor
_sourcer_instance: Optional[ImageSourcer] = None


def get_image_sourcer() -> ImageSourcer:
    """Obtiene la instancia singleton del sourcer"""
    global _sourcer_instance
    if _sourcer_instance is None:
        _sourcer_instance = ImageSourcer()
    return _sourcer_instance

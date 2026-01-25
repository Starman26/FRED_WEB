"""
cache.py - Sistema de cache para imagenes

Funcionalidades:
- Cache local en disco
- Expiracion configurable
- Limpieza automatica
- Metadata persistente
- Compresion opcional
"""
import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import threading
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import aiohttp
    import aiofiles
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

import requests

from src.agent.media.images.metadata import ImageMetadata, ImageSource


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuracion del cache de imagenes"""
    cache_dir: str = ".image_cache"
    max_size_mb: int = 500  # Tamano maximo del cache en MB
    default_expiry_days: int = 30  # Dias antes de que expire una imagen
    cleanup_threshold: float = 0.9  # Limpiar cuando llegue al 90% de capacidad
    compress_images: bool = True  # Comprimir imagenes grandes
    max_image_size: Tuple[int, int] = (1920, 1080)  # Tamano maximo al cachear
    jpeg_quality: int = 85  # Calidad JPEG al comprimir
    metadata_file: str = "cache_metadata.json"


class ImageCache:
    """
    Gestor de cache de imagenes con persistencia en disco.

    Caracteristicas:
    - Cache LRU con expiracion
    - Persistencia de metadatos
    - Compresion automatica
    - Thread-safe
    - Limpieza automatica
    """

    _instance: Optional["ImageCache"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[CacheConfig] = None):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[CacheConfig] = None):
        if self._initialized:
            return

        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.metadata_path = self.cache_dir / self.config.metadata_file
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}

        self._setup_cache_dir()
        self._load_metadata()
        self._initialized = True

    def _setup_cache_dir(self):
        """Crea el directorio de cache si no existe"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "images").mkdir(exist_ok=True)
        (self.cache_dir / "thumbnails").mkdir(exist_ok=True)

    def _load_metadata(self):
        """Carga metadatos del cache desde disco"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._metadata = data.get("images", {})
                    # Convertir timestamps
                    for key, meta in self._metadata.items():
                        if "cached_at" in meta:
                            meta["cached_at"] = datetime.fromisoformat(meta["cached_at"])
                        if "expires_at" in meta:
                            meta["expires_at"] = datetime.fromisoformat(meta["expires_at"])
                        if "last_access" in meta:
                            self._access_times[key] = datetime.fromisoformat(meta["last_access"])
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                self._metadata = {}

    def _save_metadata(self):
        """Guarda metadatos del cache a disco"""
        try:
            data = {"images": {}}
            for key, meta in self._metadata.items():
                meta_copy = meta.copy()
                if "cached_at" in meta_copy and isinstance(meta_copy["cached_at"], datetime):
                    meta_copy["cached_at"] = meta_copy["cached_at"].isoformat()
                if "expires_at" in meta_copy and isinstance(meta_copy["expires_at"], datetime):
                    meta_copy["expires_at"] = meta_copy["expires_at"].isoformat()
                if key in self._access_times:
                    meta_copy["last_access"] = self._access_times[key].isoformat()
                data["images"][key] = meta_copy

            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _generate_cache_key(self, url: str) -> str:
        """Genera una clave unica para la URL"""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _get_cache_path(self, cache_key: str, extension: str = "jpg") -> Path:
        """Obtiene la ruta del archivo en cache"""
        return self.cache_dir / "images" / f"{cache_key}.{extension}"

    def _get_thumbnail_path(self, cache_key: str) -> Path:
        """Obtiene la ruta del thumbnail"""
        return self.cache_dir / "thumbnails" / f"{cache_key}_thumb.jpg"

    def get(self, url: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Obtiene una imagen del cache.

        Returns:
            Tuple de (path, metadata) o None si no existe/expiro
        """
        cache_key = self._generate_cache_key(url)

        if cache_key not in self._metadata:
            return None

        meta = self._metadata[cache_key]

        # Verificar expiracion
        if "expires_at" in meta:
            if isinstance(meta["expires_at"], str):
                expires = datetime.fromisoformat(meta["expires_at"])
            else:
                expires = meta["expires_at"]

            if datetime.utcnow() > expires:
                self.remove(url)
                return None

        # Verificar que el archivo existe
        cache_path = Path(meta.get("local_path", ""))
        if not cache_path.exists():
            self.remove(url)
            return None

        # Actualizar tiempo de acceso
        self._access_times[cache_key] = datetime.utcnow()

        return cache_path, meta

    def put(
        self,
        url: str,
        content: bytes,
        metadata: Optional[ImageMetadata] = None,
        expiry_days: Optional[int] = None
    ) -> Path:
        """
        Almacena una imagen en el cache.

        Args:
            url: URL original de la imagen
            content: Contenido binario de la imagen
            metadata: Metadatos de la imagen
            expiry_days: Dias hasta expiracion

        Returns:
            Path al archivo cacheado
        """
        # Verificar espacio
        self._check_and_cleanup()

        cache_key = self._generate_cache_key(url)

        # Detectar formato
        extension = self._detect_format(content, url)
        cache_path = self._get_cache_path(cache_key, extension)

        # Comprimir si es necesario
        if self.config.compress_images and PIL_AVAILABLE:
            content = self._compress_image(content, extension)

        # Guardar archivo
        with open(cache_path, "wb") as f:
            f.write(content)

        # Generar thumbnail
        if PIL_AVAILABLE:
            self._create_thumbnail(cache_path, cache_key)

        # Guardar metadata
        now = datetime.utcnow()
        expiry = expiry_days or self.config.default_expiry_days

        meta_dict = {
            "url": url,
            "local_path": str(cache_path),
            "cached_at": now,
            "expires_at": now + timedelta(days=expiry),
            "size_bytes": len(content),
            "format": extension,
            "hash": hashlib.sha256(content).hexdigest(),
        }

        if metadata:
            meta_dict.update({
                "title": metadata.title,
                "source": metadata.source.value if hasattr(metadata.source, 'value') else metadata.source,
                "author": metadata.author,
                "license_type": metadata.license.type.value if hasattr(metadata.license.type, 'value') else str(metadata.license.type),
                "category": metadata.category.value if hasattr(metadata.category, 'value') else metadata.category,
                "tags": metadata.tags,
            })

        self._metadata[cache_key] = meta_dict
        self._access_times[cache_key] = now
        self._save_metadata()

        return cache_path

    def _detect_format(self, content: bytes, url: str) -> str:
        """Detecta el formato de la imagen"""
        # Por magic bytes
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return "png"
        if content[:2] == b'\xff\xd8':
            return "jpg"
        if content[:4] == b'GIF8':
            return "gif"
        if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return "webp"
        if b'<svg' in content[:100]:
            return "svg"

        # Por extension de URL
        url_lower = url.lower()
        for ext in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg']:
            if ext in url_lower:
                return 'jpg' if ext == 'jpeg' else ext

        return "jpg"  # Default

    def _compress_image(self, content: bytes, extension: str) -> bytes:
        """Comprime la imagen si es muy grande"""
        if extension == "svg":
            return content  # No comprimir SVG

        try:
            from io import BytesIO
            from PIL import Image

            img = Image.open(BytesIO(content))

            # Redimensionar si es muy grande
            max_w, max_h = self.config.max_image_size
            if img.width > max_w or img.height > max_h:
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            # Convertir a RGB si es necesario
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Guardar comprimido
            output = BytesIO()
            img.save(output, format='JPEG', quality=self.config.jpeg_quality, optimize=True)
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Error compressing image: {e}")
            return content

    def _create_thumbnail(self, image_path: Path, cache_key: str, size: Tuple[int, int] = (200, 200)):
        """Crea un thumbnail de la imagen"""
        try:
            from PIL import Image

            thumb_path = self._get_thumbnail_path(cache_key)
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            img.save(thumb_path, format='JPEG', quality=75)
        except Exception as e:
            logger.warning(f"Error creating thumbnail: {e}")

    def remove(self, url: str):
        """Elimina una imagen del cache"""
        cache_key = self._generate_cache_key(url)

        if cache_key in self._metadata:
            meta = self._metadata[cache_key]
            # Eliminar archivo
            cache_path = Path(meta.get("local_path", ""))
            if cache_path.exists():
                cache_path.unlink()
            # Eliminar thumbnail
            thumb_path = self._get_thumbnail_path(cache_key)
            if thumb_path.exists():
                thumb_path.unlink()
            # Eliminar metadata
            del self._metadata[cache_key]
            self._access_times.pop(cache_key, None)
            self._save_metadata()

    def clear(self):
        """Limpia todo el cache"""
        for key in list(self._metadata.keys()):
            meta = self._metadata[key]
            cache_path = Path(meta.get("local_path", ""))
            if cache_path.exists():
                cache_path.unlink()

        self._metadata.clear()
        self._access_times.clear()
        self._save_metadata()

    def _get_cache_size_mb(self) -> float:
        """Calcula el tamano total del cache en MB"""
        total = 0
        for meta in self._metadata.values():
            total += meta.get("size_bytes", 0)
        return total / (1024 * 1024)

    def _check_and_cleanup(self):
        """Verifica el tamano del cache y limpia si es necesario"""
        current_size = self._get_cache_size_mb()
        threshold = self.config.max_size_mb * self.config.cleanup_threshold

        if current_size < threshold:
            return

        logger.info(f"Cache cleanup triggered: {current_size:.1f}MB / {self.config.max_size_mb}MB")

        # Ordenar por ultimo acceso (LRU)
        sorted_keys = sorted(
            self._access_times.keys(),
            key=lambda k: self._access_times.get(k, datetime.min)
        )

        # Eliminar hasta llegar al 50% de capacidad
        target_size = self.config.max_size_mb * 0.5
        for key in sorted_keys:
            if current_size <= target_size:
                break

            if key in self._metadata:
                meta = self._metadata[key]
                size_mb = meta.get("size_bytes", 0) / (1024 * 1024)
                self.remove(meta.get("url", ""))
                current_size -= size_mb

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadisticas del cache"""
        total_size = self._get_cache_size_mb()
        return {
            "total_images": len(self._metadata),
            "total_size_mb": round(total_size, 2),
            "max_size_mb": self.config.max_size_mb,
            "usage_percent": round((total_size / self.config.max_size_mb) * 100, 1),
            "cache_dir": str(self.cache_dir),
        }

    def download_and_cache(
        self,
        url: str,
        metadata: Optional[ImageMetadata] = None,
        timeout: int = 30
    ) -> Optional[Path]:
        """
        Descarga una imagen y la almacena en cache.

        Returns:
            Path al archivo cacheado o None si falla
        """
        # Verificar si ya esta en cache
        cached = self.get(url)
        if cached:
            return cached[0]

        try:
            # Descargar
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "FrEDie-ImageBank/1.0"}
            )
            response.raise_for_status()

            # Cachear
            return self.put(url, response.content, metadata)

        except Exception as e:
            logger.error(f"Error downloading image {url}: {e}")
            return None


# Singleton accessor
_cache_instance: Optional[ImageCache] = None


def get_image_cache(config: Optional[CacheConfig] = None) -> ImageCache:
    """Obtiene la instancia singleton del cache"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ImageCache(config)
    return _cache_instance

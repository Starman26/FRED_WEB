"""
Image Bank Service - Sistema de gestion de imagenes para el tutor

Componentes:
- ImageMetadata: Schema de metadatos de imagen
- ImageCache: Cache local de imagenes
- ImageSourcer: Servicio de busqueda de imagenes (local + online)
- LicenseManager: Validacion y atribucion de licencias
- ImageBank: Registro de imagenes locales del dataset
"""

from src.agent.services.images.metadata import (
    ImageMetadata,
    ImageLicense,
    ImageSource,
    ImageCategory,
    LicenseType,
)
from src.agent.services.images.cache import ImageCache
from src.agent.services.images.sourcer import ImageSourcer, get_image_sourcer
from src.agent.services.images.bank import ImageBank, get_image_bank

__all__ = [
    "ImageMetadata",
    "ImageLicense",
    "ImageSource",
    "ImageCategory",
    "LicenseType",
    "ImageCache",
    "ImageSourcer",
    "get_image_sourcer",
    "ImageBank",
    "get_image_bank",
]

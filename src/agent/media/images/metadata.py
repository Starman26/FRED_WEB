"""
metadata.py - Schema de metadatos para imagenes del banco de imagenes

Define:
- Tipos de licencia soportados
- Categorias de imagenes
- Fuentes de imagenes
- Schema completo de metadatos
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from datetime import datetime
import hashlib


class LicenseType(str, Enum):
    """Tipos de licencia soportados"""
    # Creative Commons
    CC0 = "cc0"  # Public Domain
    CC_BY = "cc-by"  # Attribution
    CC_BY_SA = "cc-by-sa"  # Attribution-ShareAlike
    CC_BY_NC = "cc-by-nc"  # Attribution-NonCommercial
    CC_BY_NC_SA = "cc-by-nc-sa"  # Attribution-NonCommercial-ShareAlike
    CC_BY_ND = "cc-by-nd"  # Attribution-NoDerivs
    CC_BY_NC_ND = "cc-by-nc-nd"  # Attribution-NonCommercial-NoDerivs

    # Otras licencias libres
    PUBLIC_DOMAIN = "public-domain"
    UNSPLASH = "unsplash"  # Unsplash License
    PEXELS = "pexels"  # Pexels License
    PIXABAY = "pixabay"  # Pixabay License

    # Licencias propietarias/educativas
    EDUCATIONAL_FAIR_USE = "educational-fair-use"
    PROPRIETARY_WITH_PERMISSION = "proprietary-permission"
    INTERNAL_ONLY = "internal"

    # Desconocido
    UNKNOWN = "unknown"


class ImageSource(str, Enum):
    """Fuentes de imagenes"""
    LOCAL = "local"  # Imagen local del dataset
    UNSPLASH = "unsplash"  # unsplash.com
    PEXELS = "pexels"  # pexels.com
    PIXABAY = "pixabay"  # pixabay.com
    WIKIMEDIA = "wikimedia"  # Wikimedia Commons
    GOOGLE = "google"  # Google Images (solo referencia)
    CUSTOM_URL = "custom_url"  # URL personalizada
    GENERATED = "generated"  # Imagen generada (AI, diagrams)
    SCREENSHOT = "screenshot"  # Captura de pantalla


class ImageCategory(str, Enum):
    """Categorias de imagenes para el tutor industrial"""
    # PLCs
    PLC_GENERAL = "plc-general"
    PLC_SIEMENS = "plc-siemens"
    PLC_ALLEN_BRADLEY = "plc-allen-bradley"
    PLC_DIAGRAM = "plc-diagram"
    PLC_WIRING = "plc-wiring"

    # Cobots/Robots
    COBOT_GENERAL = "cobot-general"
    COBOT_UR = "cobot-ur"  # Universal Robots
    COBOT_FANUC = "cobot-fanuc"
    ROBOT_ARM = "robot-arm"

    # Comunicaciones
    PROFINET = "profinet"
    PROFIBUS = "profibus"
    ETHERNET = "ethernet"
    NETWORK_DIAGRAM = "network-diagram"

    # HMI
    HMI_SCREEN = "hmi-screen"
    HMI_DESIGN = "hmi-design"
    SCADA = "scada"

    # Programacion
    LADDER_LOGIC = "ladder-logic"
    FBD_DIAGRAM = "fbd-diagram"
    SCL_CODE = "scl-code"
    FLOWCHART = "flowchart"

    # Seguridad
    SAFETY_EQUIPMENT = "safety-equipment"
    SAFETY_DIAGRAM = "safety-diagram"
    SAFETY_SIGN = "safety-sign"

    # General
    INDUSTRIAL_GENERAL = "industrial-general"
    FACTORY = "factory"
    AUTOMATION = "automation"
    DIAGRAM = "diagram"
    ICON = "icon"
    OTHER = "other"


class ImageLicense(BaseModel):
    """Informacion de licencia de una imagen"""
    type: LicenseType = Field(default=LicenseType.UNKNOWN)
    name: str = Field(default="Unknown License")
    url: Optional[str] = Field(default=None, description="URL de la licencia")
    requires_attribution: bool = Field(default=True)
    allows_commercial: bool = Field(default=False)
    allows_modification: bool = Field(default=False)
    attribution_text: Optional[str] = Field(
        default=None,
        description="Texto de atribucion requerido"
    )

    @classmethod
    def from_type(cls, license_type: LicenseType) -> "ImageLicense":
        """Crea una licencia con valores por defecto segun el tipo"""
        license_info = LICENSE_DEFAULTS.get(license_type, {})
        return cls(type=license_type, **license_info)

    def get_attribution_html(self, author: str = None, source_url: str = None) -> str:
        """Genera HTML de atribucion"""
        if not self.requires_attribution:
            return ""

        parts = []
        if author:
            parts.append(f"Photo by {author}")
        if source_url:
            parts.append(f'from <a href="{source_url}" target="_blank">source</a>')
        if self.url:
            parts.append(f'(<a href="{self.url}" target="_blank">{self.name}</a>)')

        return " ".join(parts) if parts else ""


# Valores por defecto para cada tipo de licencia
LICENSE_DEFAULTS = {
    LicenseType.CC0: {
        "name": "CC0 1.0 Universal (Public Domain)",
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "requires_attribution": False,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.CC_BY: {
        "name": "Creative Commons Attribution 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "requires_attribution": True,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.CC_BY_SA: {
        "name": "Creative Commons Attribution-ShareAlike 4.0",
        "url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "requires_attribution": True,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.CC_BY_NC: {
        "name": "Creative Commons Attribution-NonCommercial 4.0",
        "url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "requires_attribution": True,
        "allows_commercial": False,
        "allows_modification": True,
    },
    LicenseType.UNSPLASH: {
        "name": "Unsplash License",
        "url": "https://unsplash.com/license",
        "requires_attribution": False,  # Not required but appreciated
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.PEXELS: {
        "name": "Pexels License",
        "url": "https://www.pexels.com/license/",
        "requires_attribution": False,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.PIXABAY: {
        "name": "Pixabay License",
        "url": "https://pixabay.com/service/license/",
        "requires_attribution": False,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.PUBLIC_DOMAIN: {
        "name": "Public Domain",
        "requires_attribution": False,
        "allows_commercial": True,
        "allows_modification": True,
    },
    LicenseType.EDUCATIONAL_FAIR_USE: {
        "name": "Educational Fair Use",
        "requires_attribution": True,
        "allows_commercial": False,
        "allows_modification": False,
    },
    LicenseType.INTERNAL_ONLY: {
        "name": "Internal Use Only",
        "requires_attribution": False,
        "allows_commercial": False,
        "allows_modification": False,
    },
}


class ImageMetadata(BaseModel):
    """
    Metadatos completos de una imagen.

    Incluye:
    - Identificacion unica
    - Informacion de la fuente
    - Licencia y atribucion
    - Categorias y tags
    - Cache info
    """
    # Identificacion
    id: str = Field(..., description="ID unico de la imagen")
    hash: Optional[str] = Field(default=None, description="SHA256 del contenido")

    # Contenido
    title: str = Field(default="", description="Titulo descriptivo")
    description: Optional[str] = Field(default=None, description="Descripcion detallada")
    alt_text: str = Field(default="", description="Texto alternativo para accesibilidad")

    # Fuente
    source: ImageSource = Field(default=ImageSource.LOCAL)
    source_url: Optional[str] = Field(default=None, description="URL original de la imagen")
    source_page: Optional[str] = Field(default=None, description="URL de la pagina donde se encontro")
    author: Optional[str] = Field(default=None, description="Autor/fotografo")
    author_url: Optional[str] = Field(default=None, description="URL del perfil del autor")

    # Licencia
    license: ImageLicense = Field(default_factory=lambda: ImageLicense())

    # Clasificacion
    category: ImageCategory = Field(default=ImageCategory.OTHER)
    tags: List[str] = Field(default_factory=list, description="Tags para busqueda")
    keywords: List[str] = Field(default_factory=list, description="Palabras clave SEO")

    # Dimensiones y formato
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    format: Optional[str] = Field(default=None, description="jpg, png, webp, svg, etc")
    size_bytes: Optional[int] = Field(default=None)

    # Cache
    local_path: Optional[str] = Field(default=None, description="Ruta local en cache")
    cached_at: Optional[datetime] = Field(default=None)
    cache_expires: Optional[datetime] = Field(default=None)

    # Metadatos adicionales
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Calidad estimada 0-1")
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevancia al contexto")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def generate_id(self) -> str:
        """Genera un ID unico basado en la fuente y URL"""
        source_str = f"{self.source.value}:{self.source_url or self.title}"
        return hashlib.sha256(source_str.encode()).hexdigest()[:16]

    def get_attribution(self) -> str:
        """Genera texto de atribucion si es requerido"""
        if not self.license.requires_attribution:
            return ""

        parts = []
        if self.title:
            parts.append(f'"{self.title}"')
        if self.author:
            parts.append(f"by {self.author}")
        if self.license.name:
            parts.append(f"({self.license.name})")

        return " ".join(parts)

    def get_attribution_html(self) -> str:
        """Genera HTML de atribucion"""
        return self.license.get_attribution_html(
            author=self.author,
            source_url=self.source_url
        )

    def is_cached(self) -> bool:
        """Verifica si la imagen esta en cache y no ha expirado"""
        if not self.local_path:
            return False
        if self.cache_expires and datetime.utcnow() > self.cache_expires:
            return False
        return True

    def matches_query(self, query: str) -> float:
        """Calcula score de coincidencia con una query de busqueda"""
        query_lower = query.lower()
        score = 0.0

        # Coincidencia en titulo
        if query_lower in self.title.lower():
            score += 0.4

        # Coincidencia en tags
        for tag in self.tags:
            if query_lower in tag.lower():
                score += 0.2
                break

        # Coincidencia en keywords
        for kw in self.keywords:
            if query_lower in kw.lower():
                score += 0.15
                break

        # Coincidencia en descripcion
        if self.description and query_lower in self.description.lower():
            score += 0.1

        # Coincidencia en categoria
        if query_lower in self.category.value:
            score += 0.15

        return min(score, 1.0)

    class Config:
        use_enum_values = True


class ImageSearchResult(BaseModel):
    """Resultado de busqueda de imagenes"""
    images: List[ImageMetadata] = Field(default_factory=list)
    total: int = Field(default=0)
    query: str = Field(default="")
    source: ImageSource = Field(default=ImageSource.LOCAL)
    search_time_ms: float = Field(default=0.0)
    has_more: bool = Field(default=False)
    next_page: Optional[int] = Field(default=None)


class ImageRequest(BaseModel):
    """Solicitud de imagen para el tutor"""
    query: str = Field(..., description="Busqueda o descripcion de la imagen necesaria")
    category: Optional[ImageCategory] = Field(default=None)
    preferred_source: Optional[ImageSource] = Field(default=None)
    require_commercial_license: bool = Field(default=False)
    require_attribution_free: bool = Field(default=False)
    min_width: Optional[int] = Field(default=None)
    min_height: Optional[int] = Field(default=None)
    max_results: int = Field(default=5)
    use_cache: bool = Field(default=True)

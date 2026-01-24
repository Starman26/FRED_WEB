import os
import sys
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

from src.api.models import HealthResponse, ErrorResponse
from src.api.routes.chat import router as chat_router
from src.agent.services import init_services, get_services_status

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("api")


# LIFECYCLE
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando API...")
    
    # Iniciar servicios 
    services = init_services()
    logger.info(f"Servicios: {services}")
    
    yield
    
    # Shutdown
    logger.info("Cerrando API...")


# APP
app = FastAPI(
    title="ATLAS Agent API",
    description="""
API REST para interactuar con el agente.
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Chat", "description": "Endpoints de conversación con el agente"},
        {"name": "Health", "description": "Estado de la API y servicios"},
    ]
)


# MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    
    return response


# EXCEPTION HANDLERS
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        ).model_dump()
    )


# ROUTERS
app.include_router(chat_router)


# HEALTH ENDPOINTS
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    services = get_services_status()
    
    return HealthResponse(
        status="ok" if services.get("initialized") else "degraded",
        services={
            "supabase": services.get("supabase_connected", False),
            "embeddings": services.get("embeddings_ready", False),
        }
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Información básica de la API"""
    return {
        "name": "ATLAS Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )

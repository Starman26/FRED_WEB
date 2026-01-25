"""
test_image_bank.py - Test del sistema de banco de imagenes

Run with: python test_image_bank.py

Prueba:
1. Metadata y licencias
2. Cache de imagenes
3. Banco local
4. Sourcer (busqueda)
"""
import sys
import os
sys.path.insert(0, '.')

# Configurar directorio de cache para tests
os.environ.setdefault("IMAGE_CACHE_DIR", ".test_image_cache")


def test_metadata():
    """Prueba el schema de metadatos"""
    print("\n" + "="*60)
    print("TEST 1: Image Metadata Schema")
    print("="*60)

    from src.agent.media.images.metadata import (
        ImageMetadata,
        ImageLicense,
        LicenseType,
        ImageSource,
        ImageCategory,
        ImageRequest,
    )

    # Crear licencia
    license = ImageLicense.from_type(LicenseType.UNSPLASH)
    print(f"[OK] License created: {license.name}")
    print(f"     Requires attribution: {license.requires_attribution}")
    print(f"     Allows commercial: {license.allows_commercial}")

    # Crear metadata
    img = ImageMetadata(
        id="test_img_001",
        title="Test PLC Image",
        source=ImageSource.UNSPLASH,
        source_url="https://example.com/image.jpg",
        author="Test Author",
        license=license,
        category=ImageCategory.PLC_SIEMENS,
        tags=["plc", "siemens", "s7-1200"],
        width=1920,
        height=1080,
    )

    print(f"\n[OK] ImageMetadata created")
    print(f"     ID: {img.id}")
    print(f"     Title: {img.title}")
    print(f"     Category: {img.category}")
    print(f"     Tags: {img.tags}")

    # Test query matching
    score = img.matches_query("siemens plc")
    print(f"\n[OK] Query matching: 'siemens plc' -> score {score:.2f}")

    # Test request
    request = ImageRequest(
        query="siemens s7-1200",
        category=ImageCategory.PLC_SIEMENS,
        require_commercial_license=True,
        max_results=5
    )
    print(f"\n[OK] ImageRequest created")
    print(f"     Query: {request.query}")
    print(f"     Category: {request.category}")


def test_cache():
    """Prueba el cache de imagenes"""
    print("\n" + "="*60)
    print("TEST 2: Image Cache")
    print("="*60)

    from src.agent.media.images.cache import ImageCache, CacheConfig

    # Crear cache con config de test
    config = CacheConfig(
        cache_dir=".test_image_cache",
        max_size_mb=50,
        default_expiry_days=7
    )

    cache = ImageCache(config)
    print(f"[OK] Cache initialized")
    print(f"     Directory: {cache.cache_dir}")

    # Test put/get
    test_content = b"fake image content for testing"
    test_url = "https://example.com/test_image.jpg"

    path = cache.put(test_url, test_content)
    print(f"\n[OK] Image cached at: {path}")

    # Retrieve
    result = cache.get(test_url)
    if result:
        cached_path, meta = result
        print(f"[OK] Retrieved from cache: {cached_path}")
        print(f"     Size: {meta.get('size_bytes')} bytes")
        print(f"     Format: {meta.get('format')}")
    else:
        print("[FAIL] Could not retrieve from cache")

    # Stats
    stats = cache.get_stats()
    print(f"\n[OK] Cache stats:")
    print(f"     Total images: {stats['total_images']}")
    print(f"     Size: {stats['total_size_mb']} MB")

    # Cleanup
    cache.remove(test_url)
    print("\n[OK] Test image removed from cache")


def test_bank():
    """Prueba el banco de imagenes local"""
    print("\n" + "="*60)
    print("TEST 3: Image Bank (Local Dataset)")
    print("="*60)

    from src.agent.media.images.bank import ImageBank, get_image_bank
    from src.agent.media.images.metadata import ImageCategory

    bank = get_image_bank(".test_assets/images")
    print(f"[OK] Bank initialized")
    print(f"     Assets dir: {bank.assets_dir}")

    # Stats
    stats = bank.get_stats()
    print(f"\n[OK] Bank stats:")
    print(f"     Total images: {stats['total_images']}")
    print(f"     Categories: {list(stats['by_category'].keys())}")

    # Search
    result = bank.search(query="plc siemens", max_results=5)
    print(f"\n[OK] Search 'plc siemens':")
    print(f"     Found: {result.total} images")
    for img in result.images[:3]:
        print(f"     - {img.title} (score: {img.relevance_score:.2f})")

    # Get by category
    plc_images = bank.get_by_category(ImageCategory.PLC_SIEMENS)
    print(f"\n[OK] PLC_SIEMENS category: {len(plc_images)} images")

    # Needs download
    pending = bank.needs_download()
    print(f"\n[OK] Images pending download: {len(pending)}")


def test_sourcer():
    """Prueba el servicio de sourcing"""
    print("\n" + "="*60)
    print("TEST 4: Image Sourcer")
    print("="*60)

    from src.agent.media.images.sourcer import ImageSourcer, get_image_sourcer
    from src.agent.media.images.metadata import ImageRequest, ImageCategory

    sourcer = get_image_sourcer()
    print(f"[OK] Sourcer initialized")
    print(f"     Available sources: {sourcer.get_available_sources()}")

    # Search local only
    request = ImageRequest(
        query="plc industrial",
        max_results=3
    )

    result = sourcer.search(request, include_online=False)
    print(f"\n[OK] Local search 'plc industrial':")
    print(f"     Found: {result.total} images")
    print(f"     Time: {result.search_time_ms:.1f}ms")

    # Note about online search
    print("\n[INFO] Online search requires API keys:")
    print("       - UNSPLASH_ACCESS_KEY for Unsplash")
    print("       - PEXELS_API_KEY for Pexels")
    print("       - PIXABAY_API_KEY for Pixabay")
    print("       Wikimedia works without API key")

    # Test Wikimedia (no API key needed)
    if "wikimedia" in sourcer.get_available_sources():
        from src.agent.media.images.sourcer import WikimediaProvider
        wiki = WikimediaProvider()
        result = wiki.search("PLC Siemens", per_page=3)
        print(f"\n[OK] Wikimedia search 'PLC Siemens':")
        print(f"     Found: {len(result.images)} images")
        for img in result.images[:2]:
            print(f"     - {img.title[:50]}...")
            print(f"       License: {img.license.name}")


def test_full_workflow():
    """Prueba el flujo completo"""
    print("\n" + "="*60)
    print("TEST 5: Full Workflow")
    print("="*60)

    from src.agent.media.images import (
        ImageSourcer,
        ImageBank,
        ImageCache,
        ImageMetadata,
        ImageCategory,
        LicenseType,
        get_image_sourcer,
        get_image_bank,
    )
    from src.agent.media.images.metadata import ImageRequest

    # 1. Get sourcer
    sourcer = get_image_sourcer()

    # 2. Create request
    request = ImageRequest(
        query="industrial robot arm",
        category=ImageCategory.COBOT_GENERAL,
        require_commercial_license=False,
        max_results=3
    )

    print(f"[WORKFLOW] Searching for: {request.query}")
    print(f"           Category: {request.category}")

    # 3. Search (local first, then online if needed)
    result = sourcer.search(request, include_online=True)

    print(f"\n[RESULT] Found {result.total} images")
    print(f"         Search time: {result.search_time_ms:.1f}ms")

    for i, img in enumerate(result.images, 1):
        print(f"\n  Image {i}:")
        print(f"    Title: {img.title[:60]}...")
        print(f"    Source: {img.source}")
        print(f"    License: {img.license.name}")
        print(f"    Attribution required: {img.license.requires_attribution}")
        if img.source_url:
            print(f"    URL: {img.source_url[:60]}...")

    print("\n[OK] Full workflow completed successfully!")


def cleanup():
    """Limpia archivos de test"""
    import shutil
    for path in [".test_image_cache", ".test_assets"]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[CLEANUP] Removed {path}")


if __name__ == "__main__":
    print("="*60)
    print("IMAGE BANK SYSTEM TEST SUITE")
    print("="*60)

    try:
        test_metadata()
        test_cache()
        test_bank()
        test_sourcer()
        test_full_workflow()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)

        # Cleanup
        print("\n[INFO] Cleaning up test files...")
        cleanup()

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

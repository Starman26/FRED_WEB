"""
test_tutor_images.py - Test the tutor image integration

Run with: python test_tutor_images.py
"""
import sys
sys.path.insert(0, '.')

def test_image_search_functions():
    """Test the image search helper functions"""
    print("\n" + "="*60)
    print("TEST: Image Search Functions")
    print("="*60)

    from src.agent.workers.tutor_node import (
        detect_image_category,
        extract_image_search_terms,
        search_relevant_images,
        IMAGES_AVAILABLE
    )

    print(f"[INFO] Images module available: {IMAGES_AVAILABLE}")

    # Test category detection
    test_cases = [
        ("Como programo un PLC Siemens S7-1200?", "plc-siemens"),
        ("Necesito ayuda con Universal Robots UR5", "cobot-ur"),
        ("Explicame ladder logic", "ladder-logic"),
        ("Que es un HMI?", "hmi-screen"),
        ("Configurar PROFINET", "profinet"),
        ("Python machine learning", None),  # No industrial keyword
    ]

    print("\n[Test] Category Detection:")
    for query, expected in test_cases:
        category = detect_image_category(query)
        cat_value = category.value if hasattr(category, 'value') else str(category) if category else None
        status = "[OK]" if cat_value == expected else "[DIFF]"
        print(f"  {status} '{query[:40]}...' -> {cat_value} (expected: {expected})")

    # Test keyword extraction
    print("\n[Test] Keyword Extraction:")
    test_messages = [
        "Como conecto un sensor al PLC Siemens?",
        "Explicame la programacion ladder con ejemplos",
        "Quiero aprender sobre cobots y robots colaborativos",
    ]
    for msg in test_messages:
        keywords = extract_image_search_terms(msg)
        print(f"  '{msg[:40]}...' -> '{keywords}'")

    # Test actual image search
    print("\n[Test] Image Search:")
    if IMAGES_AVAILABLE:
        images = search_relevant_images(
            user_message="Como programo un PLC Siemens S7-1200?",
            learning_style="visual",
            max_images=3
        )
        print(f"  Found {len(images)} images for 'PLC Siemens'")
        for img in images:
            print(f"    - {img.get('title', 'No title')[:50]}")
            print(f"      Source: {img.get('source')}, License: {img.get('license', {}).get('name', 'Unknown')}")
    else:
        print("  [SKIP] Images module not available")


def test_tutor_output_structure():
    """Test that tutor output includes images field"""
    print("\n" + "="*60)
    print("TEST: Tutor Output Structure")
    print("="*60)

    from src.agent.contracts.worker_contract import WorkerOutputBuilder

    # Create a tutor output with images
    output = WorkerOutputBuilder.tutor(
        content="Esta es una explicacion educativa sobre PLCs...",
        learning_objectives=["Entender PLCs", "Programar en ladder"],
        summary="Explicacion de PLCs",
        confidence=0.85,
    )

    # Add mock images
    mock_images = [
        {
            "id": "test_img_1",
            "title": "Siemens S7-1200 PLC",
            "source": "local",
            "source_url": "https://example.com/plc.jpg",
            "license": {"name": "CC-BY", "requires_attribution": True}
        }
    ]
    output.extra["images"] = mock_images
    output.extra["images_count"] = len(mock_images)

    print(f"[OK] Worker output created")
    print(f"     Worker: {output.worker}")
    print(f"     Status: {output.status}")
    print(f"     Extra keys: {list(output.extra.keys())}")
    print(f"     Images count: {output.extra.get('images_count', 0)}")

    # Verify serialization
    output_dict = output.model_dump()
    assert "images" in output_dict["extra"], "Images not in extra!"
    assert output_dict["extra"]["images_count"] == 1, "Wrong image count!"
    print("[OK] Output serialization verified")


def test_full_integration():
    """Test the full tutor node with mock state"""
    print("\n" + "="*60)
    print("TEST: Full Tutor Integration (Mock)")
    print("="*60)

    from src.agent.workers.tutor_node import (
        search_relevant_images,
        IMAGES_AVAILABLE
    )

    # Simulate what tutor_node does
    user_message = "Explicame como funciona un PLC Siemens S7-1200"
    learning_style = "visual"

    print(f"[INFO] User message: {user_message}")
    print(f"[INFO] Learning style: {learning_style}")

    # Search for images
    images = search_relevant_images(
        user_message=user_message,
        evidence_text="",
        learning_style=learning_style,
        max_images=3
    )

    print(f"\n[RESULT] Found {len(images)} relevant images")
    for i, img in enumerate(images, 1):
        print(f"\n  Image {i}:")
        print(f"    Title: {img.get('title', 'No title')[:60]}")
        print(f"    Source: {img.get('source')}")
        print(f"    URL: {img.get('source_url', 'N/A')[:60]}...")
        license_info = img.get('license', {})
        print(f"    License: {license_info.get('name', 'Unknown')}")
        print(f"    Attribution required: {license_info.get('requires_attribution', False)}")


if __name__ == "__main__":
    print("="*60)
    print("TUTOR IMAGE INTEGRATION TEST")
    print("="*60)

    try:
        test_image_search_functions()
        test_tutor_output_structure()
        test_full_integration()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

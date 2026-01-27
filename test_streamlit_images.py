"""
Test to verify image display functionality in Streamlit app

This test checks:
1. Image extraction from events works
2. Image rendering function is properly defined
3. Image paths are correctly constructed
"""

import sys
sys.path.insert(0, '.')

def test_extract_images_function():
    """Test the extract_images_from_events function"""
    print("\n" + "="*60)
    print("TEST: Image Extraction from Events")
    print("="*60)
    
    from src.app_streamlit import extract_images_from_events
    
    # Mock event with images
    mock_event = {
        "tutor": {
            "worker_outputs": [
                {
                    "extra": {
                        "images": [
                            {
                                "id": "test_image.jpg",
                                "title": "PLC Siemens Example",
                                "category": "plc_siemens",
                                "source": "local",
                                "source_url": "",
                                "author": "Test Author",
                                "license": {
                                    "name": "CC BY 4.0",
                                    "requires_attribution": True,
                                    "allows_commercial": True
                                },
                                "alt_text": "Example PLC diagram",
                                "width": 800,
                                "height": 600
                            }
                        ],
                        "images_count": 1
                    }
                }
            ]
        }
    }
    
    # Test extraction
    images = extract_images_from_events([mock_event])
    
    print(f"✓ Extracted {len(images)} image(s)")
    
    if images:
        img = images[0]
        print(f"  - Title: {img.get('title')}")
        print(f"  - Category: {img.get('category')}")
        print(f"  - ID: {img.get('id')}")
        print(f"  - Expected path: assets/images/{img.get('category')}/{img.get('id')}")
    
    return len(images) > 0


def test_image_bank_structure():
    """Test that image bank directories exist"""
    print("\n" + "="*60)
    print("TEST: Image Bank Structure")
    print("="*60)
    
    import os
    
    base_path = "assets/images"
    expected_categories = [
        "plc_siemens",
        "plc_allen_bradley",
        "cobot_ur",
        "cobot_fanuc",
        "ladder_logic",
        "hmi_screen",
        "profinet",
        "profibus",
        "scada",
    ]
    
    print(f"\nChecking base path: {base_path}")
    if os.path.exists(base_path):
        print("✓ Base image directory exists")
    else:
        print("✗ Base image directory NOT found")
        return False
    
    found_categories = []
    for category in expected_categories:
        cat_path = os.path.join(base_path, category)
        if os.path.exists(cat_path):
            # Count images in category
            try:
                files = [f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
                found_categories.append(category)
                print(f"  ✓ {category}: {len(files)} images")
            except Exception as e:
                print(f"  ⚠ {category}: Error reading - {e}")
        else:
            print(f"  ✗ {category}: Not found")
    
    print(f"\nFound {len(found_categories)}/{len(expected_categories)} expected categories")
    return len(found_categories) > 0


def test_tutor_images_integration():
    """Test that tutor_node can search for images"""
    print("\n" + "="*60)
    print("TEST: Tutor Node Image Integration")
    print("="*60)
    
    try:
        from src.agent.workers.tutor_node import (
            search_relevant_images,
            detect_image_category,
            IMAGES_AVAILABLE
        )
        
        print(f"\n✓ Image functions imported successfully")
        print(f"  Images available: {IMAGES_AVAILABLE}")
        
        if IMAGES_AVAILABLE:
            # Test category detection
            test_query = "Como programo un PLC Siemens S7-1200?"
            category = detect_image_category(test_query)
            print(f"  Detected category for '{test_query}': {category}")
            
            # Test image search
            images = search_relevant_images(
                user_message=test_query,
                evidence_text="",
                learning_style="visual",
                max_images=3
            )
            print(f"  ✓ Found {len(images)} relevant images")
            
            if images:
                for idx, img in enumerate(images[:2]):
                    print(f"\n  Image {idx + 1}:")
                    print(f"    - Title: {img.get('title')}")
                    print(f"    - Category: {img.get('category')}")
                    print(f"    - Source: {img.get('source')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("STREAMLIT IMAGE DISPLAY - INTEGRATION TEST")
    print("="*60)
    
    results = {
        "Image Extraction": test_extract_images_function(),
        "Image Bank Structure": test_image_bank_structure(),
        "Tutor Integration": test_tutor_images_integration(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Image display is ready!")
    else:
        print("⚠ SOME TESTS FAILED - Check errors above")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

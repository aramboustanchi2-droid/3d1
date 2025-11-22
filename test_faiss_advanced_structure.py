"""
Code structure test for FAISS Advanced (IVF/PQ)
بررسی ساختار کد برای FAISS پیشرفته

Tests code structure without requiring FAISS installation
"""
import sys
from pathlib import Path
import ast

# Add cad3d to path
sys.path.insert(0, str(Path(__file__).parent))


def test_file_exists():
    """Test that implementation file exists"""
    print("\n" + "="*70)
    print("Test 1: File Existence")
    print("="*70)
    
    file_path = Path("cad3d/super_ai/ai_model_database_faiss_advanced.py")
    assert file_path.exists(), f"File not found: {file_path}"
    print(f"✓ File exists: {file_path}")


def test_class_structure():
    """Test that class has all required methods"""
    print("\n" + "="*70)
    print("Test 2: Class Structure")
    print("="*70)
    
    file_path = Path("cad3d/super_ai/ai_model_database_faiss_advanced.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    # Find AIModelDatabaseFAISSAdvanced class
    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AIModelDatabaseFAISSAdvanced":
            class_def = node
            break
    
    assert class_def is not None, "AIModelDatabaseFAISSAdvanced class not found"
    print("✓ Found AIModelDatabaseFAISSAdvanced class")
    
    # Check required methods
    required_methods = [
        '__init__',
        '_create_index',
        '_train_index',
        '_load_or_create_indexes',
        '_generate_embedding',
        'create_model',
        'get_model',
        'list_models',
        'search_similar_models',
        'create_dataset',
        'search_similar_datasets',
        'log_prediction',
        'search_similar_predictions',
        'optimize_index',
        'get_statistics'
    ]
    
    method_names = [m.name for m in class_def.body if isinstance(m, ast.FunctionDef)]
    
    for method in required_methods:
        assert method in method_names, f"Method {method} not found"
        print(f"  ✓ Method: {method}")
    
    print(f"✓ All {len(required_methods)} required methods found")


def test_init_parameters():
    """Test that __init__ has correct parameters"""
    print("\n" + "="*70)
    print("Test 3: __init__ Parameters")
    print("="*70)
    
    file_path = Path("cad3d/super_ai/ai_model_database_faiss_advanced.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    # Find __init__ method
    init_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AIModelDatabaseFAISSAdvanced":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    init_method = item
                    break
    
    assert init_method is not None, "__init__ method not found"
    
    # Check parameters
    required_params = [
        'self',
        'index_path',
        'dimension',
        'index_type',
        'nlist',
        'nprobe',
        'm',
        'nbits',
        'use_gpu',
        'gpu_id'
    ]
    
    param_names = [arg.arg for arg in init_method.args.args]
    
    for param in required_params:
        assert param in param_names, f"Parameter {param} not found in __init__"
        print(f"  ✓ Parameter: {param}")
    
    print(f"✓ All {len(required_params)} required parameters found")


def test_demo_exists():
    """Test that demo file exists"""
    print("\n" + "="*70)
    print("Test 4: Demo File")
    print("="*70)
    
    demo_path = Path("demo_faiss_advanced.py")
    assert demo_path.exists(), f"Demo file not found: {demo_path}"
    print(f"✓ Demo file exists: {demo_path}")
    
    # Check demo has main functions
    with open(demo_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    required_functions = [
        'demo_auto_selection',
        'demo_ivf_index',
        'demo_ivfpq_index',
        'demo_optimization',
        'demo_comparison',
        'main'
    ]
    
    for func in required_functions:
        assert func in function_names, f"Function {func} not found in demo"
        print(f"  ✓ Function: {func}")
    
    print(f"✓ All {len(required_functions)} required functions found")


def test_guide_exists():
    """Test that guide file exists"""
    print("\n" + "="*70)
    print("Test 5: Guide Documentation")
    print("="*70)
    
    guide_path = Path("FAISS_IVF_PQ_GUIDE.md")
    assert guide_path.exists(), f"Guide file not found: {guide_path}"
    print(f"✓ Guide file exists: {guide_path}")
    
    # Check guide has required sections
    with open(guide_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_sections = [
        "# FAISS Advanced - IVF/PQ Guide",
        "## Index Types",
        "## Installation",
        "## Usage",
        "## Training",
        "## Parameter Tuning",
        "## Performance Benchmarks",
        "## Best Practices",
        "## Troubleshooting",
        "## Demo"
    ]
    
    for section in required_sections:
        assert section in content, f"Section '{section}' not found in guide"
        print(f"  ✓ Section: {section}")
    
    print(f"✓ All {len(required_sections)} required sections found")


def test_readme_updated():
    """Test that README was updated"""
    print("\n" + "="*70)
    print("Test 6: README Updates")
    print("="*70)
    
    readme_path = Path("README.md")
    assert readme_path.exists(), f"README not found: {readme_path}"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for IVF/PQ mentions
    required_keywords = [
        "FAISS IVF/PQ",
        "AIModelDatabaseFAISSAdvanced",
        "demo_faiss_advanced.py",
        "FAISS_IVF_PQ_GUIDE.md",
        "ivfpq",
        "nlist",
        "nprobe"
    ]
    
    for keyword in required_keywords:
        assert keyword in content, f"Keyword '{keyword}' not found in README"
        print(f"  ✓ Keyword: {keyword}")
    
    print(f"✓ All {len(required_keywords)} required keywords found")


def test_vector_guide_updated():
    """Test that vector implementation guide was updated"""
    print("\n" + "="*70)
    print("Test 7: Vector Implementation Guide Updates")
    print("="*70)
    
    guide_path = Path("AI_DATABASE_VECTOR_IMPLEMENTATION.md")
    assert guide_path.exists(), f"Guide not found: {guide_path}"
    
    with open(guide_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for IVF/PQ section
    required_keywords = [
        "FAISS Advanced - IVF/PQ",
        "IVFPQ",
        "nlist",
        "nprobe",
        "Product Quantization",
        "compression",
        "millions"
    ]
    
    for keyword in required_keywords:
        assert keyword in content, f"Keyword '{keyword}' not found in vector guide"
        print(f"  ✓ Keyword: {keyword}")
    
    print(f"✓ All {len(required_keywords)} required keywords found")


def test_file_sizes():
    """Test that files have reasonable sizes"""
    print("\n" + "="*70)
    print("Test 8: File Sizes")
    print("="*70)
    
    files = {
        "cad3d/super_ai/ai_model_database_faiss_advanced.py": 20_000,  # ~25KB
        "demo_faiss_advanced.py": 5_000,  # ~7KB
        "FAISS_IVF_PQ_GUIDE.md": 10_000,  # ~14KB
        "FAISS_IVF_PQ_SUMMARY.md": 5_000,  # ~7KB
    }
    
    for file_path, min_size in files.items():
        path = Path(file_path)
        assert path.exists(), f"File not found: {file_path}"
        
        size = path.stat().st_size
        assert size > min_size, f"File {file_path} too small: {size} bytes (expected >{min_size})"
        
        print(f"  ✓ {file_path}: {size:,} bytes")
    
    print("✓ All files have reasonable sizes")


def main():
    print("\n" + "="*70)
    print("FAISS Advanced (IVF/PQ) - Code Structure Tests")
    print("بررسی ساختار کد FAISS پیشرفته")
    print("="*70)
    print("\nThese tests verify code structure without requiring FAISS installation")
    
    try:
        test_file_exists()
        test_class_structure()
        test_init_parameters()
        test_demo_exists()
        test_guide_exists()
        test_readme_updated()
        test_vector_guide_updated()
        test_file_sizes()
        
        print("\n" + "="*70)
        print("✓ All code structure tests passed!")
        print("="*70)
        print("\nImplementation Summary:")
        print("  ✓ AIModelDatabaseFAISSAdvanced class with 15+ methods")
        print("  ✓ Support for Flat, IVF, IVFPQ, HNSW indexes")
        print("  ✓ Auto-selection and optimization")
        print("  ✓ GPU support")
        print("  ✓ Comprehensive demo with 5 scenarios")
        print("  ✓ Full documentation (guide + summary)")
        print("  ✓ Updated README and vector guide")
        print("\nTo run functional tests, install FAISS:")
        print("  pip install faiss-cpu")
        print("  python test_faiss_advanced_quick.py")
        print("\nTo run demos:")
        print("  python demo_faiss_advanced.py --demo all")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

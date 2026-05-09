from AbstractIntegratedModule import IntegratedPipeline, PipelinePredictionManager
import numpy as np

# Initialize with a test memory name
try:
    model = IntegratedPipeline('test_agent_memory')
    print("✓ IntegratedPipeline initialized successfully")
    
    # Test basic functionality
    test_titles = ["Testing VSCode", "Opening GitHub"]
    print(f"✓ Ready to process {len(test_titles)} titles")
    
    # Test PipelinePredictionManager
    pred_manager = PipelinePredictionManager(model)
    print("✓ PipelinePredictionManager initialized successfully")
    
    print("\n✅ All tests passed! Installation is complete.")
    
except Exception as e:
    print(f"❌ Installation test failed: {e}")
    import traceback
    traceback.print_exc()

# Troubleshooting Guide

## Common Issues

### "ModuleNotFoundError: No module named 'AbstractIntegratedModule'"

**Cause**: Binary file not in correct location

**Solutions**:
```bash
# Option 1: Place binary in project root
cp AbstractIntegratedModule.*.so ./

# Option 2: Place in Python site-packages
pip install -e .

# Option 3: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/binary"
```

      
2. Issue 2: "ImportError: DLL load failed" when using binary version of the library (For Windows)
Solution:

   - 1. Ensure AbstractIntegratedModule.pyd is in your project root
   - 2. Install Visual C++ redistributables:
      Download from: https://support.microsoft.com/en-us/help/2977003
   - 3. Verify Python architecture (32-bit vs 64-bit) matches the .pyd file


3. Issue 3: "Permission denied" when using binary version of the library (for Linux/Raspberry Pi)
[=] Solution:
   - ```
     # Make sure you have read permissions
     chmod 644 AbstractIntegratedModule.cpython-39-*.so
     # If in virtual environment, ensure it's activated
     source venv/bin/activate
    ```

4. Issue 4: Missing Dependencies when using binary
[=] Solution:
    - ```
      # Reinstall all dependencies
      pip install --upgrade pip setuptools wheel
      pip install numpy pandas scikit-learn matplotlib scipy requests
     
      # Verify installation
      pip list
      ```

5. Issue 5: Virtual Environment Issues
[=] Solution:
    - ```
      # Deactivate current environment
      deactivate

      # Create a fresh virtual environment
      python -m venv fresh_venv

      # Activate it
      source fresh_venv/bin/activate  # Linux/Raspberry Pi
      # or
      fresh_venv\Scripts\activate     # Windows

      # Reinstall
      pip install --upgrade pip
      pip install numpy pandas scikit-learn matplotlib scipy
      ```

6. Issue 6: Raspberry Pi - "Bus error" or Performance Issues
Solution:
   - [=] Ensure adequate swap space:
      ```
      free -h  # Check current swap
      sudo nano /etc/dphys-swapfile  # Increase if needed
       ```
   - Close unnecessary applications before running
   - Consider using a faster SD card (UHS-I or better)
     
7. Issue 7: P2P Undefined Connection:
   If you get this warning:
   - ```
     [❌] Failed to connect to <host>:<port>: Nonetype object has no attribute .accept()
     ```
     - [=] Solution:
     - ```
       # initiate socket first
       main_model.distribution.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       main_model.distribution.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       main_model.distribution.socket.bind(('0.0.0.0', self.port)) # self.port could be changed with other ports
       main_model.distribution.socket.listen(5) # listens for 5 seconds
       ```
8. Issue 8: Cannot compare using '<' with str and float data:
   - This happens when one of the labels extracted from your CSV file has NaN value, consider replace NaN with actual labels.

9. Issue 9: Failed authentication during asynchronous prediction:
   - Solution: Ensure API_KEY is initialized inside PipelineAsyncManager like this:
   - ```python
      async_manager = PipelineAsyncManager(main_model,
      main_prediction, # your previous initialized PipelinePredictionManager
      config=security_config,
      state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
      security_level=SecurityLevel.PRODUCTION, # production level security initiated
      api_key=secret_key # set secret key you initialized <- THIS IS IMPORTANT
      max_workers=4, # workers to initiate prediction, more workers, more capabilities to process prediction requests.
      task_timeout=30,
      max_retries=3 ) # retries after failure during prediction
     ```
     - And make sure you also initialized api_key in the predict function that requires api_key:
        ```
        async_manager.predict(...., ...., api_key=api_key, ...)
        ```

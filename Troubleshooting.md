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

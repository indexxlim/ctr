#!/usr/bin/env python
"""
Fix npy-append-array compatibility with numpy 1.24+

The isfileobj function was removed from numpy.lib.format in numpy 1.24,
but npy-append-array still tries to import it. This script patches the
npy_append_array/format.py file to define isfileobj locally.
"""

import sys
import site

def fix_npy_append_array():
    """Patch npy-append-array to work with numpy 1.24+"""

    # Find site-packages directory
    site_packages = site.getsitepackages()[0]
    format_file = f"{site_packages}/npy_append_array/format.py"

    try:
        with open(format_file, 'r') as f:
            content = f.read()

        # Check if already patched
        if 'def isfileobj(f):' in content:
            print("✓ npy-append-array already patched")
            return True

        # Apply patch
        old_import = "from numpy.lib.format import header_data_from_array_1_0, isfileobj"
        new_import = """from numpy.lib.format import header_data_from_array_1_0

# isfileobj was removed in numpy 1.24+, define it here
def isfileobj(f):
    \"\"\"Check if object is a file-like object.\"\"\"
    return hasattr(f, 'read') and hasattr(f, 'seek')"""

        if old_import in content:
            content = content.replace(old_import, new_import)

            with open(format_file, 'w') as f:
                f.write(content)

            print("✓ Successfully patched npy-append-array")
            return True
        else:
            print("⚠ Could not find expected import statement")
            return False

    except Exception as e:
        print(f"✗ Error patching npy-append-array: {e}")
        return False

if __name__ == "__main__":
    success = fix_npy_append_array()
    sys.exit(0 if success else 1)

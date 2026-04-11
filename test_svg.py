import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from tools import create_svg_image
from config import get_visuals_dir

def test_svg():
    svg_code = """
<svg width="200" height="200" xmlns="http://www.w3.org/2001/svg">
  <rect width="100%" height="100%" fill="skyblue" />
  <circle cx="100" cy="100" r="80" stroke="white" stroke-width="4" fill="gold" />
  <text x="100" y="115" font-size="40" text-anchor="middle" fill="orange" font-family="Arial">O</text>
</svg>
"""
    print("Testing create_svg_image...")
    result = create_svg_image(svg_code, filename="test_output.svg")
    print(f"Result: {result}")
    
    visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
    test_file = visuals_dir / "test_output.svg"
    
    if test_file.exists():
        print(f"PASS: File created at {test_file}")
    else:
        print(f"FAIL: File not found at {test_file}")

if __name__ == "__main__":
    test_svg()

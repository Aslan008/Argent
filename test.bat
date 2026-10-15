@echo off
start /b python -c "import time, sys; [print(f'fast tick {i}', flush=True) or time.sleep(0.1) for i in range(20)]"

@echo off
echo Quick test of tech support demo (3 generations only)...

python -m embedlab.cli_evolution evolve ^
  --hierarchy data/tech_support_hierarchy.yaml ^
  --output evolution_runs/test_evolve ^
  --generations 3 ^
  --population 5 ^
  --verbose

echo.
echo Quick test complete! Check evolution_runs\test\ for results.
pause

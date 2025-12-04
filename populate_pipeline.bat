@echo off
REM Demo pipeline for seeds 0-10
setlocal
for /L %%S in (0,1,10) do (
  echo === Seed %%S ===
  call .\.venv\Scripts\python.exe P2\wfc_stage.py --prob P2\wfc_stats\probabilities.json --adj P2\wfc_stats\adjacency.json --outdir P2\p2_tiles --seed %%S --tiles_x 40 --png_scale 36 --coarse_factor 3 --relax_on_fail || goto :error
  call .\.venv\Scripts\python.exe P2\orchestrate_bands_l0.py --manifest P2\p2_tiles\manifest_seed%%S.json --rules P2\wfc_stats\zone_rules.json --outdir P2\p3_orchestrate --debugdir P2\debug --seed %%S || goto :error
  call .\.venv\Scripts\python.exe P2\populate_zones_l1.py --manifest P2\p2_tiles\manifest_seed%%S.json --zones P2\p3_orchestrate\zones_l0_seed%%S.json --patterns P2\wfc_stats\patterns.json --geometry P2\wfc_stats\element_geometry.json --outdir P2\p4_populate --debugdir P2\debug --png_scale 36 --seed %%S || goto :error
  call .\.venv\Scripts\python.exe P2\render_children_l1.py --manifest P2\p2_tiles\manifest_seed%%S.json --children P2\p4_populate\children_l1_seed%%S.json --geometry P2\wfc_stats\element_geometry.json --out P2\debug\children_l1_geom_seed%%S.png || goto :error
  call .\.venv\Scripts\python.exe P2\trim_debug_image.py --input P2\debug\children_l1_geom_seed%%S.png --output P2\debug\children_l1_geom_seed%%S_trim.png --background "#101010" || goto :error
)
echo Demo complete
goto :eof
:error
echo Pipeline failed on seed %%S
exit /b 1


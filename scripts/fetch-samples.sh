#!/usr/bin/env bash
# Fetch the public sample models into samples/ (gitignored) and check them.
#
#   Clinic_HVAC.ifc        NIBS / USACE Common BIM Files, medical clinic HVAC
#                          (Revit MEP 2013, IFC2x3, 263 spaces), from
#                          buildingSMART's community sample files
#   rme_advanced_2025.rvt  Autodesk's Revit MEP advanced sample, 2025 release
#                          (the 2024, 2026 and 2027 ones decode fewer joins)
#   Snowdon Towers Sample HVAC.rvt
#                          Autodesk's Revit 2024 sample
#   HVAC_NordicLCA_Housing_Concrete_BuildingPermit.{rvt,ifc}
#                          Granlund's MagiCAD model from BIM4LCA (VTT, CC BY-SA
#                          4.0): two files out of an 870 MB archive
#
# Then, e.g.:
#   (cd rvt-read/wasm && cargo run --release --example analyse ../../samples/Clinic_HVAC.ifc)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p samples
cd samples

fetch() { # name url sha256
  if [ -f "$1" ] && shasum -a 256 "$1" | grep -q "^$3"; then
    echo "have $1"
    return
  fi
  echo "fetching $1"
  curl -fsSL -o "$1" "$2"
  check "$1" "$3"
}

check() { # name sha256
  if ! shasum -a 256 "$1" | grep -q "^$2"; then
    echo "$1: checksum mismatch" >&2
    exit 1
  fi
}

fetch Clinic_HVAC.ifc \
  "https://media.githubusercontent.com/media/buildingsmart-community/Community-Sample-Test-Files/main/IFC%202.3.0.1%20(IFC%202x3)/Medical-Dental%20Clinic/Clinic_HVAC.ifc" \
  39c88a79f48fbe56da86afb0fb3ebd188f8930df3df7dbfbd9ecaa535aaeab9b
fetch rme_advanced_2025.rvt \
  https://revit.downloads.autodesk.com/download/2025RVT_RTM/Docs/InProd/rmeadvancedsampleproject.rvt \
  0845729234e073b52d42ece73680b9c6d8ef13bb0bc768fe4e9c59e3ec8886d5
fetch "Snowdon Towers Sample HVAC.rvt" \
  "https://revit.downloads.autodesk.com/download/2024RVT_RTM/Docs/InProd/Samples/Snowdon%20Towers%20Sample%20HVAC.rvt" \
  585385991b1f8a168881c4bc36546bc90e7bfb427c263d801fe367fa2ebb0fa8

granlund=HVAC_NordicLCA_Housing_Concrete_BuildingPermit
if [ -f "$granlund.rvt" ] && [ -f "$granlund.ifc" ]; then
  echo "have $granlund"
else
  echo "fetching $granlund (870 MB archive)"
  curl -fsSL -o HVAC.zip https://www.nordicsustainableconstruction.com/Media/638596535832776143/HVAC.zip
  unzip -oj -q HVAC.zip "HVAC/MAGICAD-FOR-REVIT/$granlund.rvt" "HVAC/IFC/$granlund.ifc"
  rm HVAC.zip
fi
check "$granlund.rvt" 7b8fec73381b87f4d44d6265ec0ac568deefd641ad93df796ca0f4eac47e24ba
check "$granlund.ifc" 25f3f21240fbe3c4ed7b2cc90d5072fd81d0535f1bbed911ca653d7d92d88c46
echo "samples/ ready"

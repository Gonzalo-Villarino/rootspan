#!/bin/bash
cd "$(dirname "$0")" || exit 1
exec "00_Software/ml-env/bin/python" "09_Periderm_GUI/launch.py"

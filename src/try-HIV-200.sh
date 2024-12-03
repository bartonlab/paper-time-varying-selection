#!/bin/bash
cd ..
python new_infer.py -tag 703010256-3 -name 100 -dir data/HIV200 -g2tv 100 || echo "CH256-3 failed, continuing with next script..." 
python new_infer.py -tag 703010256-3 -name 150 -dir data/HIV200 -g2tv 150 || echo "CH256-3 failed, continuing with next script..." 
python new_infer.py -tag 703010256-3 -name 200 -dir data/HIV200 -g2tv 200 || echo "CH256-3 failed, continuing with next script..." 
echo "All scripts have been attempted."
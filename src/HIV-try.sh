#!/bin/bash
cd ..
python new_infer.py -tag 705010162-3 -dir data/HIV200 --raw --cut || echo "CH162-3 failed, continuing with next script..." 
python new_infer.py -tag 706010164-3 -dir data/HIV200 --raw --cut || echo "CH164-3 failed, continuing with next script..." 
echo "All scripts have been attempted."

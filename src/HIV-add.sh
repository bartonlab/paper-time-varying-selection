#!/bin/bash
cd ..
python inference_HIV.py -tag 700010040-3 --raw --add || echo "CH040-3 failed, continuing with next script..." 
python inference_HIV.py -tag 700010040-5 --raw --add || echo "CH040-5 failed, continuing with next script..." 
echo "All scripts have been attempted."
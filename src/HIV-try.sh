#!/bin/bash
cd ..
python inference_HIV.py -tag 705010185-5 -output output-new || echo "CH185-5 failed, continuing with next script..." 
python new_infer.py -tag 705010185-5 -name _50 -output output-new -g2tv 50 || echo "CH185-5 failed, continuing with next script..." 
python new_infer.py -tag 705010185-5 -name _200 -output output-new -g2tv 200 || echo "CH185-5 failed, continuing with next script..." 
python inference_HIV.py -tag 703010256-3 -output output-new || echo "CH256-3 failed, continuing with next script..." 
python new_infer.py -tag 703010256-3 -name _50 -output output-new -g2tv 50 || echo "CH256-3 failed, continuing with next script..." 
python new_infer.py -tag 703010256-3 -name _200 -output output-new -g2tv 200 || echo "CH256-3 failed, continuing with next script..." 
python inference_HIV.py -tag 700010040-3 -output output-new || echo "CH040-3 failed, continuing with next script..." 
python new_infer.py -tag 700010040-3 -name _50 -output output-new -g2tv 50 || echo "CH040-3 failed, continuing with next script..." 
python new_infer.py -tag 700010040-3 -name _200 -output output-new -g2tv 200 || echo "CH040-3 failed, continuing with next script..." 
echo "All scripts have been attempted."
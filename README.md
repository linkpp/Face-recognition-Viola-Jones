Face recognition Opencv, Find the best face! 
============================================
Input: A video/ webcam (realtime) <br/>
Output: Face detect & recognition, find the best face in all.

Prequisite
------------------------------
- Opencv 3.3.6
- Python 3
- statistics (for compute stdev)
- matplotlib (For visualize histogram if you need)

Run
------------------------------
- Train your model first:
    + place your images in train folder 
    + call function : train_and_save_model("train/")
- Edit your input video path in main.py.
- run:
  + python3 main.py
  (see more in demo video)
  
Video demo: https://youtu.be/T2cMSd4WX9M <br/>

Note
-------------------------------
The best face consider by 4 factor:
- Accurancy
- Straight face
- Contrast
- Entropy

Contact
------------------------------
Copyright by LinhPhan <br/>
Contact me: https://fb.com/deluxe.psk
 

so this branch contains all the demo for text branch it uses the fine tuned bio bert to take a text sympyom in the format as :
"The patient presents with shortness of breath, persistent cough, fever, chills, and chest pain."
and output proabbilties similary take image and output probabilites 
but the probelm is the fine tuned weights were too heavy to upload to github so download from this url 
https://drive.google.com/file/d/16ij_cjh-aG_pI0NTw2jA32nH-t-_4OGv/view?usp=drive_link

and put inside text_module and this will work and run demo with 
streamlit run app.py

it also has chest xray explaaintbility using GRAD-CAM 


BUT to run the demo following requirements are required they can be done using pip in a single line 

Open CMD
pip install torch streamlit transformers torchxrayvision pytorch-grad-cam numpy pillow opencv-python

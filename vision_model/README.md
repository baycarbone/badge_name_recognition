# Readme

- clone repo
- create a python venv: `python3 -m venv .venv`
- activate the venv: `. .venv/bin/activate`
- install dependencies
```
pip install --no-cache-dir -r ./requirements.txt
```
Or individually (might be missing some here):
```
pip install ultralytics
pip install supervision
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch
pip install sentence_transformers
```
- Create the directory to store badge images and serve those images (a `badge` directory will automatically be created in this directory)
```
mkdir input_images
cd input_images/
twistd3 web --listen tcp:9595 --path .
```
 
- launch the python app: `python3 -m main --feed 4 --resolution 1920 1080 --name-url http://example.com:8080/v1/models/internvl2:predict --badge-location http://example.com:9696/badge
  - http://example.com:9595/badge: this url is where the badge images will be served from.   
  - you can adjust the resolution based on what the camera supports, the default if not specified is 640x480
  - the feed number is the camera index. Usually feed 0 is the built-in webcam and the next is any other camera plugged in. You may need to try not just 1, but 2, 3, 4 like in my case until you find the right index.


# Notes
The model was trained on a badge image dataset found on roboflow (can't remember which one).
When you're done, make sure the model is no longer running, seems `ctrl+c` does not always do it and you need to kill the process.
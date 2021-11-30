
## Linux Setup 


If the instructions below don't work you might need to run: 

```bash 
sudo apt-get install libgtk2.0-dev
```


### Python 3.8 

```bash
sudo apt install python3.8-venv

python3.8 -m venv ~/venvs/py38-cv
source ~/venvs/py38-cv/bin/activate

pip install opencv-python
pip install pygame
```


### Python 3.7

```bash
sudo apt install python3.7-venv

python3.7 -m venv ~/venvs/py37-cv
source ~/venvs/py37-cv/bin/activate

pip install opencv-python
pip install pygame
```

### Using conda

```bash
sudo apt-get install libgtk2.0-dev
conda install -c conda-forge opencv=4.1.0
pip install opencv-contrib-python  : Needed to get cv2.data submodule
```


## Mac / Windows Setup

Sorry, you are on your own; CV devs are not big on macs or Windows, probably due to the fact that 
setting up GPU processing on those systems used to be a nightmare until not so long ago. 

You might try the (analogs) of the steps above,  (on Macs probably something involving `brew` or 
some rogue crap like ...) 

If you get it to work, please contribute your PR!


## Running `slouch_detect.py`

Make sure the to set `STORE_IMGS = False` in the code, 
unless you really want to store images for a while.

```
# assuming your pip or conda environemnt is already activated, just run:

cd talk-comp-viz-bad-posture-detector
python slouch_detect.py

```


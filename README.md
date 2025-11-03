# PacMan

A A3C Convultional Neural Network that learns to play the game Kung Fu Master


//IMPORTANT: In initialize.bat, change "set PATH..." to point to the swigwin folder. If not in repo, download the swingin folder from https://swig.org/. 

//install py 3.11 if not existing

py install 3.11

//create the new venv with python 3.11

py -3.11 -m venv env_KFM

//activates the enviroment

.\\env_KFM\\Scripts\\activate



//installers:

python.exe -m pip install --upgrade pip
pip install numpy

// go to https://pytorch.org/get-started/locally/ in order to know the specific pytorch best for your system. Update your NVIDIA drivers for it to work too.
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


pip install imageio
pip install IPython
pip install imageio[ffmpeg]
pip install imageio[pyav]

pip install gymnasium[atari]


//pip install imageio && pip install IPython && pip install imageio[ffmpeg] && pip install imageio[pyav] && pip install gymnasium[atari]


conda create -n APT 
conda activate APT
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
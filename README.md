# NUTS: raNdom dimensionality redUction non axiomaTic reasoning few Shot learner for perception
Supplementary Material For the Paper


# Setup

## 1 Clone NUTS

cd ~/projects
git clone https://github.com/dwanev/NUTS 

## 2 Install ONA (Open NARS for applications)

 - NUTS uses open NARS for applications (ONA). Which  can be found here: https://github.com/opennars/OpenNARS-for-Applications. ONA is installations instructions can be found on the webpage above, but in brief:

cd ~/projects
git clone https://github.com/opennars/OpenNARS-for-Applications  
cd OpenNARS-for-Applications  
./build.sh  

 - Set up an environment variable so the NUTS can find ONE

export ONA_PATH_TO_NAR=~/projects/OpenNARS-for-Applications 


## 3 create a virtual environment 

conda create -n nuts python=3.9
conda activate nuts

### 4 install dependencies 

cd ~/projects/NUTS
conda activate nuts  
pip install -r requirements.txt





# Misc

### To start NARs (Just FYI)

cd projects/OpenNARS-for-Applications  
conda activate open_nars  
./NAR shell  


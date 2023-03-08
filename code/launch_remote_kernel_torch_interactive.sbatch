#! /usr/bin/env zsh

### #SBATCH --job-name=juptun_cmdstanpy
### #SBATCH --cpus-per-task=6 #14 #25 #43
### #SBATCH --mem=24G
### #SBATCH --time=2-00:00:00
#SBATCH --output ../logs/jupyter-log-%J.txt
### #SBATCH --partition=gablab
### #SBATCH --partition=om_bigmem
### 

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"


env_iaa_cmdstan

echo "Conda version:"
which conda
conda -V

echo "Python running from:"
which python


## https://ipyrad.readthedocs.io/HPC_Tunnel.html

## get tunneling info
XDG_RUNTIME_DIR=""
iplport=9028
ipnport=8458 ### $(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

### Dont work
# echo 'ssh -N -L $iplport:$ipnip:$ipnport daeda@openmind7.mit.edu' > "../logs/jupyter-log-currenttunnel.txt"
# echo 'ssh -N -L ${iplport}:${ipnip}:${ipnport} daeda@openmind7.mit.edu' > "../logs/jupyter-log-currenttunnel0.txt"
# echo -e 'ssh -N -L $iplport:$ipnip:$ipnport daeda@openmind7.mit.edu' > "../logs/jupyter-log-currenttunnel1.txt"
# printf 'ssh -N -L $iplport:$ipnip:$ipnport daeda@openmind7.mit.edu' > "../logs/jupyter-log-currenttunnel2.txt"
### Works
printf 'ssh -N -L %s:%s:%s daeda@openmind7.mit.edu' "${iplport}" "${ipnip}" "${ipnport}" > "../logs/jupyter-log-currenttunnel.txt"

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Config found at ~/.jupyter/jupyter_notebook_config.py

    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $iplport:$ipnip:$ipnport daeda@openmind7.mit.edu
    -----------------------------------------------------------------
    "

# if [ "$ipnip" -ne "$s2" ]; then
#     echo -e "
#     WARNING, host ip changed to $ipnport
#     "

echo -e "
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$iplport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

echo -e "
    $iplport:$ipnip:$ipnport
    "

## start an ipcluster instance and launch jupyter server
jupyter notebook --no-browser --port=$ipnport --ip=$ipnip

exit

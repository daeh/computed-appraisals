#! /usr/bin/env zsh

#SBATCH --output ../logs/jupyter-log-%J.txt

### remember that logs directory needs to exist for sbatch to launch

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"
source "${HOME}/.me.conf"

env_cam

echo "Python running from:"
which python

## get tunneling info
XDG_RUNTIME_DIR=""
iplport=9028
ipnport=8458 ### $(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)


printf "ssh -N -L %s:%s:%s ${USER}@${HOST}.mit.edu" "${iplport}" "${ipnip}" "${ipnport}" > "../logs/jupyter-log-currenttunnel.txt"

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Config found at ~/.jupyter/jupyter_server_config.py

    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $iplport:$ipnip:$ipnport ${USER}@openmind7.mit.edu
    -----------------------------------------------------------------
    "

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
jupyter server --no-browser --port=$ipnport --ip=$ipnip

exit

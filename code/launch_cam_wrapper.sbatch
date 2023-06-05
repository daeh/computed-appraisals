#! /usr/bin/env zsh

#SBATCH --job-name=camMain
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-04:00:00
#SBATCH --output ../logs/sbatchlog_cammain-%J.txt
#SBATCH --partition=gablab

### remember that logs directory needs to exist for sbatch to launch

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"
source "${HOME}/.me.conf"

env_cam

cd "${PROJCAMPATH}/code" || exit

python cam_main.py

echo ''
echo 'sbatchrun FINISHED. exiting.'
exit

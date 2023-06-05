#! /usr/bin/env zsh

#SBATCH --job-name=camWebPPLwpr
#SBATCH --cpus-per-task=40
#SBATCH --mem=30GB
#SBATCH --time=0-10:00:00
#SBATCH --output=../logs/sbatchlog-iaa_runwppl_%J_stdout.txt

### remember that logs directory needs to exist for sbatch to launch

scriptpath=$1
picklepath=$2

lsb_release -a

echo ""
echo "PATH INITIAL"
echo $PATH
echo ""

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"
source "${HOME}/.me.conf"

env_cam

echo "Python running from:"
which python

cd "$(dirname -- "${scriptpath}")" || exit 1

echo ""
echo "Executing:"
echo "python cam_webppl.py ${picklepath}"
echo "-------"
echo ""
echo ""

python cam_webppl.py "${picklepath}"
exit_status=$?

if [ "${exit_status}" -ne 0 ];
then
    echo "script exited with exit status: ${exit_status}"
    exit "${exit_status}"
fi
echo "EXIT STATUS ${exit_status}"

echo ''
echo 'sbatch FINISHED. exiting.'
exit "${exit_status}"


#! /usr/bin/env zsh

#SBATCH --job-name=camTorchWpr
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0-10:00:00
#SBATCH --output=../logs/sbatchlog-camTorchWpr_%J_stdout.txt

### remember that logs directory needs to exist for sbatch to launch

task_id=$(($SLURM_ARRAY_TASK_ID))
task_num=$((task_id))

scriptpath=$1
pklpath=$2
behavior=$3

echo "task_id :: $task_id"
echo "task_num :: $task_num"
echo "HOSTNAME :: $HOSTNAME"

lsb_release -a

echo ""
echo "PATH INITIAL ::"
echo $PATH
echo ""

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"
source "${HOME}/.me.conf"

env_cam

echo ""
echo "conda path ::"
which conda

echo ""
echo "conda version ::"
echo $(conda --version)

echo ""
echo "conda info ::"
conda info

echo ""
echo "python path ::"
which python

cd "$(dirname -- "${scriptpath}")" || exit 1

echo ""
echo "Executing ::"
echo "python cam_emotorch.py --picklepath ${pklpath} --jobnum $task_num --behavior ${behavior}"
echo "-------"
echo ""
echo ""

#########################

python cam_emotorch.py --picklepath "${pklpath}" --jobnum $task_num --behavior "${behavior}"
exit_status=$?

echo ""
echo ""
echo "-------"
echo ""

if [ "${exit_status}" -ne 0 ];
then
    echo "script exited with exit status: ${exit_status}"
    exit "${exit_status}"
fi
echo "EXIT STATUS ${exit_status}"

echo ''
echo 'sbatchrun FINISHED. exiting.'
exit "${exit_status}"


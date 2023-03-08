#! /usr/bin/env zsh

#SBATCH --job-name=iaaPyTorchWpr
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=daeda@mit.edu --mail-type=ALL
#SBATCH --output=../logs/sbatchlog-iaaPyTorchWpr_%J_stdout.txt
### #SBATCH --partition=use-everything

### remember that logs directory needs to exist for sbatch to launch

task_num=$(($SLURM_ARRAY_TASK_ID))
echo "task_num: $task_num"
task_num2=$((task_num))
echo "task_num2 $task_num2"

pklpath=$1
behavior=$2
echo "pklpath $pklpath"
echo "behavior $behavior"

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"


env_iaa_cmdstan

echo "Conda version:"
which conda
conda -V

echo "Python running from:"
which python

cd /om/user/daeda/ite_iaa/ite_gb_inverseappraisal/code || exit

#########################

echo ""
echo "Executing:"
echo "python iaa21_run_pytorch_torchdata.py --picklepath ${pklpath} --jobnum $task_num2 --behavior ${behavior}"
echo "-------"
echo ""
echo ""

python iaa21_run_pytorch_torchdata.py --picklepath "${pklpath}" --jobnum $task_num2 --behavior "${behavior}"
exit_status=$?

if [ "${exit_status}" -ne 0 ];
then
    echo "script exited with exit status: ${exit_status}"
    exit "${exit_status}"
fi
echo "EXIT STATUS ${exit_status}"

echo ''
echo 'sbatchrun FINISHED. exiting.'
exit "${exit_status}"


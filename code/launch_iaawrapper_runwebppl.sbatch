#! /usr/bin/env zsh

#SBATCH --job-name=iaa_runwppl
#SBATCH --cpus-per-task=38
#SBATCH --mem=40GB
#SBATCH --time=1-01:00:00
#SBATCH --mail-user=daeda@mit.edu --mail-type=ALL
#SBATCH --output=../logs/sbatchlog-iaa_runwppl_%J_stdout.txt
### #SBATCH --error=../logs/sbatchlog-runwppl_%J_stderr.txt
### #SBATCH --partition=gablab
### #SBATCH --partition=om_bigmem

### remember that logs directory needs to exist for sbatch to launch

### was 20gb before adding debug

source /usr/share/Modules/init/zsh

source "${HOME}/.merc"


env_iaa_cmdstan

echo "Conda version:"
which conda
conda -V

echo "Python running from:"
which python

cd /om/user/daeda/ite_iaa/ite_gb_inverseappraisal/code || exit

echo 'sending to python ' '[' $1 ']'

python iaa_gendata_webppl.py $1
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


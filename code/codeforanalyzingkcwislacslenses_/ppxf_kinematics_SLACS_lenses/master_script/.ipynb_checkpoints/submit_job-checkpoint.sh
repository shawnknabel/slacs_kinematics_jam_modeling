#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_data=5G,h_rt=336:00:00,highp
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M $shawnknabel@gmail.com
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
#module load gcc/4.9.3
module load python

## substitute the command to run your code
## in the two lines below:
echo '/u/scratch/k/knabel/ -v hello.py'
/u/scratch/k/knabel/ -v hello.py

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####
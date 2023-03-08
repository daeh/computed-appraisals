#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%

import re
import subprocess
from joblib import Parallel, delayed, cpu_count
from pprint import pprint
import numpy as np
import pandas as pd
import datetime


def get_seff(job):
    jobid_, jobname, ncpu, maxrss = job
    jobid = jobid_.strip()
    seff_shellout = subprocess.run(['seff', jobid], capture_output=True)
    return dict(jobid=jobid, jobname=jobname, ncpu=ncpu, maxrss=maxrss, stdout=seff_shellout.stdout.decode(), shellout=seff_shellout)


def main(daysago=8, job_names=None, job_ids=None):
    """
    daysago=0
    job_names=['stanRecompile']#['stanOpt_test_r2']#['stanOpt_test']#['iaa_compile_allmodel_results']
    job_ids=None
    """

    if job_names is None:
        job_names = list()
    if job_ids is None:
        job_ids = list()

    dt_today = datetime.date.today()
    dt_delta = dt_today - datetime.timedelta(days=daysago)

    cmd_list = ["sacct", "-S", f"{dt_delta.strftime('%Y-%m-%d')}", "-u", "daeda", "--format=JobID%40,Jobname%40,NCPUS,MaxRSS"]  # "--endtime", "2020-02-21"

    sacct_shellout = subprocess.run(cmd_list, capture_output=True)
    # pprint(sacct_shellout.stdout.decode())
    # aaa = sacct_shellout.stdout.decode()
    # print(aaa)

    jobs = re.findall(r'([0-9]{7,10}_[0-9]{1,4}|[0-9]{7,10}) +(\S+)\s+([0-9]{1,2})\s+([0-9]{6,10})', sacct_shellout.stdout.decode())
    # re.findall(r'([0-9]{7,10}_[0-9]{1,4}|[0-9]{7,10}) +(\S+)\s+([0-9]{1,2})\s+([0-9]{6,12})K +', sacct_shellout.stdout.decode())
    # jobs = re.findall(r'([0-9]{7,10}_[0-9]{1,4}|[0-9]{7,10}) +(\S+)', sacct_shellout.stdout.decode())

    jobs_trimmed = list()
    if job_names or job_ids:
        for job in jobs:
            jobid_, jobname, ncpu, maxrss = job
            jobid = jobid_.strip()
            if (job_names and jobname in job_names) or (job_ids and (jobid in job_ids or int(jobid) in job_ids)):
                jobs_trimmed.append(job)
        jobs = jobs_trimmed

    print(f"from {daysago} days ago, nJobs = {len(jobs)}")

    ####

    with Parallel(n_jobs=min(len(jobs), cpu_count())) as pool:
        seff_stdout_list = pool(delayed(get_seff)(job) for job in jobs)
    """
    for i_job,job in enumerate(jobs):
        if i_job % 50 == 0:
            print(f"--job {i_job+1}")
        jobid_,jobname,ncpu,maxrss = job

        jobid = jobid_.strip()
        seff_shellout = subprocess.run(['seff', jobid], capture_output=True)
        # pprint(seff_shellout.stdout.decode())

        # re.search(r'\nState\:\s(\S+)(:?\n|\s)', seff_shellout.stdout.decode())
    """
    jobs_stats = list()
    errors = dict()
    for seff_result_dict in seff_stdout_list:
        jobname = seff_result_dict['jobname']
        jobid = seff_result_dict['jobid']
        ncpu = seff_result_dict['ncpu']
        maxrss = seff_result_dict['maxrss']
        stdout = seff_result_dict['stdout']
        if 'State' in stdout:
            status = re.search(r'\nState\:\s([^\n]*)', stdout).group(1)
            status0 = status.split(' ')[0]
            if status0 not in ['PENDING', 'RUNNING']:  # ['TIMEOUT (exit code 0)', 'OUT_OF_MEMORY (exit code 0)', 'PENDING', 'COMPLETED (exit code 0)', 'RUNNING']
                memut = re.search(r'Memory Utilized\:\s+([0-9]+\.[0-9]+)\s(\S+)\s.*', stdout)
                memeff = re.search(r'Memory Efficiency\:\s+([0-9]+\.[0-9]+)%\sof\s([0-9]+\.[0-9]+)\s(\S+)\s.*', stdout)

                job_stats_ = {
                    'name': jobname,
                    'jobid': jobid,
                    'ncpu': int(ncpu),
                    'MaxRSS': int(maxrss),
                    'MemUtalized': float(memut.group(1)),
                    'MemEff': float(memeff.group(1)),
                    'MemReq': float(memeff.group(2)),
                    'status': status0,
                }
                if memut.group(2) != 'GB':
                    job_stats_['MemUtalized'] = float(memut.group(1)) / {'KB': 1e6, 'MB': 1e3}[memut.group(2)]

                jobs_stats.append(job_stats_)
        else:
            errors[jobid] = seff_result_dict

    jobs_stats_df = pd.DataFrame(jobs_stats).sort_values(by=['name', 'jobid'])

    statuses = np.unique(jobs_stats_df.loc[:, 'status'].values)

    jobids = np.unique(jobs_stats_df.loc[:, 'name'].values)

    jobs_stats_overMem = jobs_stats_df.loc[jobs_stats_df['status'] == 'OUT_OF_MEMORY', :]
    jobs_stats_overTime = jobs_stats_df.loc[jobs_stats_df['status'] == 'TIMEOUT', :]

    jobs_stats_df.loc[jobs_stats_df['status'].isin(['COMPLETED', 'OUT_OF_MEMORY', 'FAILED', 'TIMEOUT']), :].groupby(['name', 'ncpu', 'MemReq', 'status']).max()
    jobs_stats_underMem = jobs_stats_df.loc[(jobs_stats_df['MemEff'] < 50) & (jobs_stats_df['status'] == 'COMPLETED'), :]

    jobs_other = jobs_stats_df.loc[~jobs_stats_df['status'].isin(['COMPLETED', 'OUT_OF_MEMORY', 'TIMEOUT']), :]

    if errors:
        print(f'ERRORS FOUND: {len(errors)}')
        print(errors)

    return jobs_stats_df, jobs_stats_overMem, jobs_stats_overTime, jobs_stats_underMem, jobs_other


if __name__ == '__main__':
    jobs_stats_df, jobs_stats_overMem, jobs_stats_overTime, jobs_stats_underMem, jobs_other = main(daysago=8)
    print(jobs_stats_underMem.head())
    print(jobs_stats_underMem.shape[0])

    # jobs_stats_overTime.groupby(['name', 'ncpu', 'MemReq']).min()

    jobs_stats_df.groupby(['name', 'ncpu', 'MemReq']).max()

    jobs_stats_overMem.groupby(['name', 'ncpu', 'MemReq']).min()

    jobs_stats_underMem.groupby(['name', 'ncpu', 'MemReq']).mean()

    jobs_stats_underMem.groupby(['name', 'ncpu', 'MemReq']).max()


# %%

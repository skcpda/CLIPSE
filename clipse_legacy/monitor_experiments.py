#!/usr/bin/env python3
"""
Monitor SANW experiments on the cluster.
"""
import subprocess
import time
import os
from datetime import datetime


def run_ssh_command(cmd):
    """Run SSH command and return output."""
    try:
        result = subprocess.run(
            f'sshpass -p "900n@M" ssh -o StrictHostKeyChecking=no poonam@172.24.16.132 "{cmd}"',
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)


def check_job_status():
    """Check status of all jobs."""
    stdout, stderr = run_ssh_command("squeue -u poonam")
    if stderr:
        print(f"Error checking jobs: {stderr}")
        return []
    
    lines = stdout.strip().split('\n')
    jobs = []
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if len(parts) >= 6:
                jobs.append({
                    'jobid': parts[0],
                    'partition': parts[1],
                    'name': parts[2],
                    'user': parts[3],
                    'status': parts[4],
                    'time': parts[5],
                    'nodes': parts[6] if len(parts) > 6 else 'N/A'
                })
    return jobs


def check_logs():
    """Check for log files."""
    stdout, stderr = run_ssh_command("ls -la *.out 2>/dev/null || echo 'No output files'")
    return stdout


def monitor_experiments():
    """Main monitoring loop."""
    print("🔍 SANW Experiments Monitor")
    print("=" * 50)
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        jobs = check_job_status()
        if jobs:
            print("📊 Job Status:")
            for job in jobs:
                status_emoji = "🟢" if job['status'] == "R" else "🟡" if job['status'] == "PD" else "🔴"
                print(f"  {status_emoji} Job {job['jobid']}: {job['name']} - {job['status']} ({job['time']})")
        else:
            print("📊 No jobs found")
        
        # Check for output files
        print("\n📁 Output Files:")
        logs = check_logs()
        print(logs)
        
        # Check for any running jobs
        running_jobs = [j for j in jobs if j['status'] == 'R']
        if not running_jobs and jobs:
            print("\n⏳ All jobs are pending...")
        elif not jobs:
            print("\n✅ No jobs in queue")
            break
        
        print("\n" + "="*50)
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    try:
        monitor_experiments()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

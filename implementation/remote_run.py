import paramiko

# Remote connection details
host = "172.24.16.136"
username = "p20200470"
password = "123"

# Connect to the server
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=host, username=username, password=password)

# Run your script
stdin, stdout, stderr = ssh.exec_command("python3 sbert_lpa_baseline.py")
print(stdout.read().decode())
print(stderr.read().decode())

ssh.close()


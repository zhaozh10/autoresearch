## SSH Access to GPU Node

# IMPORTANT！！！！

Do not infer that the password is wrong.
We already verified that this exact password works with:

export SSHPASS='...'
sshpass -e ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no hawking 'echo hello'

In this environment, inline assignment:
SSHPASS='...' sshpass -e ...
is unreliable.

Use a single Bash call with two lines:
1) export SSHPASS='...'
2) run sshpass -e ssh -tt ...

Do not use inline SSHPASS assignment. Do not claim the password is invalid unless the export form also fails for a trivial command.


export SSHPASS='*p!c.:Bx1E#+'
sshpass -e ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no hawking 'cd ~/autoresearch && /home/homesOnMaster/zzhao/.local/bin/uv run test.py'


To run experiments on a SLURM cluster, you must first SSH into the GPU node:
1. To get the allocated node name [NODENAME], run: `squeue --me`
2. 
```bash
export SSHPASS='...'
sshpass -e ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no [NODENAME] 'cd ~/autoresearch && /home/homesOnMaster/zzhao/.local/bin/uv run [python_file]'
```
3. The password is `*p!c.:Bx1E#+`
4. All training commands (`uv run train.py` etc.) must be run on the GPU node, not the login node.
5. If you need to temporarily use a uv environment like `~/uv/[ENV]`, use:

```bash
export SSHPASS='...'
sshpass -e ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no [NODENAME] 'cd ~/autoresearch && /home/homesOnMaster/zzhao/.local/bin/uv run --python ~/uv/[ENV]/bin/python [python_file]'




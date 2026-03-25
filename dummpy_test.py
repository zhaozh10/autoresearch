import torch
print(torch.__version__)
print(torch.cuda.is_available())
print("welcome")


# export SSHPASS='*p!c.:Bx1E#+'
# sshpass -e ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no hawking 'cd ~/autoresearch && /home/homesOnMaster/zzhao/.local/bin/uv run test.py'
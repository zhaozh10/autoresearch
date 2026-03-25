## SSH Access to GPU Node

To run experiments, you must first SSH into the GPU node:

1. Run: `ssh NODENAME` (replace NODENAME with the actual allocated node name)
2. When prompted for password, enter: `*p!c.:Bx1E#+`
3. Once connected, cd to the working directory: `cd ~/autoresearch/`
4. All training commands (`uv run train.py` etc.) must be run on the GPU node, not the login node.

To get the allocated node name, run: `squeue --me`
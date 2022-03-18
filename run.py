import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

if args.n == 0:
    for cl in [0,1]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 1:
    for cl in [2]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 2:
    for cl in [3]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 3:
    for cl in [4,5]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 5:
    for cl in [6]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 6:
    for cl in [7,8]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)

if args.n == 7:
    for cl in [9]:
        subprocess.call(f"python main.py --label {cl} --mode train > logs/{cl}.txt", shell=True)
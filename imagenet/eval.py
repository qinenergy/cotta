from glob import glob
import numpy as np


def read_file(filename):
    lines = open(filename, "r").readlines()
    res = []
    for l in lines:
        if "error" in l and "]: error % [" in l:
            res.append(float(l.strip().split(" ")[-1][:-1]))
    assert len(res)==15
    return np.mean(np.array(res))


def read_files(files):
    res = []
    for f in files:
        res.append(read_file(f))
    print("read", len(files), "files.")
    print(res)
    return np.mean(np.array(res)), np.std(np.array(res))


print("read source files:")
print(read_files(glob("source_*.txt")))

print("read adabn files:")
print(read_files(glob("norm_*.txt")))

print("read tent files:")
print(read_files(glob("tent[0-9]_*.txt")))

print("read cotta files:")
print(read_files(glob("cotta[0-9]_*.txt")))

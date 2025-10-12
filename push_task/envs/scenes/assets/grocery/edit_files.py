import os

for file in os.listdir():
    if file.endswith('obj'):
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            for line in lines:
                if len(line.split())>0 and line.split()[-1].strip().endswith("mtl"):
                    new_line = line.split()[0] + " " + file[:-4] + ".obj\n"
                else:
                    new_line = line
                f.write(new_line)
    elif file.endswith('mtl'):
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            for line in lines:
                if line.split()[-1].strip().endswith("png"):
                    new_line = line.split()[0] + " " + file[:-4] + ".png\n"
                else:
                    new_line = line
                f.write(new_line)
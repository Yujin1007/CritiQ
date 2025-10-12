import os
dims = {}
for file in os.listdir():
    if not file.endswith("obj"): continue
    with open(file, "r") as f:
        lines = f.readlines()
        ys = [1e6, -1e6]
        xs = ys[:]
        zs = ys[:]
        for line in  lines:
            if line.startswith("v "):
                x,y,z  = (map(lambda x: float(x.strip()), line.split()[1:]))
                xs[0] = min(xs[0], x)
                xs[1] = max(xs[1], x)
                ys[0] = min(ys[0], y)
                ys[1] = max(ys[1], y)
                zs[0] = min(zs[0], z)
                zs[1] = max(zs[1], z)
        dims[file[:-4]] = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    
print(f"{dims=}")
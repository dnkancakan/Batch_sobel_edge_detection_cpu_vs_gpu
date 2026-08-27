import os
import random

W = 64
H = 64
COUNT = 120

def make_image(idx):
    random.seed(idx)
    pixels = [[random.randint(0, 40) for _ in range(W)] for _ in range(H)]
    x0 = random.randint(5, W // 2)
    y0 = random.randint(5, H // 2)
    x1 = random.randint(x0 + 5, W - 5)
    y1 = random.randint(y0 + 5, H - 5)
    for y in range(y0, y1):
        for x in range(x0, x1):
            pixels[y][x] = random.randint(180, 255)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    r = random.randint(5, 15)
    for y in range(H):
        for x in range(W):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                pixels[y][x] = 255
    return pixels

os.makedirs("data/input", exist_ok=True)
for i in range(COUNT):
    pixels = make_image(i)
    with open("data/input/image_%03d.pgm" % i, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(bytes([pixels[y][x] for y in range(H) for x in range(W)]))
print("generated %d images in data/input/" % COUNT)

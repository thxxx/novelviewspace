# rays : {"img_path": "r_1.png", "coord": {"x": 1.9533121585845947, "y": -1.6662538051605225, "z": 3.107759475708008}}
import json
from matplotlib import pyplot as plt
from utils import just_line

train_num=30

with open(f"dataset/hotdog_ray.json", "r") as js:
    rays = json.load(js)
    print("ray dataset length : ", len(rays))

with open(f"results/hotdog_0531/valids.json", "r") as js:
    valid_rays = json.load(js)
    print("ray dataset length : ", len(valid_rays))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

on_line_positions = just_line(rays[:train_num], 4, "hotdogs", 512)

# plt.scatter([r['camera_position_on_line'][0] for r in on_line_positions], [r['camera_position_on_line'][2] for r in on_line_positions], c="r")
# plt.scatter([r['coord']['x'] for r in rays[:80]], [r['coord']['z'] for r in rays[:80]], c="b")
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

ax.scatter([r['coord']['x'] for r in rays[:train_num]], [r['coord']['y']
           for r in rays[:train_num]], [r['coord']['z'] for r in rays[:train_num]], c="b")
# ax.scatter([r['origin_position'][0] for r in valid_rays], [r['origin_position'][1]
#            for r in valid_rays], [r['origin_position'][2] for r in valid_rays], c="r", marker='o', s=100)
ax.scatter([r['camera_position_on_line'][0] for r in on_line_positions], [r['camera_position_on_line'][1] for r in on_line_positions], [r['camera_position_on_line'][2] for r in on_line_positions], c="r")
print([r['camera_position_on_line'][2] for r in on_line_positions])

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
# Set the labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig(f'test3d.png', dpi=300)

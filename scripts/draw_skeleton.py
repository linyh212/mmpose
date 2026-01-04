import os, json, cv2
import numpy as np

img_dir, json_file, out_dir = os.sys.argv[1:]
os.makedirs(out_dir, exist_ok=True)

with open(json_file) as f:
    data = json.load(f)

kp_map={}
for ann in data["annotations"]:
    kp_map.setdefault(ann["image_id"], []).append(ann["keypoints"])

skeleton = [
 (5,7),(7,9),(6,8),(8,10),(5,6),
 (5,11),(6,12),(11,12),
 (11,13),(13,15),(12,14),(14,16)
]

for img in data["images"]:
    p = os.path.join(img_dir, img["file_name"])
    im = cv2.imread(p)
    for kp in kp_map.get(img["id"],[]):
        pts=[(int(kp[i*3]),int(kp[i*3+1])) for i in range(17)]
        for a,b in skeleton:
            cv2.line(im,pts[a],pts[b],(0,255,0),2)
        for x,y in pts:
            cv2.circle(im,(x,y),3,(0,0,255),-1)
    cv2.imwrite(f"{out_dir}/{img['file_name']}",im)
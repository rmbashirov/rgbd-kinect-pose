import pickle
import os
import os.path as osp
import cv2
import subprocess


def main():
    with open('/home/vakhitov/Desktop/dump.pickle', 'rb') as f:
        z = pickle.load(f)

    dirpath = '/home/vakhitov/Desktop/tmp/222'
    os.makedirs(dirpath, exist_ok=True)

    index = 0
    for el in z['frames']:
        if 'color' in el:
            img = el['color'][:, :, [0, 1, 2]]
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            cv2.imwrite(osp.join(dirpath, f'{index:06d}.jpg'), img)
            index += 1
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "30",
        "-i", os.path.join(dirpath, "%06d.jpg"),
        "-c:v", "libx264",
        "-vf", "fps=30",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        os.path.join(dirpath, "video.mp4")
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8"))


if __name__ == "__main__":
    main()

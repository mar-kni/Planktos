import cv2
filename= "brownian_sticky_pink"
video_path = f"{filename}.mp4"
output_prefix = "snapshot"

#find the length of the video in seconds


# Choose times (in seconds) to extract frames
times_to_save = [0.3, 4.0, 8.0, 12.0]  # adjust for your sim

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

for t in times_to_save:
    frame_num = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    if success:
        out_path = f"{filename}_{output_prefix}_t{t:.1f}s.png"
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")
    else:
        print(f"Failed to read frame at {t}s")

cap.release()

#mke a figure using matplotlib to display the snapshots
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, len(times_to_save), figsize=(15, 5))
for i, t in enumerate(times_to_save):
    img = plt.imread(f"{filename}_{output_prefix}_t{t:.1f}s.png")
    axs[i].imshow(img)
    axs[i].axis('off')
plt.tight_layout()

plt.show()
#save the figure
fig.savefig(f"{filename}_snapshotsfig.png", dpi=300)

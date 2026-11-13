# Data: where the videos and datasets come from

## Videos to develop against

Start with **free stock footage**. It's legal to use, and it's easy to get several camera angles.

| Source | How to search | Notes |
|---|---|---|
| Pexels Videos (pexels.com/videos) | "restaurant", "cafe interior", "waiter serving", "people eating", "drive thru", "traffic intersection" | Free license, check each clip's page |
| Pixabay Videos (pixabay.com/videos) | same | Free license, check each clip's page |
| Your own recording | Phone on a tripod, high angle, at a café | **Ask for permission.** Never publish identifiable people without consent |

**What makes a good clip:**
- A **fixed camera** (no panning). Zones assume the camera doesn't move.
- A **high angle** looking down at tables, like a CCTV view.
- At least **30 seconds**, ideally with people sitting down, being served and leaving.
- 720p or 1080p.

Save clips in `data/videos/` (git-ignored). Write down each clip's source URL and license in
`data/videos/SOURCES.md`, a habit that matters professionally.

> Avoid downloading from YouTube. It breaks YouTube's terms of service, and you can't
> redistribute the footage in a public demo anyway.

## Datasets

| Dataset | Used for |
|---|---|
| **COCO** (via pretrained YOLO) | Baseline detection: `person`, `cup`, `bowl`, `wine glass`, `fork`, `knife`, `spoon`, `dining table`, `pizza`, `sandwich`, `cake`, `car`, `truck`, `motorcycle`… |
| **Food-101** | Dish recognition (classification) on plate crops |
| **UECFOOD-256** | Food detection with bounding boxes |
| **Roboflow Universe** | Community datasets: search "restaurant", "plate", "waiter", "drive thru". Check licenses |
| **MOT17** (motchallenge.net) | Evaluating person tracking |
| **UA-DETRAC** | Evaluating vehicle detection and tracking |
| **Own labels** (CVAT or Roboflow) | 300–1,000 frames per view: `staff`, `guest`, `plate_full`, `plate_empty`, vehicles at the drive-thru |

## Privacy rules for this project

1. Track IDs are temporary and never linked to a real identity.
2. No faces in the public repo or demo unless the footage is stock with a license that allows it.
3. `--privacy` mode blurs the upper third of every person box in saved output.
4. No license plate reading; vehicles are anonymous tracks.

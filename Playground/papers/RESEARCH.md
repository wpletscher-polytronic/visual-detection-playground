# Bullet Hole Detection — Research Summary

## Insight
- Niche field. Low research volume.
- Most work still uses YOLOv8. ~2 generations behind current SOTA.
- Best published score: YOLOv8m, 96.7% mAP50 (Butt et al., 2023). Near ceiling.
- Bottleneck = data, not architecture. Overlapping holes, boundary ring hits, limited target-design diversity, labeling time.
- Classical CV (thresholding, blob detection, Hough) fails on watermarks/digits/varying contrast. Confirmed by our own tests.
- YOLOv11/v12 barely tried in this niche yet. Low-risk to test. Expect marginal gains only.
- Bigger lever: foundation-model auto-labeling (Grounding DINO/SAM). Cuts labeling cost. Could fund broader dataset.



## Sources

### Butt et al., 2023 — Inholland Univ., NL — Paper
Name: "Application of YOLOv8 and Detectron2 for Bullet Hole Detection and Score Calculation from Shooting Cards" — Marya Butt et al., *AI* (MDPI) 2024, 5(1), 5.
Link: https://doi.org/10.3390/ai5010005 · preprint: https://www.preprints.org/manuscript/202310.0657/v1/download
Summary: Benchmarks 7 detectors on 26×26cm 10-ring shooting cards, ring-per-class (score 1-10) rather than generic "hole" class. The reference result for the field.
Tools: YOLOv8 (n/s/m); Detectron2 (FasterRCNN-50/101, RetinaNet-50/101); Roboflow (annotation, augmentation).
Data: 1243 imgs; ~40% synthetic; 13 classes; 1 target design. (Paper reports 2784 train / 202 val after augmentation.)
Scores: YOLOv8m 96.7% mAP50; YOLOv8s 96.5% mAP50 at 2.3ms; best Detectron2 (FasterRCNN-101) 83.6% AP50.
Bottleneck: overlapping holes; boundary hits misclassified one ring high; class "10" underrepresented.
Takeaway: YOLO beats Detectron2 here. YOLOv8s = best speed/accuracy. Needs more edge-case data.

### Hah Wai Yin et al., ICAROB2025 — Malaysia — Paper
Name: "Detection of Bullet Holes for Target Board in Malaysia Army (ATM) Shooting Exercise Environment" — Jillian Hah Wai Yin et al., ICAROB 2025, Oita, Japan (Feb 13-16).
Link: https://alife-robotics.co.jp/members2025/icarob/data/html/data/OS/OS14/OS14-3.pdf
Summary: Edge deployment on real military range — images captured in situ, not lab conditions. Deployment story, not an accuracy contribution.
Tools: YOLOv8n vs YOLOv5; Raspberry Pi 4B + camera; Blynk IoT app.
Data: military base target boards.
Scores: not specified.
Bottleneck: inconsistent detection; holes missed.
Takeaway: field-tested, real deployment. Not mature yet.

### Belcher et al., Jan 2026 — arXiv 2601.17062 — Paper
Name: "A Computer Vision Pipeline for Iterative Bullet Hole Tracking in Rifle Zeroing" — Robert M. Belcher, Brendan C. Degryse, Leonard R. Kosta, Christopher J. Lowrance.
Link: https://arxiv.org/abs/2601.17062
Summary: Tracks holes *across* successive firing rounds so new shots are separable from old ones — ORB homography for perspective, IoU for identity. Notable trick: augmentation that *removes* holes to synthesise earlier rounds.
Tools: YOLOv8 (detection); IoU matching (tracking); ORB + homography (alignment across shots).
Data: 33 sequences; 22 synthetic + 11 live-fire.
Scores: 97.0% mAP on hole detection; 88.8% accuracy assigning holes to the correct firing iteration.
Bottleneck: not specified.
Takeaway: confirms homography-alignment approach works. Still YOLOv8.

### Competition scoring paper — via ResearchGate snippet — Paper
Name: not resolved to a title/DOI. Describes the "Assistant RDC" Android app for shooting-competition referees (RDC = results determination commission).
Link: none found — snippet only.
Summary: YOLOv11 detects holes of mixed calibers, then a geometric size model classifies each hole; aimed at replacing manual referee scoring at large competitions.
Tools: YOLOv11; Android app ("Assistant RDC").
Data: not specified.
Scores: not specified.
Bottleneck: not specified.
Takeaway: 2nd confirmed real use of YOLOv11 for this task. Details unverified.

### "Project bat bullet hole detection" — Roboflow — Dataset
Name: "Bullet hole object detection" (also a 44-img "Carbine target bullet hole detection" variant).
Link: https://universe.roboflow.com/project-bat-bullet-hole-detection/bullet-hole-object-detection
Summary: The Butt et al. dataset and trained model, hosted with a hosted inference API. Best available starting point — real labels, real target design.
Tools: hosts Butt et al. model/dataset.
Data: 1243 imgs; 13 classes; CC BY 4.0.
Scores: = Butt et al.
Bottleneck: = Butt et al.
Takeaway: direct API access to the same model.

### Andrew Crook, 2021 — Roboflow — Dataset
Name: "Bullet Hole" by Andrew Crook (Nov 2021).
Link: https://universe.roboflow.com/andrew-crook/bullet-hole
Summary: Small, early upload; classes split by hole appearance (bullet hole, behind, colour) rather than by ring score. Useful only as extra appearance variety.
Tools: none published.
Data: 140 imgs; 5 classes; CC BY 4.0.
Scores: none.
Bottleneck: none.
Takeaway: raw dataset only, no study attached.

### minimaxer, 2023 — Roboflow — Dataset
Name: "Bullet Hole" by minimaxer (Dec 2023).
Link: https://universe.roboflow.com/minimaxer-t1ljj/bullet-hole-t969z
Summary: 103 images with a pre-trained model and API. Too small to train on alone; possible eval-set filler.
Tools: none published.
Data: 103 imgs; CC BY 4.0.
Scores: none.
Bottleneck: none.
Takeaway: raw dataset only, no study attached.

### KNSAshootingtargets, 2022 — Roboflow — Dataset
Name: "KNSA-shooting-target" (Jun 2022). KNSA likely a national shooting association.
Link: https://universe.roboflow.com/knsashootingtargets/knsa-shooting-target
Summary: 92 images, single "bulletholes" class. Smallest of the set; different target design, so useful for generalisation checks.
Tools: none published.
Data: 92 imgs.
Scores: none.
Bottleneck: none.
Takeaway: creator identity unconfirmed.

### bailin tian — Roboflow — Dataset
Name: "Bullet Hole" by bailin tian (workspace `bailin-tian-scyq7`).
Link: direct project URL unconfirmed — findable via https://universe.roboflow.com/search?q=class:bullet+hole
Summary: Largest informal upload found (1.11k imgs), but no documentation, licence, or study attached — provenance unknown.
Tools: none published.
Data: 1.11k imgs.
Scores: none.
Bottleneck: none.
Takeaway: largest informal upload found.

### Kanat, 2025 — Roboflow — Dataset
Name: "Kanat2.0 - YoloV11 - Bullets, Targets and Boards".
Link: https://universe.roboflow.com/kanat-hefkh/kanat2.0-yolov11-bullets-targets-and-boards
Summary: Detects boards and targets alongside bullet_hole (4 classes), so it localises the target too — but the published metrics are weak.
Tools: YOLOv11 (Roboflow 3.0).
Data: 1201 imgs (556 in the earlier version); Public Domain.
Scores: mAP50 52.1%; precision 63.2%; recall 47.5%.
Bottleneck: low recall — misses roughly half the holes.
Takeaway: confirms YOLOv11 tried on this task, and that a casual YOLOv11 run does not beat Butt et al.

### thiagodsd/bullet-from-a-gun — GitHub — Repo
Name: "bullet-from-a-gun" — benchmarking CNNs for gunshot hole detection across surfaces/materials.
Link: https://github.com/thiagodsd/bullet-from-a-gun
Summary: Compares CNN detectors on varied materials, calibers, and distances — not just paper targets. Python 3.10 / PyTorch 2.0.
Tools: CNN comparison; PyTorch 2.0.0+cu117.
Data: various materials, not just paper targets.
Scores: unknown.
Bottleneck: unknown.
Takeaway: broader scope. Not reviewed in depth.



## Deployment Status

### Our sources — none are shipping products

- Butt et al., 2023: not deployed. Model + dataset public, no app or company behind it.
- Hah Wai Yin, ICAROB2025: not deployed. Paper itself admits inconsistent detection.
- Brno "OnPoint" thesis: not deployed. No matching app found on any store.
- Belcher et al., arXiv 2601.17062: not deployed. Research pipeline, 33-sequence test set only.
- Competition scoring paper: unclear, likely pilot only. Unverified.
- All datasets / repos: not products.

### Industry — hardware electronic targets

Acoustic/piezoelectric or real-time optical sensing. A different problem to ours: they
measure a bullet crossing a plane as it happens, we score a photo after the fact.

**Kongsberg Target Systems** — Norway
Proprietary sensor hardware. Used by the US Civilian Marksmanship Program; established player.

**INTARSO** — SQ55 / TrueScore10
SQ55 = acoustic/piezoelectric. TrueScore10 = 2 high-speed cameras + LED strip, real-time.
Marketing figures only, no published accuracy.

**TTS (Theissen Training Systems)** — LOMAH / Box Targets
Acoustic sensors. Military/LE training focus, hardware-based.

### Industry — photo + AI apps (directly comparable to our task)

**ShotScore** — iOS/Android
On-device AI, works offline. Falls back to manual zone adjustment.
Live consumer product, closest direct comparison to our task.

**TargetScan ISSF Pistol & Rifle** — Android (dev: gabrowski)
Automatic pattern/hole recognition; "Eagle Eye" manual refine for close shots.
Most feature-rich found: group stats (MPI, mean radius, extreme spread), session
tracking, PDF export.

**Eley X-Shot** — Eley ammunition
Image recognition, but requires proprietary target sheets + black background for contrast.
Uncertain scores flagged red for manual correction. Reviewer scored 132-3X vs 131-2X by hand.
Tied to an ammo company that sells the matching sheets.

**TargetIQ** — browser-based
Detects card + rings from the print itself, then applies the ISSF geometric gauge rule.
10m Air Rifle only so far; explicitly not ISSF-certified. Newest and most active (2026).
User corrections reportedly feed back into the model.

**My Shooting / BallisticX** — manual-mark group analysis
No detection at all, just tap-to-mark. That these still sell suggests full auto-detection
isn't considered solved industry-wide.

### IP note

US Patent 10060713 — "shooting game… with dynamic shot position recognition on a paper
target." Existing IP in this space; worth checking before any commercial deployment.

# Submission checklist — Clearity

Repository: https://github.com/Asylkenei/clearity

| Requirement | Current status |
| --- | --- |
| Upload MOV through web UI | Implemented |
| Asynchronous analyze + polling API | Verified on Render |
| X-API-Key authentication | Implemented |
| Binary verdict | Experimental trained classifier |
| Timestamp + 1–3 JPEG frames per event | Implemented |
| Multiple possible events | Adjacent positive samples grouped into up to 12 events |
| Specific event type | Gap: only generic binary concerning-pattern class |
| Explanation of the actual visible action | Gap: binary model explains similarity, not semantic action |
| Kazakhstan regulatory source | Context link included; precise event-to-clause mapping unfinished |
| Local inference / no external AI APIs | Implemented |
| No face identification | Implemented |
| Portable after weight download | Implemented |
| Public hosted URL while laptop is off | Verified: https://clearity-3qoy.onrender.com |
| Processing below 5 minutes | Hosted synthetic 145-second / 29-MiB MOV: 63.59 seconds (warm service) |
| GitHub public and English documentation | Public repository; English README published |
| Presentation | 7-slide English PDF/PPTX and speaker script provided in presentation/deliverables/ |

Web interface URL: https://clearity-3qoy.onrender.com

API base URL: https://clearity-3qoy.onrender.com

Presentation: [PDF](presentation/deliverables/Clearity_Pitch_English.pdf).

Verified unauthenticated API requests: HTTP 401. Private dataset-review route: HTTP 404.
Jury key: shared privately after deployment; not stored here.
Do not submit a fabricated URL or claim that all substantive requirements are satisfied.

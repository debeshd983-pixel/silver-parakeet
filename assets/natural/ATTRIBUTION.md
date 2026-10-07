# Attribution — `assets/natural/`

## What these are

Twelve reference photographs downloaded from **Wikimedia Commons** on 2026-10-08 and used
as real-photograph fixtures to measure the detector's false-positive rate on natural
scenes. They are **not** used by the `sahu65` package at runtime and are **excluded from
both the wheel and the sdist** (`MANIFEST.in` prunes `assets/`).

Original filenames, in the order they were saved:

| Local file | Wikimedia Commons original |
|---|---|
| `nat00.jpg` | `Amur Tiger Panthera tigris altaica Eye 2112px edit.jpg` |
| `nat01.jpg` | `Bengal tiger (Panthera tigris tigris) female 3.jpg` |
| `nat02.jpg` | `Panthera tigris altaica 13 - Buffalo Zoo.jpg` |
| `nat03.jpg` | `Panthera tigris sumatran subspecies.jpg` |
| `nat04.jpg` | `Sunflower and a bee.jpg` |
| `nat05.jpg` | `Sunflower head 2015 G1.jpg` |
| `nat06.jpg` | `Sunflower head 2026 G1.jpg` |
| `nat07.jpg` | `Sunflower macro wide.jpg` |
| `nat08.jpg` | `Eiffel Tower and Pont Alexandre III at night.jpg` |
| `nat09.jpg` | `Eiffel Tower in 2022 02.jpg` |
| `nat10.jpg` | `Lightning striking the Eiffel Tower - NOAA.jpg` |
| `nat11.jpg` | `Louis-Emile Durandelle, The Eiffel Tower - State of...` (1878 lithograph) |

Note `nat11.jpg` is **not a photograph** — it is an 1878 lithograph. It is included
deliberately as an adversarial case: a human-made artwork that is not a camera photo.

## ⚠ ACTION REQUIRED BEFORE MAKING THIS REPOSITORY PUBLIC

**The individual license of each file has not been verified.** Wikimedia Commons files
carry a wide variety of licenses (many are CC BY-SA, some CC BY, some public domain, and
at least one here is a NOAA image which may carry separate terms). The `natXX.jpg`
renaming also stripped the attribution metadata that Commons requires.

Before pushing this repo to a public remote, either:

1. **Verify and record each file's license**, restoring the required attribution
   (author + license + source URL) into this file; or
2. **Remove `assets/natural/`** and keep only the measurements recorded in
   `MODEL_CARD.md` §3.1.

The measurements in `MODEL_CARD.md` do not depend on the files remaining in the repo.
`tests/test_classifier.py` skips its asset-based assertion when these fixtures are absent,
so removing them will not break CI.

## Other assets

`assets/ai/` and `assets/rl/` were supplied by the project owner and contain identity
documents. They are sensitive: `assets/rl/` holds an Aadhaar card and a stamped land
record. **Consider whether this repository should ever be public**, and redact those
files first if so.
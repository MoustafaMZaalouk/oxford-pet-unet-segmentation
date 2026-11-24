# License evidence pack (Oxford-IIIT Pet)

Captured live from https://www.robots.ox.ac.uk/~vgg/data/pets/ (see `retrieved_utc.txt`).

- `license_page.html`, `license_page.sha256`, `license_page_headers.txt`: raw page, its hash, HTTP headers.
- `license_excerpt.txt`: the license statement ("available to download for commercial/research purposes under a Creative Commons Attribution-ShareAlike 4.0 International License. The copyright remains with the original owners of the images.").
- `cc_by_sa_4.0_deed.html`: the CC BY-SA 4.0 deed from creativecommons.org.
- `download_checksums.txt`: SHA-256 of the dataset archives as downloaded (pinned in `src/petseg/data.py`).

Note: the page states the license for the dataset; copyright in individual images stays with original owners, so the repo does not redistribute any image.

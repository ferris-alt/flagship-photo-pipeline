# flagship-photo-pipeline
Gallery photo pipeline for flagshipcarsandclassics.com: BiRefNet mask -> showroom composite -> color normalize.
The car's pixels are copied, never regenerated. See FLAGSHIP-GALLERY-PIPELINE-PLAN.md in the Claude chat for the runbook.
Trigger: `gh workflow run process.yml -f stocks=P1063A` (or `all`). Results: one artifact per car (`car-<STOCK>`) with originals/, out/, qa/contact-sheet.jpg, report.json.

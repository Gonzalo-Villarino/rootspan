# Data and reproducibility

## Saved local release and public copy

The local `Periderm_Lab_0.6.0_Test_Lock_20261117` remains untouched. RootSpan `0.6.0-rootspan.1` is a separately named publication copy with a renamed interface, portable project-root paths, installation instructions and a public-package integrity manifest. Its original measurement calculations and learned parameters are preserved.

`docs/provenance/local_approved_release_manifest.json` records the original local release. `ROOTSPAN_MANIFEST.json` describes the current public runtime. These are different manifests for different packages. Neither is an independent signature or accuracy certificate. Do not replace the original local manifest with the public one.

## Files included

The repository contains the learning scripts, Word manuals, scientific experiment code, compact results, example crops and annotations, current app, software tests, source provenance and the small learned transition head. Original PAT source files keep their MIT licence and are pinned to commit `51cdf8966f6da17b711aeb5625205a8a2ca39467`.

The export inventory records source and initial public-copy hashes. Some packaging files were subsequently added or renamed; the runtime manifest identifies the delivered active runtime. Historical JSON and plain-text notes have local home/project paths replaced with placeholders. Where a label was programmatically prepared, it remains identified as such rather than being recast as independently verified human ground truth.

## Files kept locally

The complete working project remains on Gonzalo's Mac. The public repository omits installed Python environments, caches, duplicate uploads, authentication records and private review conversations. Most large source microscopy slides (except the original MT25 practice image), most large intermediate masks and historical learned weight files are not bundled. `docs/provenance/large_assets_not_bundled.json` lists size-filtered source/intermediate assets; it is not a list of every excluded cache or duplicate.

This is a source-and-results publication, not a byte-for-byte backup of the entire local workspace. Historical scripts can require missing source images, old model files or original directory layouts. To replay an experiment, restore its inputs from the local project using the recorded manifests. The current app instead accepts a newly uploaded image and obtains its base model during setup.

## Original PAT weights

The upstream repository provides a model download link. A separate redistribution licence for its weights was not established during this project, so this repository links to that download instead of hosting the large model. Download it yourself from the authors; the installer verifies the original file before loading only the explicitly allowlisted model classes.

- Original checkpoint SHA-256: `fb07dc1f1a254f7ba067de7f9b8b491bc7bb7f1c8da4564159ad23bcbc208f90`
- Converted tensor-only state SHA-256: `71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348`
- Small transition head SHA-256: `6cce6ef74330a9de4422cdb738144f972cedff41f07d4de3e9c9869b6e4054b9`

The original checkpoint deserialization is restricted to the exact known input hash and known classes, and uses `weights_only=True`. A different file is rejected. Downloads and third-party packages remain subject to their own terms.

## Environment and verification

Main environment and cleanup environment package versions are recorded separately in `requirements/`. The original observed platform was Python 3.13.2, Apple silicon, using MPS when available. macOS font paths remain in historical visualisation code. This release is not advertised as cross-platform.

See `docs/verification.json` for checks actually completed on the publication copy. Reusing the recorded local environments is not the same as testing installation on a clean computer. Dependency availability from an online registry and exact numeric results on another machine are not guaranteed by a local test. Large acquisition files are needed to reproduce full-slide results.

## Setup recovery

A failed conversion removes its temporary output. If either `00_Software/PAT_Reference/pat_state_dict.pt` or `06_Models/Locked_Transition_v1_20261116/weights/pat_state_dict.pt` has the wrong checksum, setup preserves it and stops. Move that file to a backup outside the project, verify the original download and pinned package versions, and rerun setup. Do not replace the expected checksum simply to force acceptance.

The launcher requires executable permission. If necessary, run `chmod +x RootSpan.command` and then `./RootSpan.command`.

# Research examples and annotation provenance

These examples come from Gonzalo Villarino's root microscopy project and archived Salk/PAT working material. This public project shares selected research crops, annotation records and results at his direction. A separate open-data licence for the underlying microscopy collection has not been established here; the code's MIT licence does not relicense those images. Consult the original research/data source before reusing the microscopy collection beyond this project. No independent claim of institutional data clearance is made.

## Which labels are which?

- `02_My_First_Annotations` and `03_Background_Annotations`: Labelme practice annotations and later refinements. These are learning/development examples, not held-out test labels.
- `05_Training_and_Evaluation/03_Assistant_Prepared_Adaptation`: **programmatically prepared sparse teaching polygons**, based on Gonzalo's biological identification of regions. Exact polygons were not independently verified by him. Their JSON records retain this distinction.
- `08_Endoderm_Drafts_NEED_REVIEW`: derived endoderm drafts; the folder name records the need for review.
- `09_Transition_Endpoint_Review`, `10_Cell_Appearance_Review` and later review folders: region/path/transition observations used during model development. Feedback here must not be counted as independent test accuracy.
- Archived annotation folders retain their source manifests. Original annotator identity should be taken from those source records, not inferred to be Gonzalo for every file.

`annotation_provenance_index.json` lists the included Labelme-style JSON files with their recorded provenance when available. A missing provenance field is reported as unspecified, not silently filled with a human author.

All 86 included source JSON files under `02_Data` were scanned (excluding the generated provenance index); 18 are Labelme-style annotation files. None has a nonempty `imageData` field. Embedded images were removed from nine original annotation JSONs while preserving their polygons and labels; the export hashes are recorded in `docs/provenance/annotation_export.json`. Image crops are separate files. The original MT25 practice PNG is included alongside its annotations. Other large original microscopy slides are kept in the local project; see the omitted-assets inventory. The Word manuals are preserved historical learning records; inspection found no comments, tracked changes or email addresses. Some retained example commands contain Gonzalo's local home path, and document metadata/learning text records the tools used historically.

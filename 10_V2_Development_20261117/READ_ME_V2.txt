V2: longer inspection, same detector

The pipeline previously stopped looking at cells at the same place where it stopped measuring. This could hide the Periderm evidence needed to identify a transition. V2 looks along the complete detected track, then measures only up to the estimated root endpoint. It uses exactly the V1 detector weights and transition rule. No training occurred.

Execution checks (not an accuracy score):
Slide        V1 provisional lengths    V2 provisional lengths
1_7          3                         4
22_22        5                         5
7_18         1                         2
Col_rep1     6                         6
WOX4_rep1    3                         5
All previously returned transition positions and measuring endpoints were unchanged.

7_18 has seven detected roots. V2 returns provisional periderm lengths for roots 6 and 7. Roots 2 and 3 have T-to-visible-end lengths only because the root-hypocotyl junction is unresolved. Roots 1 and 5 have a conflict between the T candidate and the endpoint. Root 4 lacks a supported transition. Blank values must not become zero or invented lengths.

What still needs biological checking: the new blue T lines, the orange endpoint, and the automatically followed main root. V2 is a pipeline fix, not proof that the detector is accurate on all roots. V1 is retained in the GUI selector and its locked files are unchanged.

Next learning step: inspect the recovered blue T for 7_18 root 7. Then check its orange endpoint before accepting the length. The locked detector is still based on very few reviewed appearance windows. More representative teaching and independent evaluation will be needed for a validated autonomous system.

The original PAT four-class segmentation decoder was also checked diagnostically. It did not consistently resolve the ambiguous roots and was not substituted into V2. Nothing was retrained or tuned on reserved slide 15_27.

Files: code/settings under 06_Models/Periderm_Pipeline_V2_20261117; diagnostics, comparison and regression outputs under 10_V2_Development_20261117.

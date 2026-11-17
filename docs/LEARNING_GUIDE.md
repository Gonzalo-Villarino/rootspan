# The learning journey

These records follow Gonzalo Villarino's practical learning exercise. The main root example began as one microscopy slide with six roots.

1. **Images and pixels.** An image is a grid of tiny coloured squares.
2. **Manual segmentation annotations.** In Labelme, draw outlines around a tissue and name its category. The JSON stores the outline coordinates and labels; it can also embed the image.
3. **Pixel mask.** Fill those outlines with numeric category codes. This is the answer key derived from the annotations, not a model prediction.
4. **Matching patches.** Cut the same area from the image and mask. A 1024 × 1024 patch has 1,048,576 pixel positions. Patches make large images manageable while preserving local detail.
5. **Small U-Net exercise.** PyTorch runs the model. During training, the image is the input and the mask supplies the correct answers for labelled pixels. Repeating updates adjusts model parameters.
6. **Prediction.** A trained model receives the image alone. A reference mask is used afterwards to assess its prediction, not as prediction input.
7. **Troubleshooting.** Background examples, clearer outlines and more updates improved training fit. Nearly perfect agreement on one training patch did not guarantee good predictions elsewhere.
8. **Broader examples and PAT.** The later experiments reused archived examples and the published PAT model. Some additional labels were programmatically prepared or derived from rules, rather than manually verified pixel by pixel. See the data provenance guide. Development feedback and training labels are recorded separately from independent evaluation.
9. **Transition and length.** Tissue predictions, transition rules, root geometry, calibration and the measurement definition all affect the final number.
10. **RootSpan.** The browser interface brings the current analysis together. It remains a prototype that needs independent measurement validation.

## Word manuals

- [Plant image annotations](../05_Notes/01_Plant_Image_Annotations.docx)
- [Labelme step-by-step manual](../05_Notes/02_Labelme_Step_by_Step_Manual.docx)
- [Periderm length proof walkthrough](../05_Notes/06_Periderm_Length_Proof_Walkthrough.docx)
- [Model refinement review](../05_Notes/07_Model_Refinement_Review_20261116.docx)

The Word documents are preserved learning records, including earlier commands and local paths. They are not a replacement for the current installation guide. The scripts in `03_Notebooks` retain their original order. Scripts that expect large source slides or omitted historical weight files need those assets restored first; the public repository is not a turnkey replay of every historical experiment.

## Mask codes are not model scores

In the three-category teaching mask: 1 = Periderm, 2 = Endoderm, 3 = Background. Code 0 is unlabelled and 255 is conflicting; both were ignored during training. These are category identifiers, not confidence percentages. The model computes a score for each category at each pixel and chooses a category. The later original PAT four-class model uses a different code scheme, documented in its own files.

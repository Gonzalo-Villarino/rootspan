ONE MASK, THREE CLASSES

The *_mask.png file is the full-size numeric answer key: 1 Periderm, 2 Endoderm, 3 Background.
0 is unannotated and 255 means conflicting labels. Both must be ignored during training.
The *_mask_preview.png is only a smaller colour picture: red Periderm, green Endoderm,
yellow Background, grey unannotated, pink conflict. Small regions can disappear in this preview.
mask_details.json records the inputs, hashes, class codes and pixel counts.

This program filled your saved outlines. It did not train or change a model or update patches.
The earlier two-class training scripts cannot use this new mask without adaptation.

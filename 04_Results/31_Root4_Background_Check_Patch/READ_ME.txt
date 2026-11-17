ROOT 4: PREPARING A NEW EXAMPLE

The microscopy image will be the model input during prediction.
Your mask is kept separately to check the prediction afterwards.
The model does not receive the mask when making its prediction.
This program fills your current saved outlines using program 07,
then cuts the same 1024 x 1024 square from image and mask.
1=Periderm, 2=Endoderm, 3=Background, 0=unlabelled, 255=conflict.
Grey means unlabelled, not Background. Ignore 0 and 255 when scoring.
The colour preview is for viewing; the numeric mask holds the codes.
This crop does not overlap the selected model's root 5 training crop.
Both crops are from one slide: this is a preliminary check only.
No prediction or training has occurred. Original files unchanged.

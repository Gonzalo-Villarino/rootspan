args <- commandArgs(trailingOnly=TRUE)
out <- normalizePath(args[1],mustWork=TRUE)
d <- jsonlite::fromJSON(file.path(out,"comparison_results.json"))
before <- png::readPNG(file.path(d$baseline_prediction,"predicted_mask_preview.png"))
after <- png::readPNG(file.path(d$updated_prediction,"predicted_mask_preview.png"))
mic <- png::readPNG(d$image_patch)
# Same original-pixel crop in every panel: x=255..766, y=646..901.
cut <- function(im) im[647:902,256:767,,drop=FALSE]
png(file.path(out,"root5_before_after_closeup.png"),width=1640,height=570,res=130,type="cairo")
par(mar=c(0,0,0,0),xaxs="i",yaxs="i",family="sans")
plot.new();plot.window(xlim=c(0,1640),ylim=c(570,0))
text(28,36,"Root 5: adding Background below the root",adj=c(0,0.5),cex=1.65,col="#17212D")
text(28,78,"Same microscopy patch and U-Net; 600 updates in each run. Only Background labels were added.",adj=c(0,0.5),cex=1.0,col="#465566")
names <- c("Microscopy: same area","Before: model prediction","After: model prediction")
ims <- list(mic,before,after)
for (j in 1:3) {
 x <- 28+(j-1)*536
 text(x,127,names[j],adj=c(0,0.5),cex=1.1,col="#17212D")
 rasterImage(cut(ims[[j]]),x,420,x+512,164,interpolate=FALSE)
}
text(28,458,"Red = Periderm     Green = Endoderm     Yellow = Background",adj=c(0,0.5),cex=1.05,col="#17212D")
text(28,497,sprintf("Green pixels below the Periderm outline, across the full patch: %s before; %s after.",format(d$lower_band$old_green_pixels,big.mark=","),format(d$lower_band$new_green_pixels,big.mark=",")),adj=c(0,0.5),cex=1.0,col="#17212D")
text(28,537,"This is the training example. Unlabelled edges still lack confirmed answers; new-image performance is unknown.",adj=c(0,0.5),cex=0.94,col="#465566")
dev.off()

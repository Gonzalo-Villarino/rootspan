# Plot an existing training record. No image processing or model training.
# Learning exercise. # Run: Rscript --vanilla /absolute/path/to/13_plot_learning_curve.R
# Optional arguments: saved_training_run_folder output_folder

suppressPackageStartupMessages(library(jsonlite))
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) != 1L) stop("Run this program with Rscript.")
script_file <- normalizePath(sub("^--file=", "", script_arg))
project <- dirname(dirname(script_file))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) stop("Expected at most two arguments: run folder and output folder.")
run_dir <- if (length(args) >= 1L) args[1] else file.path(
  project, "04_Results/12_Refined_Training/run_20261114T102736Z_owxsbwtv")
output_dir <- if (length(args) >= 2L) args[2] else file.path(
  project, "04_Results/14_Learning_Curve")
run_dir <- normalizePath(run_dir, mustWork = TRUE)
if (file.exists(output_dir)) stop("Output folder already exists; nothing was overwritten.")
history_file <- file.path(run_dir, "training_history.csv")
record_file <- file.path(run_dir, "trial_details.json")
weights_file <- file.path(run_dir, "model_weights.pt")
inputs <- c(history_file, record_file, weights_file, script_file)
hashes_before <- tools::md5sum(inputs)
history <- read.csv(history_file, check.names = FALSE)
record <- fromJSON(record_file)
n <- record$completed_updates
if (length(n) != 1L || !is.finite(n) || n < 1 || n != as.integer(n) ||
    nrow(history) != n || !identical(names(history), c("update", "loss_before_update")) ||
    !all(history$update == seq_len(n)) ||
    !all(is.finite(history$loss_before_update)) ||
    any(history$loss_before_update < 0)) stop("Invalid or incomplete training history.")
final_loss <- record$after_training$loss
if (length(final_loss) != 1L || !is.finite(final_loss) || final_loss < 0)
  stop("Missing or invalid final loss measurement.")
if (!isTRUE(all.equal(history$loss_before_update[1], record$before_training$loss,
                     tolerance = 1e-6))) stop("Starting loss disagrees with the record.")

# Loss printed for update 1 is measured BEFORE update 1: zero updates completed.
# The separate final loss is measured AFTER all n updates. Preserve that timing.
points <- data.frame(
  completed_updates = c(history$update - 1L, n),
  loss = c(history$loss_before_update, final_loss),
  measurement = c(rep("Before next update", n), "After final update")
)

draw_curve <- function() {
  par(mar = c(6.0, 5.5, 5.0, 1.4), family = "sans", las = 1,
      col.axis = "#46515E", col.lab = "#202B36", fg = "#46515E",
      cex.axis = 0.95, cex.lab = 1.1)
  ymax <- ceiling(max(points$loss) * 5) / 5
  if (ymax == 0) ymax <- 0.2
  xticks <- pretty(c(0, n), n = 8)
  xticks <- xticks[xticks >= 0 & xticks <= n]
  yticks <- pretty(c(0, ymax), n = 6)
  yticks <- yticks[yticks >= 0 & yticks <= ymax]
  plot(points$completed_updates, points$loss, type = "n", axes = FALSE,
       xlim = c(0, n), ylim = c(0, ymax), xaxs = "i", yaxs = "i",
       xlab = "Completed training updates",
       ylab = "Loss (lower = closer to your labels)")
  abline(h = yticks, v = xticks, col = "#E5EAF0", lwd = 0.8)
  axis(1, at = xticks, tck = -0.015)
  axis(2, at = yticks, tck = -0.015)
  box(bty = "l", col = "#7B8795")
  lines(points$completed_updates, points$loss, col = "#2563A6", lwd = 2.4)
  points(n, final_loss, pch = 21, bg = "#D55E00", col = "white", cex = 1.35)
  text(n * 0.98, max(0.05 * ymax, final_loss - 0.075 * ymax),
       sprintf("After %d updates: %.3f", n, final_loss),
       adj = c(1, 1), col = "#A34300", cex = 1.0)
  title(main = "Learning from your refined annotations", adj = 0,
        col.main = "#17212B", cex.main = 1.4, line = 2.6)
  mtext(sprintf("One microscopy patch | %d updates | Same three-category U-Net", n),
        side = 3, adj = 0, line = 1.25, cex = 0.95, col = "#566170")
  mtext("Saved training numbers only. No new training was run to make this graph.",
        side = 1, line = 4.2, cex = 0.83, col = "#566170")
}

dir.create(dirname(output_dir), recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(pattern = ".learning-curve-", tmpdir = dirname(output_dir))
dir.create(temporary)
tryCatch({
  png(file.path(temporary, "refined_training_learning_curve.png"),
      width = 1500, height = 960, res = 140,
      type = if (capabilities("cairo")) "cairo" else getOption("bitmapType"))
  tryCatch(draw_curve(), finally = dev.off())
  pdf(file.path(temporary, "refined_training_learning_curve.pdf"),
      width = 1500 / 140, height = 960 / 140, useDingbats = FALSE)
  tryCatch(draw_curve(), finally = dev.off())
  write.csv(points, file.path(temporary, "plotted_data.csv"), row.names = FALSE)
  details <- list(
    purpose = "Display an already completed training run; no training occurred",
    training_run = run_dir, training_updates = n,
    source_history = history_file, source_record = record_file,
    plotted_measurements = nrow(points),
    first_loss = points$loss[1], final_loss = final_loss,
    horizontal_axis = "Number of completed training updates",
    timing = paste("CSV losses are measured before each update, plotted at update minus one.",
                   "Final loss comes from after_training.loss, plotted after all updates."),
    smoothing_applied = FALSE, data_values_modified = FALSE,
    model_trained = FALSE, input_md5 = as.list(hashes_before),
    script_file = script_file, R_version = R.version.string,
    limitation = "Training loss on labelled pixels of the same patch; no independent test."
  )
  write_json(details, file.path(temporary, "graph_details.json"),
             auto_unbox = TRUE, pretty = TRUE, digits = NA)
  writeLines(c(
    "LEARNING CURVE: REFINED ANNOTATIONS", "",
    "The PNG is the graph for viewing; the PDF is an exportable copy.",
    "plotted_data.csv contains every displayed point, without smoothing.",
    "graph_details.json records the inputs and explains measurement timing.", "",
    "Loss measures disagreement with labelled pixels of this training example.",
    "Unlabelled and conflicting pixels were ignored in the training loss.",
    "Lower loss does not establish performance on new microscopy images.",
    "This program only reads saved results. No new training or predictions.",
    "The previous image, annotations, masks and model weights remain unchanged."
  ), file.path(temporary, "READ_ME.txt"))
  if (!identical(tools::md5sum(inputs), hashes_before)) stop("An input changed while plotting.")
  if (file.exists(output_dir)) stop("Output appeared while plotting; nothing overwritten.")
  if (!file.rename(temporary, output_dir)) stop("Could not publish the graph folder.")
}, finally = if (dir.exists(temporary)) unlink(temporary, recursive = TRUE))
cat("Graph saved:\n", file.path(output_dir, "refined_training_learning_curve.png"), "\n",
    "Read the saved training history; no model was trained or changed.\n", sep = "")

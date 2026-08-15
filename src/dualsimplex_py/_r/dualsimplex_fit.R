#!/usr/bin/env Rscript
# Headless driver for the DualSimplex R package, invoked by the Python
# wrapper `dualsimplex_py` via subprocess.
#
# Usage: Rscript dualsimplex_fit.R <config.json> <out_dir>
#
#   config.json : JSON with all solver settings (see Python wrapper).
#   out_dir     : directory containing data.csv; writes W.csv, H.csv,
#                 kept_features.csv here.
#
# This reproduces the pipeline from the user's DualSimplex.R:
#   linearize -> drop zero rows/cols -> set_data -> project(K)
#   -> (optional) distance_filter + re-project -> set.seed
#   -> init_solution("random_invertible")
#   -> default_optimization() | optim_solution(n_iterations, optim_config(...))
#   -> finalize_solution("clean")
# No plotting is performed.

suppressPackageStartupMessages({
    library(DualSimplex)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript dualsimplex_fit.R <config.json> <out_dir>")
}
config_path <- args[1]
out_dir <- args[2]

config <- jsonlite::read_json(config_path, simplifyVector = TRUE)
data_path <- file.path(out_dir, "data.csv")

# ---- 1. Load data (genes x samples, non-negative) ----
data_raw <- as.matrix(read.csv(data_path, row.names = 1, check.names = FALSE))
if (isTRUE(config$linearize)) {
    data_raw <- linearize_dataset(data_raw)      # no-op for raw counts
}
data_raw <- remove_zero_rows(data_raw)
data_raw <- data_raw[, colSums(data_raw) > 0, drop = FALSE]

# ---- 2. Sinkhorn + SVD projection ----
dso <- DualSimplexSolver$new()
dso$set_data(
    data_raw,
    max_sinkhorn_iterations = config$max_sinkhorn_iterations,
    max_dim = config$max_dim,
    sinkhorn_tol = config$sinkhorn_tol,
    svd_method = config$svd_method
)
dso$project(config$k)

# ---- 3. (Optional) distance filter + re-project ----
if (!is.null(config$plane_d_lt) || !is.null(config$zero_d_lt)) {
    dso$distance_filter(
        plane_d_lt = config$plane_d_lt,
        zero_d_lt = config$zero_d_lt,
        for_features = TRUE
    )
    dso$project(config$k)
}

# ---- 4. Solve ----
set.seed(config$seed)
dso$init_solution(config$initialization)         # default "random_invertible"

if (identical(config$optimization, "custom")) {
    cfg <- optim_config(
        method = config$optim_method,
        coef_der_X = config$coef_der_X,
        coef_der_Omega = config$coef_der_Omega,
        coef_hinge_H = config$coef_hinge_H,
        coef_hinge_W = config$coef_hinge_W
    )
    dso$optim_solution(iterations = config$n_iterations, config = cfg)
} else {
    dso$default_optimization()                   # robust schedule (paper default)
}

# ---- 5. Get W and H ----
solution <- dso$finalize_solution(reverse_sinkhorn_type = config$reverse_sinkhorn_type)
W <- solution$W    # genes x K  (component/basis signatures)
H <- solution$H    # K x samples (proportions)

write.csv(W, file.path(out_dir, "W.csv"))
write.csv(H, file.path(out_dir, "H.csv"))
write.csv(data.frame(feature = rownames(W)),
          file.path(out_dir, "kept_features.csv"), row.names = FALSE)

if (!is.null(config$save_state)) {
    dso$save_state(config$save_state)
}

cat(sprintf("DUALSIMPLEX_OK k=%d genes=%d samples=%d\n",
            ncol(W), nrow(W), ncol(H)))

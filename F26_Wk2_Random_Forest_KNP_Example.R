
# KNP Fire - post-fire regeneration probability (Random Forest)

# packages  
{
  if (!require(terra)) install.packages("terra")
  if (!require(randomForest)) install.packages("randomForest")
  if (!require(ggplot2)) install.packages("ggplot2")
  library(terra)
  library(randomForest)
  library(ggplot2)
}

set.seed(123)
data_dir <- "data/raster_predictors"



# ---- 1. load rasters ----
{
  elevation <- rast(file.path(data_dir, "KNP_elevation.tif"))
  aspect    <- rast(file.path(data_dir, "KNP_aspect.tif"))
  slope     <- rast(file.path(data_dir, "KNP_slope.tif"))
  chili     <- rast(file.path(data_dir, "KNP_chili.tif"))
  mTPI      <- rast(file.path(data_dir, "KNP_mTPI.tif"))
  climate   <- rast(file.path(data_dir, "KNP_TerraClimate.tif"))
  
  names(elevation) <- "elevation"
  names(aspect)    <- "aspect"
  names(slope)     <- "slope"
  names(chili)     <- "chili"
  names(mTPI)      <- "mTPI"
  names(climate)   <- c("CWD", "AET", "PET", "SoilMoisture", "PDSI", "SWE_max")
}


# ---- 2. check grids against the DEM ----
rasters <- list(elevation = elevation, aspect = aspect, slope = slope,
                chili = chili, mTPI = mTPI, climate = climate)

check <- data.frame(
  layer     = names(rasters),
  same_crs  = sapply(rasters, function(r) crs(r) == crs(elevation)),
  res_x     = sapply(rasters, function(r) res(r)[1]),
  res_y     = sapply(rasters, function(r) res(r)[2]),
  ncell     = sapply(rasters, ncell),
  same_grid = sapply(rasters, function(r) compareGeom(r, elevation, stopOnError = FALSE))
)
check
# elevation, aspect and slope share the same grid (~30 m)
# chili (~90 m), mTPI (~270 m) and climate (~4 km) need resampling


# ---- 3. match everything to the DEM grid ----
chili_r   <- resample(chili,   elevation, method = "bilinear")
mTPI_r    <- resample(mTPI,    elevation, method = "bilinear")
climate_r <- resample(climate, elevation, method = "bilinear")

# check for NAs created at the edges (climate only has 90 cells)
global(climate_r, "isNA")
plot(climate[["CWD"]])
plot(climate_r[["CWD"]])

# aspect is circular (359 and 1 are neighbors), so split it in two
northness <- cos(aspect * pi / 180)
eastness  <- sin(aspect * pi / 180)
names(northness) <- "northness"
names(eastness)  <- "eastness"

predictors <- c(elevation, slope, northness, eastness,
                chili_r, mTPI_r, climate_r)
names(predictors)


# ---- 4. training table ----
# random pixels from the stack, keeping x/y for the spatial CV later
df_data <- spatSample(predictors, size = 5000, method = "random",
                      na.rm = TRUE, xy = TRUE, as.df = TRUE)
head(df_data)
str(df_data)

summary(df_data$PET - df_data$AET - df_data$CWD) 

# ---- 4. restrict to the sequoia elevation band ----
elev_min <- 1500   # edit with values from the literature
elev_max <- 2300
elev_center <- (elev_min + elev_max) / 2
elev_half   <- (elev_max - elev_min) / 2

# cells outside the band become NA in every predictor
band_mask       <- (elevation >= elev_min) & (elevation <= elev_max)
predictors_band <- mask(predictors, band_mask, maskvalues = FALSE)

# training table: random pixels inside the band only
df_data <- spatSample(predictors_band, size = 2500, method = "random",
                      na.rm = TRUE, xy = TRUE, as.df = TRUE)
range(df_data$elevation)   # should fall inside the band
head(df_data)


# ---- 5. simulated regeneration (replace with field data when available) ----
# elevation effect: 0 at the center of the band, -1 at its edges (hump shape)

elev_term <- -((df_data$elevation - elev_center) / elev_half)^2

z_soil <- as.numeric(scale(df_data$SoilMoisture))
z_cwd  <- as.numeric(scale(df_data$CWD))

suitability <- 1 + 1.5 * elev_term + 0.8 * z_soil - 0.5 * z_cwd

# random noise so the model can't just recover the formula perfectly
suitability <- suitability + rnorm(nrow(df_data), mean = 0, sd = 1)
prob_regen  <- 1 / (1 + exp(-suitability))

df_data$Regeneration <- factor(ifelse(runif(nrow(df_data)) < prob_regen, "Yes", "No"))
table(df_data$Regeneration)

# ---- 6. random forest ----
# x and y are not predictors, drop them for the model
train_df <- df_data[, !(names(df_data) %in% c("x", "y"))]

# find a decent mtry first
tuned <- tuneRF(train_df[, names(train_df) != "Regeneration"],
                train_df$Regeneration,
                ntreeTry = 300, stepFactor = 1.5, improve = 0.01,
                trace = FALSE, plot = FALSE)
print(tuned)

best_mtry <- tuned[which.min(tuned[, "OOBError"]), "mtry"]
best_mtry

rf_model <- randomForest(Regeneration ~ ., data = train_df,
                         ntree = 500, mtry = best_mtry, importance = TRUE)
print(rf_model)


# ---- 7. spatial cross-validation ----
# OOB error is too optimistic with spatial data, so leave out
# one strip of the map at a time
df_data$block <- as.integer(cut(df_data$x, breaks = 5))
cv_acc <- c()

for (b in 1:5) {
  train_cv <- train_df[df_data$block != b, ]
  test_cv  <- train_df[df_data$block == b, ]
  rf_cv    <- randomForest(Regeneration ~ ., data = train_cv,
                           ntree = 300, mtry = best_mtry)
  pred_cv  <- predict(rf_cv, test_cv)
  cv_acc[b] <- mean(pred_cv == test_cv$Regeneration)
}

cv_acc
mean(cv_acc)


# ---- 8. variable importance ----
imp <- as.data.frame(importance(rf_model))
imp$Variable <- rownames(imp)

p_imp <- ggplot(imp, aes(x = reorder(Variable, MeanDecreaseAccuracy),
                         y = MeanDecreaseAccuracy)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Predictor Importance",
       x = "", y = "Mean Decrease in Accuracy") +
  theme_minimal()

p_imp

ggsave("outputs/variable_importance.png", p_imp, width = 6, height = 4, dpi = 300)


# ---- 9. predict over the whole area ----
prob_map <- predict(predictors_band, rf_model, type = "prob", na.rm = TRUE)

regen_yes_prob <- prob_map[["Yes"]]
names(regen_yes_prob) <- "P_regeneration"

writeRaster(regen_yes_prob, "outputs/KNP_regen_probability.tif", overwrite = TRUE)


# ---- 10. map ----
png("outputs/regen_probability_map.png", width = 2000, height = 1800, res = 300)

plot(regen_yes_prob,
     main = "Predicted Post-Fire Regeneration Probability (KNP Fire)",
     col = rev(terrain.colors(100)),
     range = c(0, 1),
     mar = c(3, 3, 3, 5))

dev.off()


########
plot(rf_model)                                  # OOB error vs number of trees
partialPlot(rf_model, train_df, elevation, "Yes")   # should show the hump
partialPlot(rf_model, train_df, SoilMoisture, "Yes")


glm_model <- glm(Regeneration ~ ., data = train_df, family = binomial)
glm_pred  <- ifelse(predict(glm_model, type = "response") > 0.5, "Yes", "No")
mean(glm_pred == train_df$Regeneration) 

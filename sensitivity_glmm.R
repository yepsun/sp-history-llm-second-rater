# Crossed random-effects sensitivity models (Reviewer R, major comment 5).
# Fits on the long-format data derived from item_level_scores_anonymized.csv (v2):
# one row per route (LLM, examiner) per adjudicated item with expert majority consensus.
# Run:  Rscript sensitivity_glmm.R   (requires R packages: lme4, Matrix)
library(lme4)

df <- read.csv("item_level_scores_anonymized.csv", stringsAsFactors = TRUE)
llm <- "deepseek_v4_pro"; human <- "examiner_score"
E <- c("expert_wst", "expert_fj", "expert_gxx")

disc <- df[[llm]] != df[[human]]
nuniq <- apply(df[E], 1, function(x) length(unique(x[!is.na(x)])))
has_maj <- disc & nuniq <= 2
sub <- df[has_maj, ]
maj <- apply(sub[E], 1, function(x) as.numeric(names(sort(table(x), decreasing = TRUE))[1]))

mk <- function(scores, route) data.frame(
  match = as.numeric(scores == maj), route = route, abserr = abs(scores - maj),
  student = sub$student_id, examiner = sub$examiner_id,
  case = sub$sp_case, item = paste(sub$sp_case, sub$item_seq, sep = "-"))
long <- rbind(mk(sub[[llm]], 1), mk(sub[[human]], 0))

# logistic GLMM: exact agreement with consensus, crossed random intercepts
gm <- glmer(match ~ route + (1|student) + (1|examiner) + (1|case) + (1|item),
            data = long, family = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
cat("=== crossed RE logistic GLMM (match ~ route) ===\n")
print(round(fixef(gm), 4))
ci <- confint(gm, method = "Wald", quiet = TRUE)
cat(sprintf("route log-OR %.3f, OR %.2f, 95%% CI [%.2f, %.2f]\n",
            fixef(gm)["route"], exp(fixef(gm)["route"]),
            exp(ci["route", 1]), exp(ci["route", 2])))
cat("random-effect SDs:\n"); print(round(sqrt(unlist(VarCorr(gm))), 3))

# adjusted marginal agreement difference by simulation from the fitted distribution
set.seed(20251110); S <- 400000
b0 <- fixef(gm)["(Intercept)"]; b1 <- fixef(gm)["route"]
sds <- sqrt(unlist(VarCorr(gm)))
re <- matrix(rnorm(S * 4, 0, rep(sds, each = S)), ncol = 4)
eta0 <- b0 + rowSums(re); eta1 <- b0 + b1 + rowSums(re)
cat(sprintf("model-based marginal agreement: examiner %.1f%%, LLM %.1f%% (adjusted diff %.1f pp)\n",
            mean(plogis(eta0)) * 100, mean(plogis(eta1)) * 100,
            (mean(plogis(eta1)) - mean(plogis(eta0))) * 100))

# Gaussian LMM for absolute error
mm <- lmer(abserr ~ route + (1|student) + (1|examiner) + (1|case) + (1|item),
           data = long, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
cat("=== crossed RE Gaussian LMM (abserr ~ route) ===\n")
ci2 <- confint(mm, method = "Wald", quiet = TRUE)
cat(sprintf("route coefficient %.3f, 95%% CI [%.3f, %.3f]\n",
            fixef(mm)["route"], ci2["route", 1], ci2["route", 2]))

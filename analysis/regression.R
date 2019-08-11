library(brms)
require(ggplot2)
library(lme4)
library(sjPlot)
library(car)
library(dplyr)
library(standardize)
library(cowplot)
library(projpred)
library(knitr)
library(kableExtra)
library(data.table)


marginal_effects_table <- function(model, term) {
    data = marginal_effects(model, term)[[term]]
    conditions = strsplit(term, ":")[[1]]
    data[, c(conditions, c("estimate__", "lower__", "upper__"))]
}

set_theme(base=theme_sjplot(base_family = "Palatino"),
          title.size=2,
          axis.textsize = 1.2,
          axis.title.size = 2,
          legend.title.size = 2,
          legend.item.size = 1,
          legend.size = 2)

red <- "#e74c3c"
blue <- "#3498db"
green <- "#27ae60"
purple <- "#8e44ad"
gray <- "#2c3e50"
yellow <- "#f1c40f"

df <- read.csv("../data/db_data.csv")
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- factor(2 - df$true_answer)
# only include full games
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$correct <- abs(1 - as.integer(df$correct))
df$trial_id = df$level * 5 + df$iteration
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
## df$trial_id = df$trial_id / 11 # rescale for easier interpretation of coefficients
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15
df = as.data.table(df)
df[, prev_correct := shift(.(correct), 1, type="lag"), by=test_id]
df[, prev_response := shift(.(user_answer), 1, type="lag"), by=test_id]
df[df$type == "choose", "prev_response"] = df[df$type == "choose", "prev_correct"]

head(df)

##########################################################################################
## Objective authenticity
##########################################################################################

## First test for a baseline model
prop.table(table(df$correct))
m_baseline <- brm(correct ~ 1 + (1|test_id), data=df,
                  family="bernoulli",
                  prior=c(set_prior("normal(0, 5)", class="Intercept"),
                          set_prior("cauchy(0, 1)", class="sd")),
                  cores = 4)
summary(m_baseline)
m_baseline <- add_criterion(m_baseline, "waic")

# compute the odds
b <- summary(m_baseline)$fixed[1, c(1, 3, 4)]
exp(b)

coda = posterior_samples(m_baseline)
a = data.frame(correct = coda[,1])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 2)
tn

# test for performance differences between type A (choose) and type B (forreal) questions 
m_type <- brm(correct ~ type + (1|test_id), data=df,
              family="bernoulli",
              prior = c(set_prior("normal(0, 2)", class="b"),
                        set_prior("normal(0, 5)", class="Intercept"),
                        set_prior("cauchy(0, 1)", class="sd")),
              cores = 4, sample_prior = TRUE)
summary(m_type)
b <- summary(m_type)$fixed[, c(1, 3, 4)]
exp(b)
m_type <- add_criterion(m_type, "waic")

marginal_effects_table(m_type, "type")

g = plot(marginal_effects(m_type, "type"), plots=F)[[1]]
g$layers[[2]]$aes_params$colour <- "#66B8B1"
g$layers[[1]]$aes_params$colour <- "#66B8B1"
g = g + labs(y="Detection Accuracy", x="Question Type") +
    scale_x_discrete(labels=c("Type A", "Type B")) +
    scale_color_manual("type", values=c(red, blue)) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
ggsave("../images/question_type.png", g, width=6, height=6, dpi=300, bg="transparent")

## test for the effect of trial number using a monotonic effect
m_trial_mono = brm(correct ~ mo(trial_id) + (1|test_id), data=df,
                   family = "bernoulli",
                   prior = c(set_prior("normal(0, 2)", class="b"),
                             set_prior("normal(0, 5)", class="Intercept"),
                             set_prior("cauchy(0, 1)", class="sd")),
                   control = list(max_treedepth = 15),
                   cores = 4, sample_prior = TRUE)
summary(m_trial_mono)
m_trial_mono <- add_criterion(m_trial_mono, "waic")

## split effect of trial number for type A and B questions
m_trial_type_mono = brm(correct ~ mo(trial_id) * type + (1|test_id),
                        data=df, family = "bernoulli",
                        prior = c(set_prior("normal(0, 2)", class="b"),
                                  set_prior("normal(0, 5)", class="Intercept"),
                                  set_prior("cauchy(0, 1)", class="sd")),
                        control = list(max_treedepth = 15),
                        cores = 4, sample_prior = TRUE)
summary(m_trial_type_mono)
exp(summary(m_trial_type_mono)$fixed[, c(1, 3, 4)])
m_trial_type_mono <- add_criterion(m_trial_type_mono, "waic")

g_trial_type_mono = plot(marginal_effects(m_trial_type_mono, "trial_id:type"), plots=F)[[1]] + 
    labs(y="Authenticity Judgment Accuracy", x="Trial Number") + 
    scale_colour_manual("Question type", values=c(red, blue), labels=c("Type A", "Type B")) +
    scale_fill_manual("Question type", values=c(red, blue), labels=c("Type A", "Type B")) +
    ylim(c(0.4, 0.72)) +
    theme(legend.position = "top")

## Does the trial effect depend on whether real or fake was the correct answer?
m_trial_true_mono = brm(correct ~ mo(trial_id) * true_answer + (1|test_id),
                        data=df[df$type == "forreal",],
                        family='bernoulli',
                        prior = c(set_prior("normal(0, 2)", class="b"),
                                  set_prior("normal(0, 5)", class="Intercept"),
                                  set_prior("cauchy(0, 1)", class="sd")),
                        control=list(max_treedepth=15),
                        cores = 4, sample_prior = TRUE)
summary(m_trial_true_mono)
exp(summary(m_trial_true_mono)$fixed[, c(1, 3, 4)])

g_trial_true_mono = plot(marginal_effects(m_trial_true_mono, "trial_id:true_answer"), plots=F)[[1]] + 
    labs(y="Authenticity Judgment Accuracy", x="Trial Number") + 
    scale_colour_manual("True answer", values=c(purple, green),
                        labels=c("Generated", "Authentic")) +
    scale_fill_manual("True answer", values=c(purple, green),
                      labels=c("Generated", "Authentic")) +
    ylim(c(0.4, 0.72)) +
    theme(legend.position = "top")

plots = cowplot::plot_grid(g_trial_type_mono, g_trial_true_mono, nrow=1,
                           align="h", labels = c("a)", "b)"), label_size=30)
cowplot::save_plot("../images/Fig2.pdf", plots, dpi=300,
                   base_width=16, base_height=8)

# test for differences between language models. We only focus on type-B questions (i.e. forreal)
df = df[df$type == "forreal",]

m_genlevel <- brm(correct ~ genlevel + (1|test_id), data=df[df$true_answer == 0,],
                  family='bernoulli',
                  prior = c(set_prior("normal(0, 2)", class="b"),
                            set_prior("normal(0, 5)", class="Intercept"),
                            set_prior("cauchy(0, 1)", class="sd")),
                  control = list(max_treedepth=20),
                  cores = 4, sample_prior = TRUE)
summary(m_genlevel)
plot(m_genlevel)
m_genlevel <- add_criterion(m_genlevel, "waic")

p_genlevel = marginal_effects(m_genlevel, "genlevel")
marginal_effects_table(m_genlevel, "genlevel")

hypothesis(m_genlevel, "genlevelsyl < 0")
hypothesis(m_genlevel, "genlevelhybrid < 0")

g = plot(marginal_effects(m_genlevel, "genlevel"), plots=F)[[1]]
g$layers[[2]]$aes_params$colour <- red
g$layers[[1]]$aes_params$colour <- blue
g = g + labs(y="Authenticity Judgment Accuracy", x="Generation Model") +
    ylim(c(0.45, 0.7)) +
    scale_x_discrete(labels=c("Character", "Hierarchical", "Syllable")) +
    theme_plos()

ggsave("../images/genlevel.pdf", g, dpi=300, bg="transparent")

# next we can do the same with conditional
m_condition <- brm(correct ~ conditional + (1|test_id), data=df[df$true_answer == 0,],
                   family='bernoulli',
                   prior = c(set_prior("normal(0, 2)", class="b"),
                             set_prior("normal(0, 5)", class="Intercept"),
                             set_prior("cauchy(0, 1)", class="sd")),
                   control = list(max_treedepth=20),
                   cores = 4, iter = 2000, sample_prior = TRUE)
summary(m_condition)
plot(m_condition)
m_condition <- add_criterion(m_condition, "waic")

p_condition = marginal_effects(m_condition, "conditional")
marginal_effects_table(m_condition, "conditional")

g = plot(marginal_effects(m_condition, "conditional"), plots=F)[[1]]
g$layers[[2]]$aes_params$colour <- "#66B8B1"
g$layers[[1]]$aes_params$colour <- "#66B8B1"
g = g + labs(y="Authenticity Judgment Accuracy", x="Conditioning") + ylim(c(0.45, 0.7)) +
    scale_x_discrete(labels=c("Unconditioned", "Conditioned")) +
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

ggsave("../images/conditioning.png", g, dpi=300, bg="transparent")

# Next, investigate interactions between genlevel and conditional
m_genlevel_condition = brm(correct ~ genlevel * conditional + (1|test_id),
                           data=df[df$true_answer == 0,],
                           family = 'bernoulli',
                           prior = c(set_prior("normal(0, 2)", class="b"),
                                     set_prior("normal(0, 5)", class="Intercept"),
                                     set_prior("cauchy(0, 1)", class="sd")),
                           control=list(max_treedepth=20),
                           cores = 4, iter = 4000, sample_prior = TRUE)
summary(m_genlevel_condition)
m_genlevel_condition <- add_criterion(m_genlevel_condition, "waic")
plot(m_genlevel_condition)

p_genlevel_condition = marginal_effects(m_genlevel_condition, "genlevel:conditional")
marginal_effects_table(m_genlevel_condition, "genlevel:conditional")

g = plot(marginal_effects(m_genlevel_condition, "genlevel:conditional"), plots=F)[[1]] +
    scale_colour_manual("Conditioning", values=c(red, blue), labels=c("no", "yes")) +
    scale_fill_manual("Conditioning", values=c(red, blue), labels=c("no", "yes")) +
    ylim(c(0.4, 0.7)) + 
    labs(y="Authenticity Judgment Accuracy", x="Generation Model") +
    scale_x_discrete(labels=c("Character", "Hierarchical", "Syllable")) +
    theme(legend.position = "top", legend.key = element_rect(fill = NA))

ggsave("../images/Fig4.pdf", g, width = 10, height = 6, dpi=300)

w <- loo_compare(m_genlevel, m_condition, m_genlevel_condition, criterion = "waic")
print(w, simplify=FALSE)
cbind(waic_diff = w[, 1] * -2, se        = w[, 2] * 2)
model_weights(m_genlevel, m_condition, m_genlevel_condition, weights="waic")

##########################################################################################
## Perceived authenticity
##########################################################################################

# We should only consider bias in forreal (type B) questions, as it doesn't make much 
# make much sense for choose (type A) questions.
# The perceived authenticity is equal to the user answer in forreal questions
df$perceived = df$user_answer

# there is not a clear bias towards real or generated:
prop.table(table(df$perceived))

## Is there a bias towards fake or real
p_baseline <- brm(perceived ~ 1 + (1|test_id), data=df,
                  family="bernoulli",
                  prior = c(set_prior("normal(0, 5)", class="Intercept"),
                            set_prior("cauchy(0, 1)", class="sd")),
                  control = list(adapt_delta=0.99),
                  cores = 4, sample_prior = TRUE)
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
round(exp(b), 1)
p_baseline <- add_criterion(p_baseline, "waic")

coda = posterior_samples(p_baseline)
a = data.frame(original = coda[,1])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 2)
tn

## Analyze the effect of trials using a monotonic effect
m_trial_bias_mono = brm(perceived ~ mo(trial_id) + (1|test_id),
                        data=df, family = "bernoulli",
                        prior = c(set_prior("normal(0, 2)", class="b"),
                                  set_prior("normal(0, 5)", class="Intercept"),
                                  set_prior("cauchy(0, 1)", class="sd")),
                        control = list(adapt_delta=0.99),
                        cores = 4, sample_prior = TRUE)
summary(m_trial_bias_mono)
m_trial_bias_mono <- add_criterion(m_trial_bias_mono, "waic")

g = plot(marginal_effects(m_trial_bias_mono, "trial_id"), plots=F)[[1]]
g$layers[[1]]$aes_params$colour <- "#00aedb"
g = g + labs(y="Probability authentic perception", x="Trial number") +
    theme_plos()
ggsave("../images/Fig3.pdf", g, dpi=300, width=10, height=8)


##########################################################################################
## Linguistic Feature analysis
##########################################################################################
# Add abs(difference between features in choose case)
# Find out if there's some transfer learning going on between game types.

df <- read.csv("../data/db_features-new.csv")
df <- df[df$user_answer != 0,] # remove unanswered
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
df$perceived <- df$user_answer
df$trial_id = df$level * 5 + df$iteration
## df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$perceived = factor(df$perceived, levels=c(0, 1), labels=c("generated", "authentic"))
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15

df$expert = 0
score_dev = floor(mean(df$score) + sd(df$score))
df[df$score >= score_dev, "expert"] = 1
df$expert <- as.factor(df$expert)
df <- df[df$type == "forreal",]
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$test_id <- as.integer(as.factor(df$test_id))

predictors = c("alliteration",                   # sounds
               "assonance",                      # sounds
               "rhyme_density",                  # sounds
               "mean_span",                      # sentence complexity ~~
               "mean_depth",                     # sentence complexity --|
               "pc.words",                       # contents
               "repeated.words",                 # repetition
               "word.repetitiveness",            # repetition --|
               "word.length.syllables"           # word complexity --|
)

summarizer <- function(col) {
    paste0(
        "$\\mu=",
        format(mean(col), digits=2), 
        "$ ($\\sigma=", 
        format(sd(col), digits=2),
        "$)")
}

df[, c("source", "modeltype", predictors)] %>% group_by(source, modeltype) %>% 
    summarise_all(funs(summarizer)
    ) %>%
    select(c("source", "modeltype", predictors)) %>% t() %>%
    kable("latex", align="r", escape = F, booktabs = T, linesep = "")

formula <- as.formula(paste("source ~ (", paste(predictors, collapse = "+"), ")"))
regr = standardize(formula, data=df, family = "binomial", scale=0.5)
m_objective = brm(regr$formula, data=regr$data,
                  family = "bernoulli",
                  prior = c(set_prior("normal(0, 2)", class="b"),
                            set_prior("normal(0, 5)", class="Intercept")),
                  cores = 4, sample_prior = TRUE)
summary(m_objective)

b <- summary(m_objective)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

pred.order <- c(9, 3, 1, 2, 8, 6, 7, 4, 5)
pred.names <- c(
    "Mean Depth",
    "Mean Span",
    "Word Repetition",
    "Non-PC Vocabulary",
    "Lexical Diversity",
    "Assonance",
    "Alliteration",
    "Rhyme Density",
    "Word Length"
)

formula = as.formula(
    paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df[df$trial_id <= 11,], family = "binomial", scale = 0.5)
m_subjective = brm(regr$formula, data=regr$data, family = "bernoulli",
                   control=list(adapt_delta=0.95),
                   prior = c(set_prior("normal(0, 2)", class="b"),
                             set_prior("normal(0, 5)", class="Intercept"),
                             set_prior("cauchy(0, 1)", class="sd")),
                   cores = 4, sample_prior = TRUE)
summary(m_subjective)

b <- summary(m_subjective)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

## Top performing players
formula = as.formula(
    paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df[df$expert == 1,], family = "binomial", scale = 0.5)
m_top_players = brm(regr$formula, data=regr$data,
                    family = "bernoulli",
                    prior = c(set_prior("normal(0, 2)", class="b"),
                              set_prior("normal(0, 5)", class="Intercept"),
                              set_prior("cauchy(0, 1)", class="sd")),
                    cores = 4, sample_prior = TRUE)
summary(m_top_players)


set_theme(base=theme_sjplot(base_family = "Palatino"),
          title.size=1.9,
          axis.textsize = 1.3,
          axis.title.size = 1.9,
          legend.title.size = 1,
          legend.item.size = 1,
          legend.size = 1)

p_objective <- plot_model(m_objective, show.values = TRUE,
           title = "Objective Feature Importance", bpe = "mean",
           colors = c(red, blue),
           prob.inner = .5,
           value.size=7,
           label.size=7,
           order.terms = pred.order,
           axis.labels = pred.names,
           prob.outer = .95,
           transform = NULL,
           ) + theme(text=element_text(size=11))

p_subjective <- plot_model(m_subjective, show.values = TRUE,
                           title = "Subjective Feature Importance", bpe = "mean",
                           colors = c(red, blue),
                           prob.inner = .5,
                           prob.outer = .95,
                           value.size=7,
                           label.size=7,
                           order.terms = pred.order,
                           axis.labels = pred.names,
                           transform=NULL,
                           ) + theme(text=element_text(size=11))

plots = cowplot::plot_grid(p_objective, p_subjective, nrow=1, labels="auto",
                           label_size=20, align="h")
cowplot::save_plot("../images/Fig5.pdf", plots, dpi=300,
                   base_width=15, base_height=8)

p_top_players = plot_model(m_top_players, show.values = TRUE,
                           title = "High-scoring participants", bpe = "mean",
                           colors = c(red, blue),
                           prob.inner = .5,
                           prob.outer = .95,
                           value.size=7,
                           label.size=7,
                           order.terms = pred.order,
                           axis.labels = pred.names,
                           transform = NULL,
                           ) + theme(text=element_text(size=11))

plots = cowplot::plot_grid(p_top_players, nrow=1, align="h")
cowplot::save_plot("../images/Fig6.pdf", plots, dpi=300,
                   base_width=10.5, base_height=8)

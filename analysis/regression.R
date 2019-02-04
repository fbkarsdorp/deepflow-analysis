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


marginal_effects_table <- function(model, term) {
    data = marginal_effects(model, term)[[term]]
    conditions = strsplit(term, ":")[[1]]
    data[, c(conditions, c("estimate__", "lower__", "upper__"))]
}

set_theme(base=theme_sjplot(),
          title.size=2,
          axis.textsize = 1.3,
          axis.title.size = 1.8,
          legend.title.size = 1.5,
          legend.item.size = 1,
          legend.size = 1.5)

df <- read.csv("../data/db_data.csv")
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
# only include full games
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$correct <- abs(1 - as.integer(df$correct))
df$trial_id = df$level * 5 + df$iteration
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$trial_id = df$trial_id / 11 # rescale for easier interpretation of coefficients
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15

df <- df[df$type == "forreal",] # remove choose cases for now

# TODO: clean users with weird timings
# TODO: check for really fast answers
# TODO: separate analysis for experts

head(df)

##########################################################################################
## Objective authenticity
##########################################################################################

## First test for a baseline model
prop.table(table(df$correct))
m_baseline <- brm(correct ~ 1 + (1|test_id), data=df, family="bernoulli")
summary(m_baseline)
plot(mod)
# compute the odds
b <- summary(m_baseline)$fixed[1, c(1, 3, 4)]
exp(b)

coda = posterior_samples(m_baseline)
a = data.frame(correct = coda[,1])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 2)
tn

regr = standardize(correct ~ trial_id + (1|test_id), data=df, family="binomial", scale=1.0)
m_trial = brm(regr$formula, regr$data, family='bernoulli')
summary(m_trial)
exp(summary(m_trial)$fixed[2, c(1, 3, 4)])

## Does the trial effect depend on whether real or fake was the correct answer?
regr = standardize(correct ~ trial_id * true_answer + (1|test_id), data=df, family="binomial")
m_trial_true = brm(regr$formula, data=regr$data, family = "bernoulli",
                   control=list(max_treedepth=20))
summary(m_trial_true)

g = plot(marginal_effects(m_trial_true, "trial_id:true_answer"), plots=F)[[1]] +
    labs(y="Accuracy", x="Trial number (scaled)") +
    scale_colour_brewer("True answer", palette="Set1", labels=c("Authentic", "Generated")) +
    scale_fill_brewer("True answer", palette="Set1", labels=c("Authentic", "Generated")) +
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"))

ggsave("../images/trial_effect.pdf", g, width=10, height=6, dpi=300, bg="transparent")


# is the time participants took advantageous?
regr = standardize(correct ~ time + (1|test_id), data=df, family = 'binomial')
mod <- glmer(regr$formula, data=regr$data, family="binomial")
summary(mod)
Anova(mod) # no, it appears that participants perform worse when they
           # take more time. How to explain this? Is that because the 
           # examples are harder? Or is that irrespective of the 
           # difficulty of the examples?

# test for differences between language models. First for forreal questions
m_genlevel <- brm(correct ~ genlevel + (1|test_id), data=df[df$true_answer == 0,],
                  family='bernoulli')
summary(m_genlevel)
plot(m_genlevel)

p_genlevel = marginal_effects(m_genlevel, "genlevel")
marginal_effects_table(m_genlevel, "genlevel")

hypothesis(m_genlevel, "genlevelsyl < 0")
hypothesis(m_genlevel, "genlevelhybrid < 0")

g = plot(marginal_effects(m_genlevel, "genlevel"), plots=F)[[1]] +
    labs(y="Accuracy", x="Generation Model") + ylim(c(0.45, 0.7)) + 
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"))

ggsave("../images/genlevel.pdf", g, dpi=300, bg="transparent")

# next we can do the same with conditional (we will also combine them in a single model)
m_condition <- brm(correct ~ conditional + (1|test_id), data=df[df$true_answer == 0,],
                   family='bernoulli')
summary(m_condition)
plot(m_condition)

p_condition = marginal_effects(m_condition, "conditional")
marginal_effects_table(m_condition, "conditional")

g = plot(marginal_effects(m_condition, "conditional"), plots=F)[[1]] +
    labs(y="Accuracy", x="Conditioning") + ylim(c(0.45, 0.7)) + 
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"))

ggsave("../images/conditioning.pdf", g, dpi=300, bg="transparent")

# Next, investigate interactions between genlevel and conditional
# First, forreal:
regr = standardize(correct ~ genlevel * conditional + (1|test_id),
                   data=df[df$true_answer == 0,], family = 'binomial', scale = 1)
m_genlevel_condition = brm(regr$formula, data=regr$data, family = 'bernoulli',
                           control=list(max_treedepth=20))
summary(m_genlevel_condition)
plot(m_genlevel_condition)

p_genlevel_condition = marginal_effects(m_genlevel_condition, "genlevel:conditional")
marginal_effects_table(m_genlevel_condition, "genlevel:conditional")

g = plot(marginal_effects(m_genlevel_condition, "genlevel:conditional"), plots=F)[[1]] +
    scale_colour_brewer("Conditioning", palette="Set1", labels=c("yes", "no")) +
    scale_fill_brewer("Conditioning", palette="Set1", labels=c("yes", "no")) +
    labs(y="Accuracy", x="Generation Model") +
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"))

ggsave("../images/genlevel-conditioning.pdf", g, width = 10, height = 6,
       dpi=300, bg="transparent")


##########################################################################################
## Perceived authenticity
##########################################################################################

# The perceived authenticity is equal to the user answer in forreal questions
df$perceived = df$user_answer
# we should only consider bias in forreal questions, as it doesn't make much 
# make much sense for choose questions.

# there is not a clear bias towards real or generated:
prop.table(table(df$perceived))

## Is there a bias towards fake or real
p_baseline <- brm(perceived ~ 1 + (1|test_id), data=df,
                  control=list(adapt_delta=0.95), family="bernoulli")
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
round(exp(b), 1)

coda = posterior_samples(p_baseline)
a = data.frame(original = coda[,1])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 2)
tn

# as the game progresses, does the bias change?
regr = standardize(perceived ~ trial_id + (1|test_id), data=df, family = 'binomial')
m_trial_bias = brm(regr$formula, data=regr$data, family="bernoulli",
                   control=list(adapt_delta=0.95))
summary(m_trial_bias)

b <- summary(m_trial_bias)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

g = plot(marginal_effects(m_trial_bias, "trial_id"), plots=F)[[1]] +
    labs(y="Probability authentic perception", x="Trial number (scaled)") + 
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.background = element_rect(fill = "transparent"))

ggsave("../images/trial_bias.pdf", g, width=10, height=6, dpi=300, bg="transparent")

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
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$perceived = factor(df$perceived, levels=c(0, 1), labels=c("generated", "authentic"))
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15
df <- df[df$type == "forreal",]

## df$correct <- abs(1 - as.integer(df$correct))
## scores = df %>% group_by(test_id) %>% summarise(score = sum(correct))
## experts = pull(scores[scores$score >= 15,], "test_id")
## df$expert = 0
## df[df$test_id %in% experts, "expert"] = 1


predictors = c(
               "alliteration",                   # sounds
               "assonance",                      # sounds
               "rhyme_density",                  # sounds
               ## "stressed.vowel.repetitiveness",  # sounds --|
               ## "word.onset.repetitiveness",      # sounds --|
               ## "max_span",                       # sentence complexity ~~
               "mean_span",                      # sentence complexity ~~
               ## "max_depth",                      # sentence complexity --|
               "mean_depth",                     # sentence complexity --|
               ## "num_spans",                      # sentence complexity --|
               ## "nchars",                         # length --|
               ## "nlines",                         # length --|
               ## "nwords",                         # length --|
               "pc.words",                       # contents
               ## "pronouns",                       # contents
               ## "lzw",                            # repetition
               "repeated.words",                 # repetition
               ## "syllable.repetitiveness",        # repetition --|
               "word.repetitiveness",             # repetition --|
               ## "unigram.ppl",                    # repetition
               ## "word.length",                    # word complexity --|
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
m_objective = brm(regr$formula, data=regr$data, family = "bernoulli")
summary(m_objective)

b <- summary(m_objective)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

p_objective <- plot_model(m_objective, show.values = TRUE,
           title = "Objective feature importance", bpe = "mean",
           prob.inner = .5,
           ## value.size=8,
           ## label.size=8,
           prob.outer = .95,
           transform = NULL
           )

formula = as.formula(paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df, family = "binomial", scale = 0.5)
m_subjective = brm(regr$formula, data=regr$data, family = "bernoulli",
                   control=list(adapt_delta=0.95))

summary(m_subjective)

b <- summary(m_subjective)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

p_subjective <- plot_model(m_subjective, show.values = TRUE,
           title = "Subjective feature importance", bpe = "mean",
           prob.inner = .5,
           prob.outer = .95,
           ## value.size=8,
           transform = NULL,
           geom.label.size = 0
           ) + ylim(c(-0.75, 0.75)) + theme(axis.text.y.left = element_text(size=0))

plots = cowplot::plot_grid(p_objective, NULL, p_subjective, nrow=1,
                           align="h", rel_widths = c(1.45, 0.1, 1))
cowplot::save_plot("../images/feature-importance.pdf", plots, dpi=300, base_width=15,
                   base_height=8, bg = "transparent")


library(brms)
require(ggplot2)
library(lme4)
library(sjPlot)
library(car)
library(dplyr)
library(standardize)
library(cowplot)
library(projpred)

df <- read.csv("../data/db_data.csv")
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$correct <- abs(1 - as.integer(df$correct))
df$trial_id = df$level * 5 + df$iteration
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$trial_id = df$trial_id / 11 # rescale for easier interpretation of coefficients
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15


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

m_type = brm(correct ~ type + (1|test_id), data=df,
             family='bernoulli')

summary(m_type)

coda = posterior_samples(m_type)
a = data.frame(choose = coda[,1], forreal = coda[,1] + coda[,2])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

regr = standardize(correct ~ trial_id + (1|test_id), data=df, family="binomial", scale=1.0)
m_trial = brm(regr$formula, regr$data, family='bernoulli')
summary(m_trial)
exp(summary(m_trial)$fixed[2, c(1, 3, 4)])

## Does the trial effect depend on the type of game?
regr = standardize(correct ~ trial_id * type + (1|test_id), data=df, family="binomial", scale=1.0)
m_type_trial = brm(regr$formula, regr$data, family = 'bernoulli')
summary(m_type_trial)

## Does the trial effect depend on whether real or fake was the correct answer?
regr = standardize(correct ~ trial_id * true_answer + (1|test_id), data=df[df$type=="forreal",], family="binomial")
m_trial_true = brm(regr$formula, data=regr$data, family = "bernoulli")
summary(m_trial_true)

# is the time participants took advantageous?
regr = standardize(correct ~ time * type + (1|test_id), data=df, family = 'binomial')
mod <- glmer(regr$formula, data=regr$data, family="binomial")
summary(mod)
Anova(mod) # no, it appears that participants perform worse when they
           # take more time. How to explain this? Is that because the 
           # examples are harder? Or is that irrespective of the 
           # difficulty of the examples?
plot_model(mod, type="pred", terms=c("time", "type"))

## test for generation level effects (since we noticed a training effect
## incorporate this into the model as an interaction effect)
df_gen <- df[((df$true_answer == 0) & (df$type == "forreal")) | df$type == "choose",]

# test for differences between language models:
regr = standardize(correct ~ genlevel + (1|test_id), data=df_gen, family = 'binomial', scale=1)
m_correct <- brm(correct ~ genlevel + (1|test_id), data=df_gen, family='bernoulli')
summary(m_correct)
plot(m_correct)

coda = posterior_samples(m_correct)
a = data.frame(char = coda[,1], hybrid=coda[,1] + coda[,2], syl=coda[,1] + coda[,3])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)

# include differences between game types
regr = standardize(correct ~ genlevel * type + (1|test_id), data=df_gen, family = 'binomial', scale=1)
m_correct_t = brm(regr$formula, data=regr$data, family = "bernoulli")
summary(m_correct_t)

coda = posterior_samples(m_correct_t)
head(colnames(coda), 10)

a = data.frame(syl_forreal = coda[,1] - coda[,4],
               syl_choose = coda[,1] + coda[,4],
               char_forreal = coda[,1] + coda[,2] - coda[,4],
               char_choose = coda[,1] + coda[,2] + coda[,4] + coda[,5],
               hybrid_forreal = coda[,1] + coda[,3] - coda[,4],
               hybrid_choose = coda[,1] + coda[,3] + coda[,4] + coda[,6])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

p_genlevel = marginal_effects(m_correct_t, "genlevel:type")

# next we can do the same with conditional (we might also combine them in a single model)

regr = standardize(correct ~ conditional + (1|test_id), data=df_gen, family = 'binomial', scale=1)
m_correct_c = brm(regr$formula, data=regr$data, family = 'bernoulli')
summary(m_correct_c)

exp(summary(m_correct_c)$fixed[2, c(1, 3, 4)])

regr = standardize(correct ~ conditional * type + (1|test_id), data=df_gen, family = 'binomial', scale=1)
m_correct_cond = brm(regr$formula, data=regr$data, family = 'bernoulli')
summary(m_correct_cond)

coda = posterior_samples(m_correct_cond)
head(colnames(coda), 10)

a = data.frame(uc_fr = coda[,1],
               uc_ch = coda[,1] + coda[,3] - coda[,2],
               co_fr = coda[,1] - coda[,3] + coda[,2],
               co_ch = coda[,1] + coda[,2] + coda[,3] + coda[,4])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

p_conditional = marginal_effects(m_correct_cond, "conditional:type")

plots = cowplot::plot_grid(plot(p_genlevel)[[1]], plot(p_conditional)[[1]], labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/model-interactions.pdf", plots, dpi=300, base_width=10,
                   base_height=5, base_aspect_ratio = 1.3)

##########################################################################################
## Perceived authenticity
##########################################################################################

# The perceived authenticity is equal to the user answer in forreal questions
df$perceived = df$user_answer
# The perceived authenticity in choose questions is equal to whether the answer was correct or not.
df[df$type == "choose", "perceived"] = 1 - df[df$type == "choose", "correct"]

# there is a slight bias towards real:
prop.table(table(df$perceived))

## but this appears only to be the case in choose questions.
aggregate(perceived ~ type, df, mean)

## Is there a bias towards fake or real
p_baseline <- brm(perceived ~ 1 + (1|test_id), data=df,
                  control=list(adapt_delta=0.95),
                  family="bernoulli")
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
round(exp(b), 1)

coda = posterior_samples(p_baseline)
a = data.frame(bias = coda[,1])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

# measure bias for different game types
p_basetype <- brm(perceived ~ type + (1|test_id), data=df,
                  control=list(adapt_delta=0.95),
                  family="bernoulli")
summary(p_basetype)
plot(p_basetype)

coda = posterior_samples(p_basetype)
head(colnames(coda))

a = data.frame(bias_choose = coda[,1],
               bias_forreal = coda[,1] + coda[,2])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

# as the game progresses, does the bias change?
regr = standardize(perceived ~ trial_id + (1|test_id), data=df, family = 'binomial')
mod = glmer(regr$formula, data=regr$data, family="binomial")
summary(mod)
Anova(mod) # it seems so, yes
plot_model(mod, type="pred", terms=c("trial_id"))

# We previously saw an interesting effect of time per question on the objective
# authenticity. Does that also affect the bias towards fake or real?
regr = standardize(perceived ~ time * true_answer + (1|test_id),
                   data=df[df$type=="forreal",], family = 'binomial')
mod = glmer(regr$formula, data=regr$data, family="binomial")
summary(mod)
Anova(mod) # yes, it does, though especially for real examples.
           # QUESTION: what examples are we dealing with here?
plot_model(mod, type="pred", terms=c("time", "true_answer"))


# include differences between game types
df_gen <- df[((df$true_answer == 0) & (df$type == "forreal")) | df$type == "choose",]
regr = standardize(perceived ~ genlevel * type + (1|test_id), data=df_gen, family = 'binomial', scale=1)
p_correct_t = brm(regr$formula, data=regr$data,
                  control=list(adapt_delta=0.95),
                  family = "bernoulli")
summary(p_correct_t)

coda = posterior_samples(p_correct_t)
head(colnames(coda), 10)

a = data.frame(syl_forreal = coda[,1] - coda[,4],
               syl_choose = coda[,1] + coda[,4],
               char_forreal = coda[,1] + coda[,2] - coda[,4] - coda[,5],
               char_choose = coda[,1] + coda[,2] + coda[,4] + coda[,5],
               hybrid_forreal = coda[,1] + coda[,3] - coda[,4] - coda[,6],
               hybrid_choose = coda[,1] + coda[,3] + coda[,4] + coda[,6])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

p_p_gen = marginal_effects(p_correct_t, "genlevel:type")

regr = standardize(perceived ~ conditional * type + (1|test_id), data=df_gen, family = 'binomial', scale=1)
p_correct_cond = brm(regr$formula, data=regr$data,
                     control=list(adapt_delta=0.95),
                     family = 'bernoulli')
summary(p_correct_cond)

coda = posterior_samples(p_correct_cond)
head(colnames(coda), 10)

a = data.frame(uc_fr = coda[,1],
               uc_ch = coda[,1] + coda[,3] - coda[,2],
               co_fr = coda[,1] - coda[,3] + coda[,2] - coda[,4],
               co_ch = coda[,1] + coda[,2] + coda[,3] + coda[,4])
a = apply(a, 2, function(x) 1 / (1 + exp(-x)))
tn = t(as.matrix(apply(a, 2, function(x) quantile(x, probs=c(.5, .025, .975)))))
tn = round(tn * 100, 1)
tn

p_p_cond = marginal_effects(p_correct_cond, "conditional:type")

plots = cowplot::plot_grid(plot(p_p_gen)[[1]], plot(p_p_cond)[[1]], labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/perceived-model-interactions.pdf", plots, dpi=300, base_width=10,
                   base_height=5, base_aspect_ratio = 1.3)


##########################################################################################
## Linguistic Feature analysis
##########################################################################################
# Add abs(difference between features in choose case)
# Find out if there's some transfer learning going on between game types.

df <- read.csv("../data/db_features.csv")
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
df$perceived <- df$user_answer
## df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df[df$type == "choose", "perceived"] = 2 - as.integer(
    factor(df[df$type == "choose", "correct"]))
df$trial_id = df$level * 5 + df$iteration
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15
df <- df[df$type == "forreal" | (df$type == "choose" & df$source == "fake"),]
df$correct <- abs(1 - as.integer(df$correct))
scores = df %>% group_by(test_id) %>% summarise(score = sum(correct))
experts = pull(scores[scores$score >= 15,], "test_id")
df$expert = 0
df[df$test_id %in% experts, "expert"] = 1


predictors = c("word.entropy",                  # compression
               "mean_depth",                    # syntax
               "mean_span",                     # syntax
               "nlines",                        # length
               "pc.words",                      # content
               "pronouns",                      # content
               "rhyme_density",                 # rhyme
               "stressed.vowel.repetitiveness", # sounds
               "word.length.syllables"          # complexity
               )


formula <- as.formula(paste("source ~ (", paste(predictors, collapse = "+"), ")"))
regr = standardize(formula, data=df, family = "binomial")
m_objective = brm(regr$formula, data=regr$data, family = "bernoulli")

set_theme(base=theme_sjplot(), geom.label.size=5, axis.textsize = 1.1)

summary(m_objective)
p_objective <- plot_model(m_objective, show.values = TRUE,
           title = "Objective feature importance", bpe = "mean",
           prob.inner = .5,
           value.size=5,
           prob.outer = .95
           ) + ylim(c(0.5, 4.5))

formula = as.formula(paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df, family = "binomial")
m_subjective = brm(regr$formula, data=regr$data, family = "bernoulli")

summary(m_subjective)

p_subjective <- plot_model(m_subjective, show.values = TRUE,
           title = "Subjective feature importance", bpe = "mean",
           prob.inner = .5,
           prob.outer = .95,
           value.size=5,
           ) + ylim(c(0.7, 1.5))

plots = cowplot::plot_grid(p_objective, p_subjective, labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/feature-importance.pdf", plots, dpi=300, base_width=15,
                   base_height=8)


formula = as.formula(paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df[df$expert == 1, ], family = "binomial")
m_expert = brm(regr$formula, data=regr$data,
               control=list(adapt_delta=0.95),
               family = "bernoulli")

summary(m_expert)

p_expert <- plot_model(m_expert, show.values = TRUE,
           title = "Expert feature importance", bpe = "mean",
           prob.inner = .5,
           prob.outer = .95,
           value.size=5)
p_expert

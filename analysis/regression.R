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
regr = standardize(correct ~ trial_id * true_answer + (1|test_id), data=df, family="binomial")
m_trial_true = brm(regr$formula, data=regr$data, family = "bernoulli",
                   control=list(max_treedepth=20))
summary(m_trial_true)

palette <- "Dark2"
g <- plot(marginal_effects(m_trial_true, "trial_id:true_answer"), plots=F)[[1]] +
    scale_colour_brewer("True answer", palette=palette, labels=c("Authentic", "Generated")) +
    scale_fill_brewer("True answer", palette=palette, labels=c("Authentic", "Generated")) +
    theme(legend.box.background = element_rect(fill = "transparent"),
          legend.position="bottom",
          legend.background = element_rect(fill = "transparent"), text=element_text(size=14)
    ) + labs(x="Trial id", y="")


ggsave("../images/trial_effect.png", g, dpi=300, bg="transparent")


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

# test for differences between language models. First for forreal questions
m_correct_forreal <- brm(correct ~ genlevel + (1|test_id), data=df, family='bernoulli')
summary(m_correct_forreal)
plot(m_correct_forreal)

p_genlevel_forreal = marginal_effects(m_correct_forreal, "genlevel")
marginal_effects_table(m_correct_forreal, "genlevel")

# Next, test choose questions
m_correct_choose <- brm(correct ~ genlevel + (1|test_id),
                         data=df_gen[df_gen$type == "choose",],
                        family='bernoulli')

summary(m_correct_choose)
plot(m_correct_choose)

p_genlevel_choose = marginal_effects(m_correct_choose, "genlevel")
marginal_effects_table(m_correct_choose, "genlevel")

plots = cowplot::plot_grid(plot(p_genlevel_forreal)[[1]] + ylim(c(0.45, 0.7)),
                           plot(p_genlevel_choose)[[1]] + ylim(c(0.45, 0.7)),
                           labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/genlevel-marginals.pdf", plots, dpi=300, base_width=10,
                   base_height=5, base_aspect_ratio = 1.3)

## # include differences between game types
regr = standardize(correct ~ genlevel * type + (1|test_id), data=df_gen,
                   family = 'binomial', scale=1)
m_correct_t = brm(regr$formula, data=regr$data, family = "bernoulli")
summary(m_correct_t)

marginal_effects_table(m_correct_t, "genlevel:type")

# next we can do the same with conditional (we might also combine them in a single model)

# forreal
m_correct_c_forreal = brm(correct ~ conditional + (1|test_id),
                  data=df_gen[df_gen$type == "forreal",], family = 'bernoulli')
summary(m_correct_c_forreal)

p_conditional_forreal = marginal_effects(m_correct_c_forreal, "conditional")
marginal_effects_table(m_correct_c_forreal, "conditional")

# choose
m_correct_c_choose = brm(correct ~ conditional + (1|test_id),
                  data=df_gen[df_gen$type == "choose",], family = 'bernoulli')
summary(m_correct_c_choose)

p_conditional_choose = marginal_effects(m_correct_c_choose, "conditional")
marginal_effects_table(m_correct_c_choose, "conditional")

# plot
plots = cowplot::plot_grid(plot(p_conditional_forreal)[[1]] + ylim(c(0.48, 0.68)),
                           plot(p_conditional_choose)[[1]] + ylim(c(0.48, 0.68)),
                           labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/conditioned-marginals.pdf", plots, dpi=300, base_width=10,
                   base_height=5, base_aspect_ratio = 1.3)

# conditional with type as interaction 
regr = standardize(correct ~ conditional * type + (1|test_id), data=df_gen, family = 'binomial', scale=1)
m_correct_cond = brm(regr$formula, data=regr$data, family = 'bernoulli')
summary(m_correct_cond)


p_conditional = marginal_effects(m_correct_cond, "conditional:type")
marginal_effects_table(m_correct_cond, "conditional:type")

plots = cowplot::plot_grid(plot(p_genlevel)[[1]], plot(p_conditional)[[1]], labels = c("a)", "b)"), align="h")
cowplot::save_plot("../images/model-interactions.pdf", plots, dpi=300, base_width=10,
                   base_height=5, base_aspect_ratio = 1.3)

# Next, investigate interactions between genlevel and conditional
# First, forreal:
regr = standardize(correct ~ genlevel * conditional + (1|test_id),
                   data=df_gen[df_gen$type == "forreal",], family = 'binomial', scale = 1)
m_correct_c_g_forreal = brm(regr$formula, data=regr$data, family = 'bernoulli')
summary(m_correct_c_g_forreal)

m_correct_genlevel_cond_forreal_eff = marginal_effects(
    m_correct_c_g_forreal, "genlevel:conditional")
marginal_effects_table(m_correct_c_g_forreal, "genlevel:conditional")

# Next, choose
regr = standardize(correct ~ conditional * genlevel + (1|test_id),
                   data=df_gen[df_gen$type == "choose",], family = 'binomial', scale = 1)
m_correct_c_g_choose = brm(regr$formula, data=regr$data, family = 'bernoulli')
summary(m_correct_c_g_choose)

m_correct_genlevel_cond_choose_eff = marginal_effects(
    m_correct_c_g_choose, "genlevel:conditional")
marginal_effects_table(m_correct_c_g_choose, "genlevel:conditional")

plots = cowplot::plot_grid(plot(m_correct_genlevel_cond_forreal_eff)[[1]],
                           plot(m_correct_genlevel_cond_choose_eff)[[1]],
                           labels = c("a)", "b)"), align = "h")
cowplot::save_plot("../images/cond-gen-interactions.pdf", plots, dpi=300, base_width = 10,
                   base_height = 5, base_aspect_ratio = 1.3)

##########################################################################################
## Perceived authenticity
##########################################################################################

# The perceived authenticity is equal to the user answer in forreal questions
df$perceived = df$user_answer
# we should only consider bias in forreal questions, as it doesn't make much 
# make much sense for choose questions.

# there is not a clear bias towards real or generated:
prop.table(table(df[df$type == "forreal",]$perceived))

## but this appears only to be the case in choose questions.
aggregate(perceived ~ type, df, mean)

## Is there a bias towards fake or real
p_baseline <- brm(perceived ~ 1 + (1|test_id), data=df[df$type == "forreal",],
                  control=list(adapt_delta=0.95),
                  family="bernoulli")
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
round(exp(b), 1)

# as the game progresses, does the bias change?
regr = standardize(perceived ~ trial_id + (1|test_id),
                   data=df[df$type == "forreal",], family = 'binomial')
mod = glmer(regr$formula, data=regr$data, family="binomial")
summary(mod)
Anova(mod) # it seems so, yes
plot_model(mod, type="pred", terms=c("trial_id"))


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
df[df$type == "choose" & df$source == "fake" & df$correct == "True", "perceived"] = 0
df[df$type == "choose" & df$source == "real" & df$correct == "True", "perceived"] = 1
df[df$type == "choose" & df$source == "fake" & df$correct == "False", "perceived"] = 1
df[df$type == "choose" & df$source == "real" & df$correct == "False", "perceived"] = 0
df$trial_id = df$level * 5 + df$iteration
df = df[df$trial_id <= 11,] # 10 rounds + sudden death
df$perceived = factor(df$perceived, levels=c(0, 1), labels=c("generated", "authentic"))
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15

## # choose items are represented by two rows, one for the generated fragment
## # and one for the authentic fragment. We only need to include one of them in 
## # the analysis, so we remove all real rows in choose rows.
## df <- df[df$type == "forreal" | (df$type == "choose" & df$source == "real"),]

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
regr = standardize(formula, data=df[df$type == "forreal",], family = "binomial", scale=0.5)
m_objective = brm(regr$formula, data=regr$data, family = "bernoulli")
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

value.size<-8
p_objective <- plot_model(m_objective, show.values = TRUE,
           title = "Objective feature importance", bpe = "mean",
           prob.inner = .5,
           prob.outer = .95,
           value.size=value.size,
           transform = NULL,
           order.terms = pred.order,
           axis.labels=pred.names,
           theme="theme_bw"
           ) + ylim(c(-1.25, 3)) + theme(text=element_text(size=14))


formula = as.formula(paste("perceived ~ (", paste(predictors, collapse = "+"), ") + (1|test_id)"))
regr <- standardize(formula, data=df[df$type == "forreal",], family = "binomial", scale = 0.5)
m_subjective = brm(regr$formula, data=regr$data, family = "bernoulli",
                   control=list(adapt_delta=0.95))
summary(m_subjective)

b <- summary(m_subjective)$fixed[, c(1, 3, 4)]
round(exp(b), 1)

p_subjective <- plot_model(m_subjective, show.values = TRUE,
           title = "Subjective feature importance", bpe = "mean",
           prob.inner = .5,
           prob.outer = .95,
           value.size=value.size,
           transform = NULL,
           order.terms = pred.order,
           geom.label.size = 0
           ) + ylim(c(-0.25, 0.75)) + theme(axis.text.y.left = element_text(size=0), text=element_text(size=14))


plots = cowplot::plot_grid(p_objective, NULL, p_subjective, nrow=1,
                           align="h", rel_widths = c(1.45, 0.1, 1))
cowplot::save_plot("../images/feature-importance.png", plots, dpi=300, base_width=15,
                   base_height=8, bg = "transparent")



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

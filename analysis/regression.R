library(brms)
require(ggplot2)
library(lme4)
library(sjPlot)
library(car)

df <- read.csv("../data/db_data.csv")
df <- subset(df, select=-c(real, fake))
## df <- df[df$type == "forreal", ]
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
# remove all games which were early stopped
df <- df[df$test_id %in% names(table(df$test_id))[table(df$test_id) >= 10],]
df$test_id <- as.factor(df$test_id)
df$correct <- abs(1 - as.integer(df$correct))
df$trial_id = df$level * 5 + df$iteration
df = df[df$trial_id <= 10,]
df$trial_id = df$trial_id / 10 # rescale for easier interpretation of coefficients
df$time = df$time / 1000 # convert to seconds
df$time[df$time > 15] = 15
df$time = scale(df$time) # scale and center with mean = 0
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

## Is the forreal game easier than the choose game?
mod = glmer (correct ~ type + (1|test_id), data=df,
             family='binomial', nAGQ=0)
summary(mod)
drop1(mod, test='Chisq')
## it appears that participants perform significantly worse on the forreal test
## Test again with bayesian model

m_type = brm(correct ~ type + (1|test_id), data=df,
             family='bernoulli')
summary(m_type)

## Is there a training effect in the game? That is, do participants
## get more experienced throughout the game
mod = glmer (correct ~ trial_id + (1|test_id), data=df,
             family='binomial')
summary(mod)
drop1(mod, test='Chisq')

## it appears there is 
## test again with bayesian LM
m_trial = brm(correct ~ trial_id + (1|test_id), data=df, family='bernoulli')
summary(m_trial)
exp(quantile(as.matrix(mod)[,2], probs=c(.5, .025, .975)) * 100)

## Does the trial effect depend on the type of game?
mod = glmer (correct ~ trial_id * type + (1|test_id),
             data=df[df$trial_id <= 10,], family='binomial')
summary(mod)
Anova(mod) # no, it doesn't
plot_model(mod, type="pred", terms=c("trial_id", "type"))

## Does the trial effect depend on whether real or fake was the correct answer?
mod = glmer(correct ~ trial_id * true_answer * type + (1|test_id), data=df, family="binomial")
summary(mod)
Anova(mod) # yes it does, quite strongly. 
plot_model(mod, type="pred", terms=c("trial_id", "true_answer", "type"))

# is the time participants took advantageous?
mod <- glmer(correct ~ time * type + (1|test_id), data=df, family="binomial")
summary(mod)
Anova(mod) # no, it appears that participants perform worse when they
           # take more time. How to explain this? Is that because the 
           # examples are harder? Or is that irrespective of the 
           # difficulty of the examples?
plot_model(mod, type="pred", terms=c("time", "type"))

## test for generation level effects (since we noticed a training effect
## incorporate this into the model as an interaction effect)
df_gen <- df[((df$true_answer == 0) & (df$type == "forreal")) | df$type == "choose",]

# without forreal=real tests, is there an effect of training experience?
mod = glmer(correct ~ trial_id + (1|test_id), data=df_gen, family="binomial")
summary(mod)
Anova(mod) # no there isn't
plot_model(mod, type="pred", terms=c("trial_id"))

# but it seems to differ between game types:
mod = glmer(correct ~ trial_id * type + (1|test_id), data=df_gen, family="binomial")
summary(mod)
Anova(mod) # cf. the significant interaction.
plot_model(mod, type="pred", terms=c("trial_id", "type"))

# having established all these (interacting) effects, we go for the following model:
mod = glmer(correct ~ genlevel * (type * trial_id) + (1|test_id),
            data=df_gen, family="binomial", nAGQ=0)
summary(mod)
Anova(mod) # there is no significant interaction between genlevel and (trial * type)
plot_model(mod, type = "pred", terms = c("trial_id", "genlevel", "type"))

m_correct <- brm(correct ~ genlevel * (type * trial_id) + (1|test_id),
                 data=df_gen, family='bernoulli')
summary(m_correct)
plot(m_correct)

# next we can do the same with conditional (we might also combine them in a single model)
mod = glmer(correct ~ conditional * (type * trial_id) + (1|test_id),
            data=df_gen, family="binomial")
summary(mod)
Anova(mod) # there is no significant interaction between conditional and (trial * type)
plot_model(mod, type = "pred", terms = c("trial_id", "conditional", "type"))


##########################################################################################
## Perceived authenticity
##########################################################################################

dffr = df[(df$type == "forreal"),]

# there is no bias towards real or fake:
prop.table(table(dffr$user_answer))

## but original lyrics appear more likely to be classifified as real:
aggregate(user_answer ~ true_answer, dffr, mean)

## Is there a bias towards fake or real
p_baseline <- brm(user_answer ~ 1 + (1|test_id), data=dffr, family="bernoulli")
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
exp(b) # no, there is not

# repeat with lme4
mod = glmer(user_answer ~ 1 + (1|test_id), data=dffr, family="binomial")
summary(mod)

# as the game progresses, does the bias change?
mod = glmer(user_answer ~ trial_id * true_answer + (1|test_id),
            data=dffr, family="binomial")
summary(mod)
Anova(mod) # it seems so, yes
plot_model(mod, type="pred", terms=c("trial_id"))

# is this any differerent for real or fake examples?
mod = glmer(user_answer ~ trial_id * true_answer + (1|test_id),
            data=dffr, family="binomial")
summary(mod)
Anova(mod) # yes, it is: the bias towards real increases over time
plot_model(mod, type="pred", terms=c("trial_id", "true_answer"))

# We previously saw an interesting effect of time per question on the objective
# authenticity. Does that also affect the bias towards fake or real?
mod = glmer(user_answer ~ time * true_answer + (1|test_id), data=dffr, family="binomial")
summary(mod)
Anova(mod) # yes, it does, though especially for real examples.
plot_model(mod, type="pred", terms=c("time", "true_answer"))

# note, however, that there is evidence that participants differ
# substantially in their response to time
mod2 = glmer(user_answer ~ time * true_answer + (1 + time|test_id),
             data=dffr, family="binomial", nAGQ=0)
summary(mod2)
# compare the two models with anova:
anova(mod, mod2) # we would need to test this with Bayesian models
                 # using LOO, which is more robust for this kind of thing

# Test for effect of different models on perceived authenticity. First
# we restrict the data to the forreal examples presenting generated lyrics:
df_gen <- df[(df$true_answer == 0) & (df$type == "forreal"),]

# is there an effect of the different models on subjective authenticity?
mod = glmer(user_answer ~ genlevel * trial_id + (1|test_id), data=df_gen, family = "binomial")
summary(mod)
Anova(mod) # yes, again for syllable and hybrid
plot_model(mod, type="pred", terms=c("time", "genlevel"))

m_real <- brm(user_answer ~ genlevel * trial_id + (1|test_id),
              data=df, family='bernoulli')
summary(m_real)
plot(m_real)

# is there an effect of conditional models on subjective authenticity?
mod = glmer(user_answer ~ conditional * trial_id + (1|test_id),
            data=df_gen, family = "binomial")
summary(mod)
Anova(mod) # yes, conditionally generated lyrics are more often judged to be real
           # and, interestingly, don't appear to be affected by a training interaction.
plot_model(mod, type="pred", terms=c("trial_id", "conditional"))

# combining conditioning and genlevel as non-interacting effects, we 
# observe similar patterns:
mod = glmer(user_answer ~ (conditional + genlevel) * trial_id + (1|test_id),
            data=df_gen, family = "binomial")
summary(mod)
Anova(mod) # We see that conditioning on average pushes participants into 
           # the direction of "real" judgements and the hybrid and syllable
           # models are, in comparison to the char level model, 
           # more often judged to be real. Furthemore, We observe a slight 
           # effect of training in the non-conditioning case whereas this does not
           # exists for lyrics generated _with_ conditioning
plot_model(mod, type="pred", terms=c("trial_id", "genlevel", "conditional"))

m_real.conditional <- brm(user_answer ~ (genlevel + conditional) * trial_id + (1|test_id),
                         data=df, family='bernoulli')
summary(m_real.conditional)
plot(m_real.conditional)


##########################################################################################
## Linguistic Feature analysis
##########################################################################################

sample2pair <- read.csv('../data/id2pairid.csv', sep="\t")

features <- read.csv('../data/samples.features.csv')
features$X <- NULL # drop index column

# add pair_ids to each sample
features <- merge(x=features, y=sample2pair, by.x="sample_id", by.y="fake", all.x=TRUE)
features <- merge(x=features, y=sample2pair, by.x="sample_id", by.y="true", all.x=TRUE)
features$pair_id <- features$pair.x
features$pair_id[is.na(features$pair.x)] <- features$pair.y[is.na(features$pair.x)]
features[, c("pair.x", "pair.y", "true", "fake")] <- list(NULL) # drop unused columns
features <- features[complete.cases(features[, c("pair_id")]),] # filter NAs
features <- subset(features, select=-c(line))

# next merge feature table with main dataframe for user judgements
features = merge(x=features, y=df, by="pair_id")

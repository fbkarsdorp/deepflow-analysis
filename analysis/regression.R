library(brms)
require(ggplot2)
library(lme4)

df <- read.csv("../data/db_data.csv")
df <- subset(df, select=-c(real, fake))
## df <- df[df$type == "forreal", ]
df <- df[df$user_answer != 0,]
df$user_answer <- 2 - df$user_answer
df$true_answer <- 2 - df$true_answer
df$test_id <- as.factor(df$test_id)
df$correct <- abs(1 - as.integer(df$correct))
df$trial_id = df$level * 5 + df$iteration
df$time = df$time / 10000
head(df)

## Objective authenticity

## First test for a baseline model
prop.table(table(df$correct))
m_baseline <- brm(correct ~ 1 + (1|test_id), data=df, family="bernoulli")
summary(m_baseline)
plot(mod)
# compute the odds
b <- summary(m_baseline)$fixed[1, c(1, 3, 4)]
exp(b)

## Is the forreal game easier than the choose game?
mod = glmer (correct ~ type + (1|test_id), data=df[df$trial_id <= 10,],
             family='binomial', nAGQ=0)
summary(mod)
drop1(mod, test='Chisq')
## it appears that participants perform significantly worse on the forreal test
## Test again with bayesian model

m_type = brm(correct ~ type + (1|test_id), data=df[df$trial_id <= 10,],
             family='bernoulli')
summary(m_type)

## Is there a training effect in the game? That is, do participants
## get more experienced throughout the game
library (lme4)
mod = glmer (correct ~ trial_id + (1|test_id), data=df[df$trial_id <= 10,],
             family='binomial', nAGQ=0)
summary(mod)
drop1(mod, test='Chisq')

## it appears there is 
## test again with bayesian LM
m_trial = brm(correct ~ trial_id + (1|test_id), data=df, family='bernoulli')
summary(m_trial)
exp(quantile(as.matrix(mod)[,2], probs=c(.5, .025, .975)) * 100)

## Does the trial effect depend on the type of game?
mod = glmer (correct ~ trial_id * type + (1|test_id),
             data=df, family='binomial')
summary(mod)
drop1(mod, test='Chisq') # no, it doesn't

# is the time participants took advantageous?
mod <- glmer(correct ~ time + (1|test_id), data=df, family="binomial")
summary(mod)

## test for generation level effects
m_correct <- brm(correct ~ genlevel + (1|test_id),
                 data=df, family='bernoulli')
summary(m_correct)
plot(m_correct)

m_real.conditional <- brm(user_answer ~ genlevel + conditional + (1|test_id),
                         data=df, family='bernoulli')
summary(m_real.conditional)
plot(m_real.conditional)


## Perceived authenticity
df = df[(df$type == "forreal"),]

# there is no bias towards real or fake:
prop.table(table(df$user_answer))

## but original lyrics appear more likely to be classifified as real:
aggregate(user_answer ~ true_answer, df, mean)

## us there a bias towards fake or real
p_baseline <- brm(user_answer ~ 1 + (1|test_id), data=df, family="bernoulli")
summary(p_baseline)
plot(p_baseline)
# compute the odds
b <- summary(p_baseline)$fixed[1, c(1, 3, 4)]
exp(b) # yes, it appears there is towards fake


df <- df[(df$true_answer == 0) & (df$type == "forreal"),]

m_real <- brm(user_answer ~ genlevel + (1|test_id),
              data=df, family='bernoulli')
summary(m_real)
plot(m_real)

m_real.conditional <- brm(user_answer ~ genlevel + conditional + (1|test_id),
                         data=df, family='bernoulli')
summary(m_real.conditional)
plot(m_real.conditional)

aggregate(user_answer ~ genlevel, df, mean)
aggregate(user_answer ~ genlevel + conditional, df, mean)

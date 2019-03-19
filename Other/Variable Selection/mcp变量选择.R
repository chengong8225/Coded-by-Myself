##lambda>0 控制正则化的数量（具体是什么范数）
##lambda->0+ L1范数；  lambda->∞ L0范数
##delta>0  控制凹性的程度
##t是待估计的参数的绝对值 要求t>=0

time.start <- Sys.time()
# t <- beta
mcp.penalty <- function(t, lambda, Gamma = 1.5)
{
  return(ifelse(abs(t)<=lambda*Gamma, lambda*abs(t) -t^2/(2*Gamma), lambda^2*Gamma/2)) 
}

#把数据拆分
split.data <- function(len, fold=10)
{
  index <- sample(len, len, replace = FALSE)
  cutpoint <- floor(len/fold)
  kfold <- list()
  for(i in 1:fold)
  {
    if (i == fold)
      kfold[[i]] <- index[(cutpoint*(i-1)+1):len]
    else
      kfold[[i]] <- index[(1+cutpoint*(i-1)):(cutpoint*(i))]
  }
  return(kfold)
}

# beta <- initial.beta
#mcp的损失函数
loss.function <- function(x, y, beta, lambda, Gamma)
{
  sum1 = sum((y - x%*%beta)^2)/2/length(y)
  sum2 = sum(mcp.penalty(beta, lambda, Gamma))
  return(sum1+sum2)
}

#利用坐标下降法估计beta，把beta的所有分量更新一次
estimate.beta.by.cd <- function(x, y, beta, lambda, Gamma)
{
  beta.new <- beta
  for(i in 1:ncol(x))
  {
    y.new <- y - x[,-i] %*% beta.new[-i]
    tmp <- sum(x[,i]*y.new)/sum(x[,i]*x[,i]) 
    if (abs(tmp)<= lambda*Gamma)
      beta.new[i] <- sign(tmp)*max(0,abs(tmp)-lambda)/(1-1/Gamma)
    else
      beta.new[i] <- tmp
  }
  return(beta.new)
}

get.lambda.by.cv <- function(x, y, fold, iteration)
#x是第一列全为1的设计矩阵
{
  N <- nrow(x)
  p <- ncol(x)
  ## fold <- 10
  ## iteration <- 200
  lambda <- seq(0.05,0.2, by=0.001)
  Gamma <- 3
  rss.all <- rep(0, length(lambda))
  for (i in 1:length(lambda))
  {
    index <- split.data(N, fold)
    rss <- rep(0, fold)
    for (j in 1:fold)
    {
      x.train <- x[setdiff(1:N, index[[j]]),] 
      y.train <- y[setdiff(1:N, index[[j]]),] 
      x.test <- x[index[[j]],]
      y.test <- y[index[[j]],]
      
      beta <- rep(0,p)
      beta.opt <- beta
      for (k in 1:iteration)
      {
        beta.opt <- estimate.beta.by.cd(x, y, beta, lambda[i], Gamma)
        beta <- beta.opt
      }
      #用更新后的beta计算当前rss
      rss[j] <- sum((y.test - x.test %*% beta))
    }
    rss.all[i] <- sum(rss)
  }

  #选择最小rss.all对应的lambda
  lambda.opt <- lambda[which.min(rss.all)]
  return(lambda.opt) 
}

library(MASS)
#得到数据
get.data <- function(n, p, beta, intercept, sig)
{
  x <- matrix(0,n,p)
  Sigma <- matrix(rep(0,p*p),byr=T,ncol=p)
  diag(Sigma) <- rep(1,p) # compound symmatry correlation
  x <- mvrnorm(n,rep(0,p),Sigma)
  y <- intercept + x%*%beta + sig*rnorm(n)
  list(x=x, y=y)
}

N <- 200; p <- 20
beta0 <- rep(0,p)
beta0[c(1,2,3)] <- c(1,2,4)
l <- get.data(N, p, beta0, 3, 0.5)
x <- cbind(rep(1, N),l$x); y <- l$y

initial.beta <- rnorm(p+1)

Gamma <- 3
fold.k <- 10
iter <- 300
lambda.opt <- get.lambda.by.cv(x, y, fold.k, iter) #0.009
# lambda.opt <- 0.1

for (i in 1:400)
{
  beta.new <- estimate.beta.by.cd(x, y, initial.beta, lambda.opt, Gamma)
  initial.beta <- beta.new
}

beta.new
time.end <- Sys.time()
runtime <- time.end - time.start

library(ncvreg)
mcp.object <- ncvreg(l$x, l$y, penalty = "MCP", lambda = lambda.opt, gamma = 3)
mcp.object$beta

###############################
##--Local Linear Regression--##
###############################

#(x,y)是样本点
num <- 100
x <- runif(num, min = 0, max = 1)
x <- sort(x, decreasing = FALSE)
epsilon <- rnorm(n = num, mean = 0, sd = 1/3)
y <- sin(4*x)+epsilon


##(e.x, e.y)是样本点，xx是待估计的点的x，yy是待估计点的y估计值
Epanechnikov <- function(lambda,e.x,e.y,xx)
{
  n <- length(e.x)
  K <- rep(0,n)
  for (i in 1:n)
  {
    temp <- abs(xx -e.x[i])/lambda
    K[i] <- ifelse(abs(temp)<=1,0.75*(1-temp^2),0)
  }
  yy <- sum(K*e.y)/sum(K)
  return(yy)
}

#把需要的红点的坐标返回
Red.point <- function(lambda,e.x,e.y,xx)
{
  n <- length(e.x)
  K <- rep(0,n)
  for (i in 1:n)
  {
    temp <- abs(xx -e.x[i])/lambda
    K[i] <- (temp<=1)
  }
  return(K)
}

#每个点的权重
Weights.for.NW <- function(lambda,xx0,xx)
{
  weight <- rep(0,length(xx))
  K <- rep(0,length(xx))
  for (i in 1:length(xx))
  {
    temp <- abs(xx0-xx[i])/lambda
    K[i] <- ifelse(abs(temp)<=1,0.75*(1-temp^2),0)
  }
  return(K)
}

x.real <- seq(0.01,1,0.01)
y.real <- sin(4*x.real)
x.est.nw <- seq(0.01,1,0.01)
y.est.nw <- seq(0.01,1,0.01)
for (i in 1:length(x.est.nw))
{
  y.est.nw[i] <- Epanechnikov(lambda = 0.2, x, y, x.est.nw[i])
}


#我们要对X=0.5这个点进行估计
X <- 0.05
Y <- Epanechnikov(lambda = 0.2,x,y,X)
red.points <- Red.point(lambda = 0.2,x,y,X)
####开始画图
require(grDevices)
par(mfrow=c(1,2))
plot(0,0,xlim = c(0,1), ylim = c (-1,1.5), type = "n",xlab = "x",ylab = "y")
title("N-W Knenel at Boundary")
#画出黄色高亮区域和红色的点
# xrange <- seq(from=x[min(which(red.points>0))] , to=x[max(which(red.points>0))] , by=0.01)
# polygon(xrange,*(1-(xrange-X)^2),col = "yellow",border = "yellow")

w <- Weights.for.NW(0.2,X,x.real)
polygon(c(x.real,0.01),c(w, 0) ,col = "yellow",border = "yellow")

lines(x.real, y.real, lty=1, type = "l", col="black")

#画出样本点的散点图
lines(x,y, type = "p",col="deepskyblue1")

#画出NW核的结果
lines(x.est.nw,y.est.nw, type = "l",col="blue")

#穿过估计点的水平的红色直线
h.x <- c(x[min(which(red.points>0))],x[max(which(red.points>0))])
h.y <- c(Y,Y)
lines(h.x,h.y,lty=1, type = "l",col="red")

#穿过估计点的垂直的红色直线
v.x <- c(X,X)
v.y <- c(Y,-1.2)
lines(v.x,v.y,lty=1, type = "l",col="red")

#黄色区域内的点描红
lines(x[which(red.points>0)],y[which(red.points>0)],type = "p",col="red")

lines(X,Y,type = "p",col="red",pch=16)








#得到对角矩阵w
get.W <- function(lambda,e.x,e.y,xx)
{
  n <- length(e.x)
  K <- rep(0,n)
  for (i in 1:n)
  {
    temp <- abs(xx -e.x[i])/lambda
    K[i] <- ifelse(abs(temp)<=1,0.75*(1-temp^2),0)
  }
  W <- diag(K)
  return(W)
}

#把用到的点描红
get.weights <- function(lambda, xx0, xx)
{
  K <- rep(0,length(xx))
  for (i in 1:length(xx))
  {
    temp <- abs(xx0-xx[i])/lambda
    K[i] <- ifelse(abs(temp)<=1,0.75*(1-temp^2),0)
  }
  return(K)
}


x.real <- seq(0.01,1,0.01)
y.real <- sin(4*x.real)
x.est.llr <- seq(0.01,1,0.01)
y.est.llr <- seq(0.01,1,0.01)

#对点X=0.5进行估计
X <- 0.05
w <- get.W(lambda = 0.2, x, y, X)

B <- as.matrix(t(rbind(rep(1,num),x)))
b.x0 <- c(1,0.05)
a<- b.x0%*%solve(t(B)%*%w%*%B)%*%t(B)%*%w%*%y

local.linear.regression <- function(lambda, b, Y, XX)
{
  result <- rep(0,length(XX))
  for(i in 1:length(XX))
  {
    W <- get.W(lambda , x, y, XX[i])
    x.new <- c(1,XX[i])
    result[i] <- x.new%*%solve(t(b)%*%W%*%b)%*%t(b)%*%W%*%Y
  }
  return(result)
}
y.est.llr <- local.linear.regression(lambda = 0.2, B, y, x.est.llr)
Y <- local.linear.regression(lambda = 0.2, B, y, X)

##开始画图
plot(0,0,xlim = c(0,1), ylim = c (-1,1.5), type = "n",xlab = "x",ylab = "y")
title("Local Linear Regression at Boundary")
#画出黄色高亮区域和红色的点
# xrange <- seq(from=x[min(which(red.points>0))] , to=x[max(which(red.points>0))] , by=0.01)
# polygon(xrange,*(1-(xrange-X)^2),col = "yellow",border = "yellow")

w <- get.weights(0.2,X,x.real)
polygon(c(x.real,0.01),c(w, 0) ,col = "yellow",border = "yellow")
#画出真实曲线y=sin(4x)
lines(x.real, y.real, lty=1, type = "l", col="black")

#画出样本点的散点图
lines(x,y, type = "p",col="deepskyblue1")

#画出NW核的结果
lines(x.est.llr,y.est.llr, type = "l",col="blue")


red.points <- get.weights(lambda = 0.2, X, x)
#穿过估计点的水平的红色直线
h.x <- c(x[min(which(red.points>0))],x[max(which(red.points>0))])
h.y <- c(Y,Y)
lines(h.x,h.y,lty=1, type = "l",col="red")

#穿过估计点的垂直的红色直线
v.x <- c(X,X)
v.y <- c(Y,-1.2)
lines(v.x,v.y,lty=1, type = "l",col="red")

#黄色区域内的点描红
lines(x[which(red.points>0)],y[which(red.points>0)],type = "p",col="red")

lines(X,Y,type = "p",col="red",pch=16)


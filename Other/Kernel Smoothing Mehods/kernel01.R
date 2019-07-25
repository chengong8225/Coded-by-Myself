##Figure 6.1

##############################
##------------KNN-----------##
##############################

#(x,y)��������
num <- 100
x <- runif(num, min = 0, max = 1)
x <- sort(x, decreasing = FALSE)
epsilon <- rnorm(n = num, mean = 0, sd = 1/3)
y <- sin(4*x)+epsilon

##�����(x,y)��ȫ��������(X,Y)��K���ڽ��
K.nearest.neighbor <- function(x,X,Y)
{
  d <- abs(X-x)
  return(c(mean(X[order(d)[c(1:30)]]),mean(Y[order(d)[c(1:30)]])))
}

x.real <- seq(0.01,1,0.01)
y.real <- sin(4*x.real)
y.est.epan <- seq(0.01,1,0.01)

x.est.knn <- x.real
y.est.knn <- x.real
for (i in 1:length(x.est.knn))
{
  tmp <- K.nearest.neighbor(x.real[i],x,y)
  y.est.knn[i] <- tmp[2]
}

#����Ҫ��X=0.5�������й���
X <- 0.5
Y <- K.nearest.neighbor(X,x,y)[2]
#�����ҵ���x=a�������k(k=30)���㣬���ر߽��xmin,xmax,ymin,ymax
dist <- function(a,b,A,B)
{
  dis <- abs(A-a)
  index <- order(dis)[1:30]
  para <- c(min(A[index]),max(A[index]),min(B[index]),max(B[index]))
  return(para)
}

p <- dist(X,Y,x,y)

####��ʼ��ͼ
par(mfrow=c(1,2))
require(grDevices)
plot(0,0,xlim = c(0,1), ylim = c (-1,1.5), type = "n",xlab = "x",ylab = "y")
title("Nearest-Neighbor Kernel")

#������ɫ��������ͺ�ɫ�ĵ�
rect(xleft = p[1], xright = p[2], ybottom = 0,  ytop = 1,border = "yellow",col = "yellow")

#������ʵ����y=sin(4x)
lines(x.real, y.real, lty=1, type = "l", col="black")

#�����������ɢ��ͼ
lines(x,y, type = "p",col="deepskyblue1")

#����K���ڵĽ��
lines(x.est.knn,y.est.knn, type = "l",col="blue")
# lines(x[p],y[p],type = "p",col ="green")

#�������Ƶ��ˮƽ�ĺ�ɫֱ��
h.x <- c(p[1],p[2])
h.y <- c(Y,Y)
lines(h.x,h.y,lty=1, type = "l",col="red")

#�������Ƶ�Ĵ�ֱ�ĺ�ɫֱ��
v.x <- c(X,X)
v.y <- c(Y,-1.2)
lines(v.x,v.y,lty=1, type = "l",col="red")

#��ɫ�����ڵĵ����
index <- order(abs(x-X))[1:30]
lines(x[index],y[index],type = "p",col="red")

lines(X,Y,type = "p",col="red",pch=16)






##(e.x, e.y)�������㣬xx�Ǵ����Ƶĵ��x��yy�Ǵ����Ƶ��y����ֵ
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

#����Ҫ�ĺ������귵��
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

#ÿ�����Ȩ��
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


#����Ҫ��X=0.5�������й���
X <- 0.5
Y <- Epanechnikov(lambda = 0.2,x,y,X)
red.points <- Red.point(lambda = 0.2,x,y,X)

plot(0,0,xlim = c(0,1), ylim = c (-1,1.5), type = "n",xlab = "x",ylab = "y")
title("Epanechnikov Kernel")
#������ɫ��������ͺ�ɫ�ĵ�
# xrange <- seq(from=x[min(which(red.points>0))] , to=x[max(which(red.points>0))] , by=0.01)
# polygon(xrange,*(1-(xrange-X)^2),col = "yellow",border = "yellow")

w <- Weights.for.NW(0.2,X,x.real)
polygon(c(x.real,0.01),c(w, 0) ,col = "yellow",border = "yellow")
#������ʵ����y=sin(4x)
lines(x.real, y.real, lty=1, type = "l", col="black")

#�����������ɢ��ͼ
lines(x,y, type = "p",col="deepskyblue1")

#����NW�˵Ľ��
lines(x.est.nw,y.est.nw, type = "l",col="blue")

#�������Ƶ��ˮƽ�ĺ�ɫֱ��
h.x <- c(x[min(which(red.points>0))],x[max(which(red.points>0))])
h.y <- c(Y,Y)
lines(h.x,h.y,lty=1, type = "l",col="red")

#�������Ƶ�Ĵ�ֱ�ĺ�ɫֱ��
v.x <- c(X,X)
v.y <- c(Y,-1.2)
lines(v.x,v.y,lty=1, type = "l",col="red")

#��ɫ�����ڵĵ����
lines(x[which(red.points>0)],y[which(red.points>0)],type = "p",col="red")

lines(X,Y,type = "p",col="red",pch=16)

##Figure 6.2
t <- seq(-3,3,0.05)
#Epanechnikov
D.e <- ifelse(abs(t)<=1,3/4*(1-t^2),0)
#Tri-cube
D.t <- ifelse(abs(t)<=1,(1-(abs(t))^3)^3,0)
#Gaussian
D.g <- dnorm(t,0,1)

plot(t, D.t, lty=1, type = "b", pch=16 ,col="black",xlab = "t" ,ylab = expression(K[lambda](x[0],x)))
lines(t, D.e, lty=1, type = "b", pch=16 ,col="red")
lines(t, D.g, lty=1, type = "b", pch=16 ,col="yellow")
legend(1.5,0.9, legend=c("Tri-cube","Epanechnikov","Gaussian"), fill = c("black","red","yellow"))

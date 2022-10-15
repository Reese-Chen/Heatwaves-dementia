#load the package
library(groupICA)

# Generate data 
mydata = runif(30*365*180*180, min = 1, max = 2)
X = matrix(data=mydata,nrow=30*365,ncol=180*180)
result1 <- prcomp(X, center = TRUE,scale. = TRUE,rank.=100)
dim(X)

#Apply PCA on data of each specific year
if (false){
  Y = X[1:365,]
  head(Y)
  dim(Y)
  result1 <- prcomp(Y, center = TRUE,scale. = TRUE,rank.=100)
  result1$rotation
  result1$x
  summary(result1)
}
for (i in 1:30){
  start = 365*(i-1)+1
  end = 365*i
  Y = X[start:end,]
  result1 = prcomp(Y, center = TRUE,scale. = TRUE,rank.=100)
  if (i==1){ X1 = result1$x }
  else { X1 = rbind(X1,result1$x) }
  if (i==1){ matrix1 = result1$rotation }
  else { matrix1 = rbind(matrix1,result1$rotation) }
}
dim(X1)
dim(matrix1)

#Combine all the features found by PCA on each year and apply PCA on overall data 
result2 = prcomp(X1, center = TRUE, scale. = TRUE, rank. = 2)
X2 = result2$x
matrix2 = result2$rotation
X2 = as.data.frame(X2)
dim(X2)

# Apply groupICA
group_index <- rep(c(1:30), each=365)
partition_index <- rep(rep(1:73,5),30)
result3 <- groupICA(X1, group_index, partition_index, rank_components=TRUE)

summary(result3)
matrix3 = solve(result3$V)
S = result3$Shat

#Look for correspondent matrix A
#every 30 lines of matrix A represent the correspondence matrix for one year
A = matrix1%*%matrix3
dim(A)

#Set thresholds 
#typically we set the 90th thresholds for each varible
if (false){
  mydata = runif(30*100*180*180, min = 1, max = 2)
  A = matrix(data = mydata,nrow = 972000, ncol = 100)
}
for (i in 1:30){
  start = (i-1)*180*180+1
  end = i*180*180
  Y = A[start:end,]
  A_factor = matrix(0,nrow=32400,ncol=100)
  for (j in 1:100){
    p90 = quantile(Y[,j],0.9)
    A_factor[which(Y[,j]>=p90),j] = 1
  }
  if (i==1) {A1 = A_factor}
  else {A1 = rbind(A1,A_factor)}
}

#count the feature for each country in each year
country_feature = list()
for (i in 1:233){
  country_feature$countrycode[i] = matrix(0,nrow=30,ncol=100)
}
for (i in 1:30){
  start = (i-1)*180*180+1
  end = i*180*180
  for (j in 1:100){
  }
}

#read incidence data and prevalence data of dementia
setwd("D://非最适温度//代码")
incidence = read.table(file = "incidence of dementia.csv", header = T , 
                     sep = "," , fill = TRUE , encoding = "UTF-8")
head(incidence)
prevalence = read.table(file = "prevalence of dementia.csv", header = T , 
                        sep = "," , fill = TRUE , encoding = "UTF-8")
head(prevalence)


#calculate the correlation between the incidence of dementia and the characteristics of annual heat waves 





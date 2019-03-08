
# Detect credit card fraud using SVM




#Reading data and preparing data
#The correlation matrix
#The target distribution
#K-means
#The target distribution after K-means
#The display function of the ROC curve
#Decoupage of the index of the dataset
#The cross validation only on the train data
#One model with split train data
#One model with all train data and test data
#Accuracy of the model


#Reading data and preparing data
suppressMessages(library(tidyverse))

rm(list=ls())
seed <- 123456789
set.seed(seed)

read_csv("F:/creditcard.csv") %>% 
  mutate(Time = as.factor(round((Time / 3600) %% 24))) -> donnees 

donnees$Amount <- scale(donnees$Amount, center = TRUE, scale = TRUE)

donnees <- cbind(donnees[,31],cbind(donnees[,- c(1,31)], model.matrix( ~ Time -1 , data=donnees)))

#The correlation matrix
suppressMessages(library(reshape2))

corrmatrice<- round(cor(donnees[, c(1:29)]),2)
hc <- hclust(as.dist((1-corrmatrice)/2))
cormat <- corrmatrice[hc$order, hc$order]
cormat[lower.tri(cormat)]<-NA
melted_cormat <- melt(cormat, na.rm = TRUE)
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

ggheatmap +
  geom_text(aes(Var2,Var1,label=value),color="black",size=4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

rm('corrmatrice','hc','cormat','melted_cormat','ggheatmap')

#The target distribution
ggplot() + 
  geom_histogram(data = donnees, mapping = aes(Class) , color='red' , alpha=.4, stat="count")

#K-means
nbClusters=2000
system.time(cluster <- kmeans(x = donnees[donnees$Class == 0,-1],centers = nbClusters, iter.max = 20))
##    user  system elapsed 
## 234.612   2.920 237.596
donneesKM <- as.data.frame(cluster$centers)
donneesKM$Class <- 0
donneesKM <- rbind(donneesKM,donnees[donnees$Class == 1,])
donneesKM$Class <- ordered(donneesKM$Class,levels=c(1,0))
rm("donnees")
The target distribution after K-means
ggplot() + 
  geom_histogram(data = donneesKM, mapping = aes(Class) , color='red' , alpha=.4, stat="count")

#The display function of the ROC curve
suppressMessages(library(caret))
suppressMessages(library(tidyverse))
suppressMessages(library(plotROC))

donneesModelROCALL <- NULL
MatriceConfusion <- data.frame()

fun.auc.ggplot <- function (ModelPred, ModelProb, ModelCible, Title = '', echantillon = '', df = donneesModelROCALL){
  
  donneesModel <- data.frame(Cible = as.integer(ModelCible), 
                             Cible.Nom = as.character(ModelCible),
                             ModelProb = ModelProb)
  
  basicplot <- ggplot(donneesModel, aes(d = Cible, m = ModelProb)) + 
    geom_roc(n.cuts = 50, labelsize = 3, labelround = 4)
  
  basicplot<- basicplot  +
    style_roc(xlab='Le taux de faux Positifs(1 - Specificity)',
              ylab='Le taux de vrais positifs(Sensitivity)',
              theme = theme_grey)+
    annotate("text",x=0.5,y=0.5,
             label="AUC <= 0.5 prédiction pire qu'au hasard",
             color="red",size=6, angle=45) +
    ggtitle( paste('Surface sous courbe ROC (AUC) : ',round(calc_auc(basicplot)$AUC,8),"% --",Title)) + 
    coord_fixed(ratio = 1)
  
  donneesModel$Label <- rep(paste(Title," : ",round(calc_auc(basicplot)$AUC,4),"%"),length(ModelCible))
  donneesModel$Echantillon <- echantillon
  
  matconf <- caret::confusionMatrix(ModelPred, ModelCible,mode="everything")
  
  MatriceConfusion <<- rbind( MatriceConfusion,
                              data.frame(Nom=Title,
                                         AUC        =calc_auc(basicplot)$AUC,
                                         Accuracy   =matconf$overall[1],
                                         Kappa      =matconf$overall[2], 
                                         VP         =matconf$table[1,1],
                                         FP         =matconf$table[1,2],
                                         VN         =matconf$table[2,2],
                                         FN         =matconf$table[2,1],
                                         Sensitivity=matconf$byClass[1],  
                                         Specificity=matconf$byClass[2],
                                         Precision  =matconf$byClass[5],
                                         FScore1    =matconf$byClass[7],
                                         Prevalence =matconf$byClass[8],
                                         PPV        =matconf$byClass[3], 
                                         NPV        =matconf$byClass[4],
                                         row.names =NULL))
  
  donneesModelROCALL <<- rbind(df,donneesModel)
  
  basicplot
}
#Decoupage of the index of the dataset
set.seed(seed)
validationIndex <- createDataPartition(donneesKM$Class, p=0.8, list=FALSE)
donnees <- donneesKM[validationIndex,]
echantillonTest <- donneesKM[-validationIndex,]
ggplot() + 
  geom_histogram(data = donnees, mapping = aes(Class) , color='red' , alpha=.4, stat="count")+
  geom_histogram(data = echantillonTest   , mapping = aes(Class) , color='blue', alpha=.2, stat="count")

set.seed(seed)
partitions <- 32
validationIndex <- createDataPartition(donnees$Class, p=0.9, list=TRUE, times = partitions)
#The cross validation only on the train data
suppressMessages(library(kernlab))
models <- NULL 
prediction <- NULL
prediction.Probabilite<-NULL

for (i in 1:partitions) {
  echantillon <- i
  echantillonApprentissage            <- donnees[validationIndex[[i]],]
  echantillonValidation               <- donnees[-validationIndex[[i]],]
  
  
  
  models$ksvmL <- ksvm(Class~., data=echantillonApprentissage, prob.model=T,kernel = "rbfdot",C=2, sigma = 0.04226965)
  
  prediction$ksvmL <- predict(models$ksvmL, echantillonValidation, type="response")
  prediction.Probabilite$ksvmL <- predict(models$ksvmL, echantillonValidation, type="probabilities")
  
  
  fun.auc.ggplot(prediction$ksvmL,
                 prediction.Probabilite$ksvmL[,2], 
                 echantillonValidation$Class,
                 paste('Support Vector Machine Radial-',echantillon),
                 echantillon)
}
donneesModelROCALL %>%
  filter(Cible < 3) %>%
  ggplot(aes(d = Cible, m = ModelProb, color = Label)) + 
  geom_roc(n.cuts = 0) +
  style_roc(xlab='Le taux de faux Positifs(1 - Specificity)',
            ylab='Le taux de vrais positifs(Sensitivity)',theme = theme_grey)+
  annotate("text",x=0.5,y=0.5,label="AUC <= 0.5 prédiction pire qu'au hasard",color="red",size=6, angle=45) +
  ggtitle( paste("Surface sous courbe ROC (AUC) : ",median(MatriceConfusion$AUC),"±",sd(MatriceConfusion$AUC))) + 
  coord_fixed(ratio = 1)


library(ggrepel)
ggplot(data = MatriceConfusion, mapping = aes(x = Specificity, y = Sensitivity)) + 
  geom_point(aes(color = Nom),size = 8, alpha=.6)+
  geom_label_repel(aes(label = Nom),
                   box.padding   = 0.4, 
                   point.padding = 0.6)+
  ggtitle( paste("Les models Sensibilité =",mean(MatriceConfusion$Sensitivity),"±",(max(MatriceConfusion$Sensitivity) - min(MatriceConfusion$Sensitivity))/2,
                 " Spécificité = ",mean(MatriceConfusion$Specificity),"±",(max(MatriceConfusion$Specificity) - min(MatriceConfusion$Specificity))/2)) 



library(ggrepel)
ggplot(data = MatriceConfusion, mapping = aes(x = AUC, y = Accuracy)) + 
  geom_point(aes(color = Nom),size = 10, alpha=.4)+
  geom_label_repel(aes(label = Nom),
                   box.padding   = 0.4, 
                   point.padding = 0.6)+
  ggtitle( paste("Les models AUC =",mean(MatriceConfusion$AUC),"±",(max(MatriceConfusion$AUC) - min(MatriceConfusion$AUC))/2,
                 " Accuracy = ",mean(MatriceConfusion$Accuracy),"±",(max(MatriceConfusion$Accuracy) - min(MatriceConfusion$Accuracy))/2)) 



ggplot(data = MatriceConfusion,mapping = aes(x=Nom,y=FScore1, fill=Precision)) + 
  geom_bar(stat = "identity")+
  theme(axis.text.x=element_text(angle=45,hjust=1)) 


ggplot(data = MatriceConfusion,mapping = aes(x=Nom,y=Specificity, fill=Precision)) + 
  geom_bar(stat = "identity")+
  theme(axis.text.x=element_text(angle=45,hjust=1)) 




#One model with split train data
validationIndex <- createDataPartition(donnees$Class, p=0.9, list=FALSE)
echantillonApprentissage <- donnees[validationIndex,]
echantillonValidation    <- donnees[-validationIndex,]
models$ksvmL <- ksvm(Class~., data=echantillonApprentissage, prob.model=T,kernel = "rbfdot",C=2, sigma = 0.04226965)

prediction$ksvmL <- predict(models$ksvmL, echantillonValidation, type="response")
prediction.Probabilite$ksvmL <- predict(models$ksvmL, echantillonValidation, type="probabilities")

fun.auc.ggplot(prediction$ksvmL,
               prediction.Probabilite$ksvmL[,2], 
               echantillonValidation$Class,
               paste('Support Vector Machine Radial-',echantillon),
               echantillon)



#One model with all train data and test data
echantillonApprentissage <- donnees
echantillonValidation    <- echantillonTest
models$ksvmL <- ksvm(Class~., data=echantillonApprentissage, prob.model=T,kernel = "rbfdot",C=2, sigma = 0.04226965)

prediction$ksvmL <- predict(models$ksvmL, echantillonValidation, type="response")
prediction.Probabilite$ksvmL <- predict(models$ksvmL, echantillonValidation, type="probabilities")

fun.auc.ggplot(prediction$ksvmL,
               prediction.Probabilite$ksvmL[,2], 
               echantillonValidation$Class,
               paste('Support Vector Machine Radial-',echantillon),
               echantillon)



#Accuracy of the model
caret::confusionMatrix(prediction$ksvmL, echantillonValidation$Class,mode="everything")








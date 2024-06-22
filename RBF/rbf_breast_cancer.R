# Carregar bibliotecas necessárias
library(mlbench)
library(caret)
library(cluster)
library(nnet)
library(ggplot2)

# Carregar e limpar os dados
data("BreastCancer")
data2 <- BreastCancer
data2 <- data2[complete.cases(data2),]

# Remover coluna de ID
data2 <- data2[, -1]

# Converter variáveis de fatores para numéricas
data2[, 1:9] <- lapply(data2[, 1:9], as.numeric)

# Rotular as amostras das Classes com o valor de 0 (malígno) e 1 (benígno)
data2$Class <- ifelse(data2$Class == "malignant", 0, 1)

# Função Gaussiana
gauss <- function(x, centers, sigma) {
  apply(centers, 1, function(center) {
    exp(-0.5 * sum((x - center)^2) / (sigma^2))
  })
}

# Função para ajustar e treinar a RBF Network
fit_rbf_network <- function(X_train, y_train, n_clusters, sigma) {
  # Normalizar os dados
  scaler <- preProcess(X_train, method = c("center", "scale"))
  X_train_scaled <- predict(scaler, X_train)
  
  # Encontrar os centros dos clusters
  kmeans_model <- kmeans(X_train_scaled, centers = n_clusters, nstart = 20)
  centers <- kmeans_model$centers
  
  # Transformar os dados usando a RBF
  transformed_X_train <- t(apply(X_train_scaled, 1, function(row) gauss(row, centers, sigma)))
  
  # Adicionar nomes de colunas ao data frame transformado
  colnames(transformed_X_train) <- paste0("C", 1:n_clusters)
  
  # Treinar o modelo de regressão logística
  logistic_model <- multinom(y_train ~ ., data = as.data.frame(transformed_X_train))
  
  list(scaler = scaler, centers = centers, logistic_model = logistic_model, sigma = sigma)
}

# Função para prever usando a RBF Network
predict_rbf_network <- function(model, X_test) {
  X_test_scaled <- predict(model$scaler, X_test)
  transformed_X_test <- t(apply(X_test_scaled, 1, function(row) gauss(row, model$centers, model$sigma)))
  
  # Adicionar nomes de colunas ao data frame transformado
  colnames(transformed_X_test) <- paste0("C", 1:ncol(transformed_X_test))
  
  predict(model$logistic_model, as.data.frame(transformed_X_test))
}

# Variáveis
n_simulations <- 10
sigma <- 1.0
results <- data.frame()

# Loop de simulações
for (n_clusters in 2:10) {
  accuracies <- c()
  error_percents <- c()
  
  for (i in 1:n_simulations) {
    # Separar os dados em conjuntos de treino e teste
    train_index <- createDataPartition(data2$Class, p = 0.9, list = FALSE)
    X_train <- data2[train_index, 1:9]
    y_train <- data2[train_index, 10]
    X_test <- data2[-train_index, 1:9]
    y_test <- data2[-train_index, 10]
    
    # Treinar a RBF Network
    model <- fit_rbf_network(X_train, y_train, n_clusters, sigma)
    
    # Prever no conjunto de teste
    y_pred <- predict_rbf_network(model, X_test)
    
    # Calcular a acurácia
    accuracy <- mean(y_pred == y_test)
    accuracies <- c(accuracies, accuracy)
    
    # Calcular o erro percentual
    error_percent <- mean(y_pred != y_test) * 100
    error_percents <- c(error_percents, error_percent)
  }
  
  # Armazenar resultados
  mean_accuracy <- mean(accuracies)
  std_accuracy <- sd(accuracies)
  mean_error_percent <- mean(error_percents)
  std_error_percent <- sd(error_percents)
  
  results <- rbind(results, data.frame(
    Clusters = n_clusters,
    MeanAccuracy = mean_accuracy,
    StdDevAccuracy = std_accuracy,
    MeanErrorPercent = mean_error_percent,
    StdDevErrorPercent = std_error_percent
  ))
}

# Melhor número de clusters
best_clusters <- results[which.max(results$MeanAccuracy), "Clusters"]

# Exibir resultados
print(results)

# Melhor número de clusters
best_clusters <- results[which.min(results$MeanError), "Clusters"]

# Exibir resultados ordenados por erro percentual
results <- results[order(results$MeanError), ]
print(results)

# Plotar acurácia
ggplot(results, aes(x = Clusters, y = MeanAccuracy)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = MeanAccuracy - StdDevAccuracy, ymax = MeanAccuracy + StdDevAccuracy), width = 0.2) +
  labs(title = "Mean Accuracy vs. Number of Clusters", x = "Number of Clusters", y = "Mean Accuracy")

# Plotar erro percentual
ggplot(results, aes(x = Clusters, y = MeanErrorPercent)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = MeanErrorPercent - StdDevErrorPercent, ymax = MeanErrorPercent + StdDevErrorPercent), width = 0.2) +
  labs(title = "Mean Error Percent vs. Number of Clusters", x = "Number of Clusters", y = "Mean Error Percent")

# Melhor modelo
print(paste("Best number of clusters: ", best_clusters))

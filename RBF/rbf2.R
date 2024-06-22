# Carregar bibliotecas necessárias
library(MASS)        # Para mvrnorm
library(cluster)     # Para kmeans
library(caret)       # Para preProcess e createDataPartition
library(nnet)        # Para multinom
library(ggplot2)     # Para visualização

# Gerar dados XOR
generate_xor_data <- function(n_samples, mean, std) {
  set.seed(42)
  
  # Gerar amostras gaussianas para o problema XOR
  X1 <- mvrnorm(n_samples, mean, diag(std, 2))
  X2 <- mvrnorm(n_samples, c(mean[1], -mean[2]), diag(std, 2))
  X3 <- mvrnorm(n_samples, c(-mean[1], mean[2]), diag(std, 2))
  X4 <- mvrnorm(n_samples, c(-mean[1], -mean[2]), diag(std, 2))
  
  # Atribuir rótulos para criar o problema XOR
  y1 <- rep(0, n_samples)
  y2 <- rep(1, n_samples)
  y3 <- rep(1, n_samples)
  y4 <- rep(0, n_samples)
  
  # Concatenar todos os pontos de dados
  X <- rbind(X1, X2, X3, X4)
  y <- c(y1, y2, y3, y4)
  
  # Adicionar nomes de colunas
  colnames(X) <- c("V1", "V2")
  
  list(X = X, y = y)
}

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

# Configuração dos dados
n_samples <- 100
mean <- c(1.0, 1.0)
std <- 0.3
data <- generate_xor_data(n_samples, mean, std)
X <- data$X
y <- data$y

# Visualizar os dados gerados
print(head(X))
print(head(y))

# Preparar para simulações
n_simulations <- 10
accuracies <- c()
sigma <- 1.0
best_accuracy <- 0
best_model <- NULL

for (i in 1:n_simulations) {
  # Separar os dados em conjuntos de treino e teste
  train_index <- createDataPartition(y, p = 0.9, list = FALSE)
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  X_test <- X[-train_index, ]
  y_test <- y[-train_index]
  
  # Treinar a RBF Network
  model <- fit_rbf_network(X_train, y_train, n_clusters = 4, sigma = sigma)
  
  # Prever no conjunto de teste
  y_pred <- predict_rbf_network(model, X_test)
  
  # Calcular a acurácia
  accuracy <- mean(y_pred == y_test)
  accuracies <- c(accuracies, accuracy)
  
  # Salvar o melhor modelo
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_model <- model
  }
}

# Calcular acurácia média e desvio padrão
mean_accuracy <- mean(accuracies)
std_accuracy <- sd(accuracies)

# Relatório dos resultados
cat(sprintf("Mean Accuracy: %.2f%%\n", mean_accuracy * 100))
cat(sprintf("Standard Deviation: %.2f%%\n", std_accuracy * 100))

# Visualizar a melhor fronteira de decisão
plot1 <- ggplot(data = as.data.frame(X), aes(x = V1, y = V2, color = as.factor(y))) +
  geom_point() +
  ggtitle("Best RBF Network Decision Boundary") +
  xlab("X1") +
  ylab("X2") +
  theme_minimal()
print(plot1)

# Plotar fronteira de decisão
x_seq <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 100)
y_seq <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 100)
grid <- expand.grid(x_seq, y_seq)
colnames(grid) <- colnames(X)

grid_scaled <- predict(best_model$scaler, grid)
transformed_grid <- t(apply(grid_scaled, 1, function(row) gauss(row, best_model$centers, best_model$sigma)))
colnames(transformed_grid) <- paste0("C", 1:ncol(transformed_grid))

# Ajuste aqui: garantir que `transformed_grid` seja uma matriz e não uma lista
transformed_grid <- as.matrix(transformed_grid)

Z <- predict(best_model$logistic_model, as.data.frame(transformed_grid), type = "probs")[, 2]

grid$Z <- Z
plot2 <- ggplot(grid, aes(x = Var1, y = Var2, z = Z)) +
  geom_contour(aes(fill = ..level..), alpha = 0.3) +
  geom_point(data = as.data.frame(X), aes(x = V1, y = V2, color = as.factor(y))) +
  ggtitle("Best RBF Network Decision Boundary") +
  xlab("X1") +
  ylab("X2") +
  theme_minimal()
print(plot2)

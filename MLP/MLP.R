# Função de ativação linear
linear <- function(x) {
  return(x)
}

# Derivada da função de ativação linear
linear_derivative <- function(x) {
  return(1)
}

# # Função de ativação tanh
# tanh_activation <- function(x) {
#   return(tanh(x))
# }
# 
# # Derivada da função de ativação tanh
# tanh_derivative <- function(x) {
#   return(1 - tanh(x)^2)
# }

#Inicializando dados
x_train <- seq(from = 0, to = 2*pi, by = 0.15)
x_train <- x_train + (runif(length(x_train)) - 0.5)/5

i <- sample(length(x_train))

x_train <- x_train[i]
y_train <- sin(x_train)
y_train <- y_train + (runif(length(y_train)) - 0.5)/5

plot(x_train, y_train, col = 'blue', xlim = c(0, 2*pi),
     ylim = c(-1, 1), xlab = 'x', ylab = 'y')

x_test <- seq(from = 0, to = 2*pi, by = 0.01)
y_test <- sin(x_test)

par(new = T)

plot(x_test, y_test, col = 'red', type = 'l', xlim = c(0, 2*pi),
     ylim = c(-1, 1), xlab = 'x', ylab = 'y')

legend(x = 4, y = 1, legend = c('train', 'test'),
       col = c('blue', 'red'), pch = c('o', '_'))


# Inicialização dos pesos
input_neurons <- 1
hidden_neurons <- 3
output_neurons <- 1

# Pesos
w1 <- matrix(runif(input_neurons * hidden_neurons), nrow = input_neurons)
w2 <- matrix(runif(hidden_neurons * output_neurons), nrow = hidden_neurons)
b1 <- matrix(runif(hidden_neurons), nrow = 1)
b2 <- matrix(runif(output_neurons), nrow = 1)


# Taxa de aprendizado
learning_rate <- 0.075

# Número de épocas (qtd de iterações)
epochs <- 10

# Treinamento da rede
for(epoch in 1:epochs) {
  for(i in 1:length(x_train))
  {
    # Feedforward (propagação direta)
    input_data <- matrix(x_train[i], nrow = 1)
    hidden_input <- input_data %*% w1 + b1
    hidden_output <- linear(hidden_input) # ativação da camada escondida
     
    output_input <- hidden_output %*% w2 + b2
    output <- linear(output_input) # saída final da rede neural
    
    # Cálculo do erro
    error <- y_train[i] - output
    
    # Backpropagation
    output_delta <- error * linear_derivative(output) # Calculo do delta da camada de saída, que nos diz o quanto temos que ajustar
    # os pesos da camada de saída para reduzir o erro.
    
    hidden_error <- output_delta %*% t(w2) # Cálculo do erro retropropagado da camada escondida.Temos que transpor os pesos
    # que conectam a camada escondida a camada de saída por conta de ajuste a dimensão.
    
    hidden_delta <- hidden_error * linear_derivative(hidden_output) # Cálculo do delta da camada escondida, que nos diz o quanto precisamos ajustar
    # os pesos da camada escondida para reduzir o erro.
    
    # Atualização dos pesos
    w1 <- w1 + t(input_data) %*% hidden_delta * learning_rate
    w2 <- w2 + t(hidden_output) %*% output_delta * learning_rate
    b1 <- b1 + hidden_delta * learning_rate
    b2 <- b2 + output_delta * learning_rate
  }
}

# Previsões da rede treinada
predictions <- numeric(length(x_test))
for(i in 1:length(x_test)) 
{
  input_data <- matrix(x_test[i], nrow = 1)
  hidden_input <- input_data %*% w1 + b1
  hidden_output <- linear(hidden_input)
  output_input <- hidden_output %*% w2 + b2
  predictions[i] <- linear(output_input)
}

# Plot dos resultados
plot(x_train, y_train, col = 'blue', xlim = c(0, 2*pi),
     ylim = c(-1, 1), xlab = 'x', ylab = 'y')

lines(x_test, predictions, col = 'red') # desenha uma linha no gráfico, conectando os pontos representados por x_test e predictions.
# isso representa as previsões da rede em relação aos dados de teste.

legend(x = 4, y = 1, legend = c('Train', 'Test'),
       col = c('blue', 'red'), pch = c('o', '-'))

# Cálculo do erro médio quadrático
mse <- mean((predictions - y_test)^2)
cat("Erro médio quadrático (MSE):", mse, "\n")
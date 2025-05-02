package main

import (
    "fmt"
    "math"
    "time"
)

// Función sigmoide
func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}

// Predice la probabilidad de clase positiva
func predict(features []float64, weights []float64) float64 {
    z := 0.0
    for i := range features {
        z += features[i] * weights[i]
    }
    return sigmoid(z)
}

// Entrena el modelo usando descenso del gradiente
func train(X [][]float64, y []float64, alpha float64, epochs int) []float64 {
    m := len(y)
    n := len(X[0])
    weights := make([]float64, n)

    for epoch := 0; epoch < epochs; epoch++ {
        gradients := make([]float64, n)
        for i := 0; i < m; i++ {
            prediction := predict(X[i], weights)
            error := prediction - y[i]
            for j := 0; j < n; j++ {
                gradients[j] += error * X[i][j]
            }
        }
        for j := 0; j < n; j++ {
            weights[j] -= alpha * gradients[j] / float64(m)
        }
    }

    return weights
}

// Evalúa la precisión del modelo
func evaluate(X [][]float64, y []float64, weights []float64) float64 {
    correct := 0
    for i := 0; i < len(y); i++ {
        prob := predict(X[i], weights)
        pred := 0.0
        if prob >= 0.5 {
            pred = 1.0
        }
        if pred == y[i] {
            correct++
        }
    }
    return float64(correct) / float64(len(y))
}

func main() {
    // Datos de ejemplo (XOR simplificado no lineal)
    X := [][]float64{
        {1.0, 2.0},
        {1.0, 3.0},
        {2.0, 1.0},
        {3.0, 1.0},
    }
    y := []float64{0, 0, 1, 1}

    alpha := 0.1
    epochs := 1000

    // Medir tiempo de entrenamiento
    start := time.Now()
    weights := train(X, y, alpha, epochs)
    elapsed := time.Since(start)

    // Evaluar precisión
    accuracy := evaluate(X, y, weights)

    fmt.Println("Pesos aprendidos:", weights)
    fmt.Printf("Tiempo de entrenamiento: %s\n", elapsed)
    fmt.Printf("Precisión: %.2f%%\n", accuracy*100)

    // Prueba una predicción manual
    input := []float64{1.5, 2.0}
    prob := predict(input, weights)
    fmt.Printf("Probabilidad para %v: %.4f\n", input, prob)
}

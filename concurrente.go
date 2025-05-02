package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
)

// Sigmoid
func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}

// Predicción
func predict(x []float64, weights []float64) float64 {
    z := 0.0
    for i := range x {
        z += x[i] * weights[i]
    }
    return sigmoid(z)
}

// Evaluación de precisión
func evaluate(X [][]float64, y []float64, weights []float64) float64 {
    correct := 0
    for i := range X {
        pred := predict(X[i], weights)
        if (pred >= 0.5 && y[i] == 1.0) || (pred < 0.5 && y[i] == 0.0) {
            correct++
        }
    }
    return float64(correct) / float64(len(y)) * 100
}

// Entrenamiento concurrente
func trainConcurrent(X [][]float64, y []float64, alpha float64, epochs int, numWorkers int) []float64 {
    m := len(y)
    n := len(X[0])
    weights := make([]float64, n)

    type gradientChunk struct {
        grad []float64
    }

    for epoch := 0; epoch < epochs; epoch++ {
        ch := make(chan gradientChunk, numWorkers)
        chunkSize := (m + numWorkers - 1) / numWorkers

        for w := 0; w < numWorkers; w++ {
            start := w * chunkSize
            end := (w + 1) * chunkSize
            if end > m {
                end = m
            }

            go func(start, end int, weightsSnapshot []float64) {
                localGrad := make([]float64, n)
                for i := start; i < end; i++ {
                    pred := predict(X[i], weightsSnapshot)
                    err := pred - y[i]
                    for j := 0; j < n; j++ {
                        localGrad[j] += err * X[i][j]
                    }
                }
                ch <- gradientChunk{grad: localGrad}
            }(start, end, append([]float64(nil), weights...)) // copia segura de pesos
        }

        totalGrad := make([]float64, n)
        for w := 0; w < numWorkers; w++ {
            g := <-ch
            for j := 0; j < n; j++ {
                totalGrad[j] += g.grad[j]
            }
        }

        for j := 0; j < n; j++ {
            weights[j] -= alpha * totalGrad[j] / float64(m)
        }

        if epoch%(epochs/10) == 0 || epoch == epochs-1 {
            fmt.Printf("Entrenando (concurrente): %.0f%% completado\n", float64(epoch+1)/float64(epochs)*100)
        }
    }

    return weights
}

// Datos sintéticos separables
func generateSyntheticData(m int, n int) ([][]float64, []float64) {
    X := make([][]float64, m)
    y := make([]float64, m)

    trueWeights := make([]float64, n)
    for i := range trueWeights {
        trueWeights[i] = rand.Float64()*2 - 1
    }

    for i := 0; i < m; i++ {
        X[i] = make([]float64, n)
        z := 0.0
        for j := 0; j < n; j++ {
            val := rand.Float64()
            X[i][j] = val
            z += val * trueWeights[j]
        }

        prob := sigmoid(z)
        if rand.Float64() < prob {
            y[i] = 1.0
        } else {
            y[i] = 0.0
        }
    }

    return X, y
}

// Main
func main() {
    rand.Seed(time.Now().UnixNano())

    m := 1_000_000 // número de registros
    n := 50        // número de características
    alpha := 0.05
    epochs := 100
    numWorkers := 8 // número de goroutines

    fmt.Println("Generando datos sintéticos...")
    X, y := generateSyntheticData(m, n)

    fmt.Println("Entrenando modelo logístico concurrente...")
    start := time.Now()
    weights := trainConcurrent(X, y, alpha, epochs, numWorkers)
    duration := time.Since(start)

    fmt.Printf("Tiempo de entrenamiento: %v\n", duration)
    accuracy := evaluate(X, y, weights)
    fmt.Printf("Precisión final: %.2f%%\n", accuracy)

    fmt.Println("Primeros 10 pesos aprendidos:")
    for i := 0; i < 10 && i < len(weights); i++ {
        fmt.Printf("Peso %d: %.6f\n", i, weights[i])
    }
}

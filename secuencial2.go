package main

import (
    "bufio"
    "encoding/csv"
    "fmt"
    "math"
    "math/rand"
    "os"
    "strconv"
    "time"
)

func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}

func predict(features []float64, weights []float64) float64 {
    z := 0.0
    for i := range features {
        z += features[i] * weights[i]
    }
    return sigmoid(z)
}

func train(X [][]float64, y []float64, alpha float64, epochs int) []float64 {
    m := len(y)
    n := len(X[0])
    weights := make([]float64, n)

    for epoch := 0; epoch < epochs; epoch++ {
        gradients := make([]float64, n)
        for i := 0; i < m; i++ {
            pred := predict(X[i], weights)
            err := pred - y[i]
            for j := 0; j < n; j++ {
                gradients[j] += err * X[i][j]
            }
        }
        for j := 0; j < n; j++ {
            weights[j] -= alpha * gradients[j] / float64(m)
        }

        // Barra de progreso simple
        if epoch%(epochs/10) == 0 || epoch == epochs-1 {
            pct := float64(epoch+1) / float64(epochs) * 100
            fmt.Printf("Entrenando: %.0f%% completado\n", pct)
        }
    }

    return weights
}

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

func generateSyntheticData(m int, n int) ([][]float64, []float64) {
    X := make([][]float64, m)
    y := make([]float64, m)

    // Creamos un vector de pesos "reales" ocultos para simular la verdad
    trueWeights := make([]float64, n)
    for i := range trueWeights {
        trueWeights[i] = rand.Float64()*2 - 1 // valores entre -1 y 1
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


func readCSVData(path string) ([][]float64, []float64, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, nil, err
    }
    defer file.Close()

    reader := csv.NewReader(bufio.NewReader(file))
    var X [][]float64
    var y []float64

    for {
        record, err := reader.Read()
        if err != nil {
            break
        }

        features := make([]float64, len(record)-1)
        for i := 0; i < len(record)-1; i++ {
            val, _ := strconv.ParseFloat(record[i], 64)
            features[i] = val
        }
        label, _ := strconv.ParseFloat(record[len(record)-1], 64)

        X = append(X, features)
        y = append(y, label)
    }

    return X, y, nil
}

func main() {
    rand.Seed(time.Now().UnixNano())

    // Datos sintéticos
    fmt.Println("Generando datos sintéticos (1M muestras)...")
    X, y := generateSyntheticData(1000000, 10)

    // Entrenamiento
    fmt.Println("Entrenando modelo...")
    alpha := 0.05
    epochs := 100
    weights := train(X, y, alpha, epochs)

    // Evaluación
    fmt.Println("Evaluando precisión...")
    acc := evaluate(X, y, weights)
    fmt.Printf("Precisión final: %.2f%%\n", acc*100)

	// Mostrar algunos pesos
    fmt.Println("Primeros 10 pesos aprendidos:")
    for i := 0; i < 10 && i < len(weights); i++ {
       fmt.Printf("Peso %d: %.6f\n", i, weights[i])
    }

}

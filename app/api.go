package main

import "fmt"

func main() {
	numeros := []int{10, 20, 30, 40, 50}

	for i, n := range numeros {
		fmt.Printf("Posição: %d | Valor: %d\n", i, n)
	}
}

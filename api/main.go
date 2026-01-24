package main

type calculoRequest struct {
	A  float64 `json: "a"`
	B  float64 `json: "b"`
	Op string  `json: "operacao"`
}

type calculoResponse struct {
}

func main() {

}

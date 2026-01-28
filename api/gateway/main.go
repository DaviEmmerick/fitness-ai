package main

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

const PythonWorkerURL = "http://localhost:8001/predict"

func handleAnalyze(c *gin.Context) {

	fileHeader, err := c.FormFile("file")

	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Campo 'file' é obrigatório"})
		return
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Falha ao ler arquivo"})
		return
	}
	defer file.Close()

	bodyBuffer := &bytes.Buffer{}
	writer := multipart.NewWriter(bodyBuffer)

	part, err := writer.CreateFormFile("file", fileHeader.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Falha ao criar pacote multipart"})
		return
	}

	io.Copy(part, file)
	writer.Close()

	start := time.Now()

	resp, err := http.Post(PythonWorkerURL, writer.FormDataContentType(), bodyBuffer)

	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{
			"error": "Serviço de IA indisponível. Verifique se worker.py está rodando.",
		})
		return
	}
	defer resp.Body.Close()

	pythonResponse, _ := io.ReadAll(resp.Body)
	latency := time.Since(start).Seconds()

	fmt.Printf("Request processado em %.3fs | Status Python: %d\n", latency, resp.StatusCode)

	c.Data(resp.StatusCode, "application/json", pythonResponse)

}

func main() {
	r := gin.Default()

	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
	}))

	r.POST("/analyze", handleAnalyze)

	fmt.Println("Gym Gateway (Go) rodando na porta 8080...")
	fmt.Println("Conectado ao Worker Python em:", PythonWorkerURL)

	r.Run(":8080")
}
